#!/usr/bin/env bash
# Matched K=4 evaluation of the new TRL result-only checkpoint.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RESULT_DIR=${RESULT_DIR:-"$RUNTIME/data/results/qwen25_coder7b_sft2_trl_resultonly_lr1e6_step23_tool_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set"}
ADAPTER=${ADAPTER:-"$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase1-result-only-lr1e6-23x4-20260729/final"}
SERVED_MODEL=${SERVED_MODEL:-qwen25-coder7b-sft2-trl-resultonly-lr1e6-step23-version36-equal300-passk4-logprobs20}
SERVER_CONFIG_ID=${SERVER_CONFIG_ID:-vllm-generation-config-vllm-trl-resultonly-version36-equal300-passk4-logprobs20}
INDICES_FILE=${INDICES_FILE:-data/eval_inputs/bird_dev_equal_difficulty300_seed20260729.indices.json}
PORT=${PORT:-18057}
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
EVAL_PYTHON=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}
EXPECTED=300
STATUS=${STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_equal300_passk4_20260729.status"}
RUN_LOG=${RUN_LOG:-"$OUTPUT_ROOT/logs/phase1_result_only_equal300_passk4_20260729.run.log"}
VLLM_LOG=${VLLM_LOG:-"$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.log"}
VLLM_PID_FILE=${VLLM_PID_FILE:-"$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.pid"}
EVAL_LOG=${EVAL_LOG:-"$OUTPUT_ROOT/logs/$SERVED_MODEL.eval.log"}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
completed_rows() {
  if [[ -f "$RESULT_DIR/all.jsonl" ]]; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}
wait_for_eval_gpu() {
  local gpu_ready=0
  local -a gpu_memory=()
  for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
    mapfile -t gpu_memory < <(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    if [[ "$EVAL_GPU_ID" -lt "${#gpu_memory[@]}" ]] \
      && [[ "${gpu_memory[$EVAL_GPU_ID]}" -le "$GPU_FREE_THRESHOLD_MIB" ]]; then
      gpu_ready=1
      break
    fi
    set_status waiting_gpu \
      "eval_gpu=$EVAL_GPU_ID mib=${gpu_memory[$EVAL_GPU_ID]:-unknown} rows=$(completed_rows)"
    sleep 30
  done
  if [[ "$gpu_ready" -ne 1 ]]; then
    set_status failed "timed out waiting for eval GPU $EVAL_GPU_ID"
    return 1
  fi
}

validate_complete() {
  "$EVAL_PYTHON" - \
    "$RESULT_DIR" "$RUNTIME/$INDICES_FILE" "$EXPECTED" <<'PY'
import json
import sys
from pathlib import Path

result_dir, selection_path, expected_text = sys.argv[1:]
result_dir = Path(result_dir)
expected = int(expected_text)
rows = [json.loads(line) for line in (result_dir / "all.jsonl").open() if line.strip()]
selected = set(json.load(open(selection_path))["indices"])
assert len(rows) == expected
assert {int(row["example_index"]) for row in rows} == selected
assert all(len(row.get("samples") or []) == 4 for row in rows)
assert all(sample.get("generation_stats") for row in rows for sample in row["samples"])
manifest = json.load((result_dir / "manifest.json").open())
assert manifest["protocol_version"] == "version36"
assert manifest["n_samples"] == 4
assert manifest["pass_k"] == [1, 2, 4]
assert manifest["temperature"] == 0.7
assert manifest["top_p"] == 0.95
assert manifest["record_logprobs"] is True
assert manifest["top_logprobs"] == 20
assert manifest["denotation_comparison"] == "bird-set"
for name in ("accuracy", "valid_rate", "avg_steps", "per_question_result", "trajectory", "trajectory_entropy", "action_distribution"):
    assert (result_dir / f"{name}.json").is_file(), name
PY
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$INDICES_FILE"
test -x "$EVAL_PYTHON"
test -d "$BASE_MODEL"
[[ "$EVAL_GPU_ID" =~ ^[0-9]+$ ]] || { set_status failed "invalid EVAL_GPU_ID"; exit 1; }

if [[ "$(completed_rows)" -eq "$EXPECTED" ]]; then
  /home/dengyan/miniconda3/envs/vllm-qwen35/bin/python \
    src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
  validate_complete
  set_status complete "already complete rows=$EXPECTED"
  exit 0
fi

for attempt in 1 2 3; do
  wait_for_eval_gpu
  set_status running "attempt=$attempt rows=$(completed_rows) gpu=$EVAL_GPU_ID k=4 logprobs=top20"
  set +e
  PROJECT_DIR="$RUNTIME" \
  VLLM_PYTHON="$EVAL_PYTHON" \
  PYTHON_BIN="$EVAL_PYTHON" \
  BASE_MODEL="$BASE_MODEL" \
  ADAPTER="$ADAPTER" \
  SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID="$EVAL_GPU_ID" PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON=data/eval_inputs/bird_dev_20240627.jsonl \
  INDICES_FILE="$INDICES_FILE" N=1534 N_SAMPLES=4 PASS_K=1,2,4 \
  WORKERS=6 SAMPLE_WORKERS=4 MAX_INFLIGHT=24 MAX_NUM_SEQS=24 \
  MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 MAX_TOKENS=1024 TEMPERATURE=0.7 TOP_P=0.95 \
  RECORD_LOGPROBS=1 TOP_LOGPROBS=20 HISTORY_TURNS=4 EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID="$SERVER_CONFIG_ID" \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
  VLLM_LOG="$VLLM_LOG" VLLM_PID_FILE="$VLLM_PID_FILE" EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
  exit_code=$?
  set -e
  if [[ "$exit_code" -eq 0 ]]; then
    validate_complete
    set_status complete "rows=$EXPECTED result=$RESULT_DIR"
    exit 0
  fi
  set_status retrying "attempt=$attempt exit=$exit_code rows=$(completed_rows)"
  sleep 30
done
set_status failed "exhausted retries rows=$(completed_rows)"
exit 1
