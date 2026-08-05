#!/usr/bin/env bash
# Deterministic greedy evaluation over the complete 1,534-question BIRD-dev set.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
EVAL_PYTHON=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
PORT=${PORT:-18071}
EXPECTED=${EXPECTED:-1534}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARDWARE_PROFILE=${HARDWARE_PROFILE:-$SCRIPT_DIR/../configs/hardware/rtx3090_24gb.sh}
test -f "$HARDWARE_PROFILE"
# shellcheck source=../configs/hardware/rtx3090_24gb.sh
source "$HARDWARE_PROFILE"

for name in ADAPTER SERVED_MODEL RESULT_DIR STATUS RUN_LOG; do
  test -n "${!name:-}" || { printf 'missing required variable: %s\n' "$name" >&2; exit 2; }
done
VLLM_LOG=${VLLM_LOG:-"$OUTPUT_ROOT/logs/${SERVED_MODEL}.vllm.log"}
VLLM_PID_FILE=${VLLM_PID_FILE:-"$OUTPUT_ROOT/logs/${SERVED_MODEL}.vllm.pid"}
EVAL_LOG=${EVAL_LOG:-"$OUTPUT_ROOT/logs/${SERVED_MODEL}.eval.log"}
# Do not reuse the queue-level LOCK variable inherited from parent launchers.
EVAL_LOCK=${EVAL_LOCK:-"$STATUS.lock"}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
completed_rows() {
  if [[ -f "$RESULT_DIR/all.jsonl" ]]; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}
wait_for_gpu() {
  local -a memory=()
  for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
    mapfile -t memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$EVAL_GPU_ID" -lt "${#memory[@]}" ]] \
      && [[ "${memory[$EVAL_GPU_ID]}" -le "$GPU_FREE_THRESHOLD_MIB" ]]; then
      return 0
    fi
    set_status waiting_gpu "gpu=$EVAL_GPU_ID mib=${memory[$EVAL_GPU_ID]:-unknown} rows=$(completed_rows)/$EXPECTED"
    sleep 30
  done
  set_status failed "timed out waiting for GPU $EVAL_GPU_ID"
  return 1
}
validate_complete() {
  "$EVAL_PYTHON" - "$RESULT_DIR" "$RUNTIME/data/eval_inputs/bird_dev_20240627.jsonl" "$EXPECTED" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
examples_path = Path(sys.argv[2])
expected = int(sys.argv[3])
rows = [json.loads(line) for line in (result_dir / "all.jsonl").open() if line.strip()]
examples = [json.loads(line) for line in examples_path.open() if line.strip()]
expected_indices = {int(row["example_index"]) for row in examples}
assert len(rows) == expected == len(examples)
assert {int(row["example_index"]) for row in rows} == expected_indices
assert all(len(row.get("samples") or []) == 1 for row in rows)
assert all(float(row.get("temperature")) == 0.0 for row in rows)
assert all(float(row.get("top_p")) == 1.0 for row in rows)
assert all(row.get("protocol_version") == "version36" for row in rows)
assert all(row.get("denotation_comparison") == "bird-set" for row in rows)
assert all(sample.get("generation_stats") for row in rows for sample in row["samples"])
protocol = json.loads((result_dir / "evaluation_protocol.json").read_text())
greedy = json.loads((result_dir / "greedy_at_1.json").read_text())
assert protocol["mode"] == "greedy"
assert greedy["available"] is True
for name in (
    "sample0_at_1", "trajectory_accuracy", "correct_count_distribution",
    "failure_types", "valid_rate", "avg_steps", "per_question_result",
    "trajectory", "trajectory_entropy", "action_distribution",
):
    assert (result_dir / f"{name}.json").is_file(), name
PY
}

mkdir -p "$OUTPUT_ROOT/logs" "$RESULT_DIR"
exec >>"$RUN_LOG" 2>&1
exec 9>"$EVAL_LOCK"
if ! flock -n 9; then
  while [[ "$(state_of "$STATUS")" != complete ]]; do sleep 30; done
  exit 0
fi
cd "$RUNTIME"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
test -d "$BASE_MODEL"
test -f data/eval_inputs/bird_dev_20240627.jsonl
test "$(grep -cve '^[[:space:]]*$' data/eval_inputs/bird_dev_20240627.jsonl)" -eq "$EXPECTED"
if [[ -n "${ADAPTER_SHA256:-}" ]]; then
  test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = "$ADAPTER_SHA256"
fi

if [[ "$(completed_rows)" -eq "$EXPECTED" ]]; then
  "$EVAL_PYTHON" src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
  validate_complete
  set_status complete "already complete rows=$EXPECTED"
  exit 0
fi

for attempt in 1 2 3; do
  wait_for_gpu
  set_status running "attempt=$attempt rows=$(completed_rows)/$EXPECTED gpu=$EVAL_GPU_ID greedy=1 concurrency=$RTX3090_GREEDY_WORKERS"
  set +e
  PROJECT_DIR="$RUNTIME" VLLM_PYTHON="$EVAL_PYTHON" PYTHON_BIN="$EVAL_PYTHON" \
  BASE_MODEL="$BASE_MODEL" ADAPTER="$ADAPTER" SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID="$EVAL_GPU_ID" PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON=data/eval_inputs/bird_dev_20240627.jsonl INDICES_FILE= \
  N="$EXPECTED" N_SAMPLES=1 PASS_K=1 WORKERS="$RTX3090_GREEDY_WORKERS" \
  SAMPLE_WORKERS="$RTX3090_GREEDY_SAMPLE_WORKERS" \
  MAX_INFLIGHT="$RTX3090_EVAL_MAX_INFLIGHT" \
  MAX_NUM_SEQS="$RTX3090_EVAL_MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$RTX3090_EVAL_MAX_NUM_BATCHED_TOKENS" \
  MAX_MODEL_LEN="$RTX3090_EVAL_MAX_MODEL_LEN" \
  GPU_MEMORY_UTILIZATION="$RTX3090_EVAL_GPU_MEMORY_UTILIZATION" \
  MAX_STEPS=30 MAX_TOKENS=1024 TEMPERATURE=0 TOP_P=1 \
  RECORD_LOGPROBS=1 TOP_LOGPROBS=20 HISTORY_TURNS=4 EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID="${SERVER_CONFIG_ID:-vllm-version36-full-dev-greedy-c24-logprobs20}" \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$VLLM_PID_FILE" EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
  exit_code=$?
  set -e
  if [[ "$exit_code" -eq 0 ]]; then
    "$EVAL_PYTHON" src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
    validate_complete
    set_status complete "rows=$EXPECTED result=$RESULT_DIR"
    exit 0
  fi
  set_status retrying "attempt=$attempt exit=$exit_code rows=$(completed_rows)/$EXPECTED"
  sleep 30
done
set_status failed "exhausted retries rows=$(completed_rows)/$EXPECTED"
exit 1
