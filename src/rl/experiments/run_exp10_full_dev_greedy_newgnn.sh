#!/usr/bin/env bash
# Full BIRD-dev deterministic greedy evaluation for Exp10 on NewGNN GPU7.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
EVAL_PYTHON=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER=${ADAPTER:-$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp10-rank-conservative-score-masked-lr1e6-23x4-gated-v3-20260731/final}
SERVED_MODEL=${SERVED_MODEL:-exp10-score-masked-full-dev-greedy}
RESULT_DIR=${RESULT_DIR:-$OUTPUT_ROOT/eval_runtime_version36_20260728/data/results/exp10_score_masked_version36_dev1534_greedy_t0_logprobs20_20260801}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.run.log}
VLLM_LOG=${VLLM_LOG:-$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.vllm.log}
VLLM_PID_FILE=${VLLM_PID_FILE:-$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.vllm.pid}
EVAL_LOG=${EVAL_LOG:-$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.eval.log}
GPU_ID=${GPU_ID:-7}
PORT=${PORT:-18077}
CONCURRENCY=${CONCURRENCY:-24}
EXPECTED=${EXPECTED:-1534}
FREE_MIB=${FREE_MIB:-512}
FREE_CHECKS=${FREE_CHECKS:-10}
FREE_POLL_SECONDS=${FREE_POLL_SECONDS:-30}
ADAPTER_SHA256=55012ded431de464f5ebc03eb753ac7a0cc929a6cacde17107becf4b3eff59c3

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
rows() {
  if [[ -f "$RESULT_DIR/all.jsonl" ]]; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

mkdir -p "$OUTPUT_ROOT/logs" "$RESULT_DIR"
exec 9>"$OUTPUT_ROOT/logs/exp10_full_dev_greedy_newgnn_20260801.lock"
if ! flock -n 9; then
  set_status duplicate "another queue instance owns the lock"
  exit 0
fi

[[ "$GPU_ID" == 7 ]] || { set_status failed "shared-server policy permits only GPU7 for this job"; exit 2; }
test -x "$EVAL_PYTHON"
test -d "$BASE_MODEL"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = "$ADAPTER_SHA256"
test -f "$RUNTIME/data/eval_inputs/bird_dev_20240627.jsonl"
test "$(grep -cve '^[[:space:]]*$' "$RUNTIME/data/eval_inputs/bird_dev_20240627.jsonl")" -eq "$EXPECTED"

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
assert len(expected_indices) == expected
assert {int(row["example_index"]) for row in rows} == expected_indices
assert all(len(row.get("samples") or []) == 1 for row in rows)
assert all(float(row.get("temperature")) == 0.0 for row in rows)
assert all(float(row.get("top_p")) == 1.0 for row in rows)
assert all(row.get("protocol_version") == "version36" for row in rows)
assert all(row.get("denotation_comparison") == "bird-set" for row in rows)
assert all(sample.get("generation_stats") for row in rows for sample in row["samples"])
manifest = json.loads((result_dir / "manifest.json").read_text())
accuracy = json.loads((result_dir / "accuracy.json").read_text())
summary = json.loads((result_dir / "summary.json").read_text())
assert manifest["n_samples"] == 1
assert manifest["temperature"] == 0.0
assert manifest["top_p"] == 1.0
assert manifest["protocol_version"] == "version36"
assert manifest["denotation_comparison"] == "bird-set"
assert manifest["record_logprobs"] is True
assert accuracy["pass@1"]["total"] == expected
assert summary["total"] == expected
for name in (
    "accuracy", "valid_rate", "avg_steps", "per_question_result",
    "trajectory", "trajectory_entropy", "action_distribution",
):
    assert (result_dir / f"{name}.json").is_file(), name
PY
}

if [[ "$(rows)" -eq "$EXPECTED" ]]; then
  cd "$RUNTIME"
  "$EVAL_PYTHON" src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
  validate_complete
  set_status complete "already complete rows=$EXPECTED"
  exit 0
fi

consecutive=0
while [[ "$consecutive" -lt "$FREE_CHECKS" ]]; do
  memory=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  compute=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')
  if [[ "$memory" -le "$FREE_MIB" && "$compute" -eq 0 ]]; then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  set_status waiting_gpu "gpu=$GPU_ID free_checks=$consecutive/$FREE_CHECKS memory_mib=$memory compute=$compute rows=$(rows)"
  [[ "$consecutive" -eq "$FREE_CHECKS" ]] || sleep "$FREE_POLL_SECONDS"
done

# Close the check/start race as far as nvidia-smi permits without a cluster scheduler.
memory=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
compute=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')
[[ "$memory" -le "$FREE_MIB" && "$compute" -eq 0 ]] || {
  set_status failed "GPU7 became occupied at launch memory_mib=$memory compute=$compute"
  exit 3
}

for attempt in 1 2 3; do
  set_status running "attempt=$attempt gpu=$GPU_ID rows=$(rows)/$EXPECTED concurrency=$CONCURRENCY"
  set +e
  cd "$RUNTIME"
  PROJECT_DIR="$RUNTIME" VLLM_PYTHON="$EVAL_PYTHON" PYTHON_BIN="$EVAL_PYTHON" \
  BASE_MODEL="$BASE_MODEL" ADAPTER="$ADAPTER" SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID="$GPU_ID" PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON=data/eval_inputs/bird_dev_20240627.jsonl INDICES_FILE= \
  N="$EXPECTED" N_SAMPLES=1 PASS_K=1 WORKERS="$CONCURRENCY" SAMPLE_WORKERS=1 \
  MAX_INFLIGHT="$CONCURRENCY" MAX_NUM_SEQS="$CONCURRENCY" \
  MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 MAX_TOKENS=1024 TEMPERATURE=0 TOP_P=1 \
  RECORD_LOGPROBS=1 TOP_LOGPROBS=20 HISTORY_TURNS=4 EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID=vllm-version36-exp10-full-dev-greedy-c24-logprobs20 \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$VLLM_PID_FILE" EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh >>"$RUN_LOG" 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    validate_complete
    set_status complete "rows=$EXPECTED result=$RESULT_DIR concurrency=$CONCURRENCY"
    exit 0
  fi
  set_status retrying "attempt=$attempt exit=$code rows=$(rows)/$EXPECTED"
  sleep 30
done

set_status failed "exhausted retries rows=$(rows)/$EXPECTED"
exit 1
