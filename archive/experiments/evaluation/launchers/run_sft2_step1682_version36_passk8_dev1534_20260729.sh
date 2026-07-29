#!/usr/bin/env bash
# Frozen SFT2 checkpoint-1682 K=8 BIRD-dev evaluation on the otherwise idle GPU1.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs
RESULT_DIR="$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_tool_version36_dev1534_passk8_bird_set"
ADAPTER="$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
SERVED_MODEL=qwen25-coder7b-sft2-step1682-version36-tool-dev1534-passk8
PORT=18052
EXPECTED=1534
STATUS="$OUTPUT_ROOT/logs/bird_sft2_step1682_v36_passk8.status"
RUN_LOG="$OUTPUT_ROOT/logs/bird_sft2_step1682_v36_passk8.run.log"
VLLM_LOG="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.log"
VLLM_PID_FILE="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.pid"
EVAL_LOG="$OUTPUT_ROOT/logs/$SERVED_MODEL.eval.log"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

completed_rows() {
  if [[ -f "$RESULT_DIR/all.jsonl" ]]; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

validate_complete() {
  /home/dengyan/miniconda3/envs/vllm-qwen35/bin/python - "$RESULT_DIR/all.jsonl" "$EXPECTED" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
assert len(rows) == expected, len(rows)
indices = [int(row["example_index"]) for row in rows]
assert len(set(indices)) == expected, len(set(indices))
infra = [
    row["example_index"]
    for row in rows
    if row.get("failure_type") in {
        "api_error",
        "transport_error",
        "provider_carrier_error",
    }
]
assert not infra, infra[:20]
PY
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"

if [[ "$(completed_rows)" -eq "$EXPECTED" ]]; then
  validate_complete
  set_status complete "already complete rows=$EXPECTED"
  exit 0
fi

for attempt in 1 2 3; do
  set_status running "attempt=$attempt rows=$(completed_rows) gpu=1 concurrency=24 k=8"
  set +e
  PROJECT_DIR="$RUNTIME" \
  PYTHON_BIN=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python \
  BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
  ADAPTER="$ADAPTER" \
  SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID=1 \
  PORT="$PORT" \
  RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON=data/eval_inputs/bird_dev_20240627.jsonl \
  N="$EXPECTED" \
  N_SAMPLES=8 \
  PASS_K=1,2,4,8 \
  WORKERS=3 \
  SAMPLE_WORKERS=8 \
  MAX_INFLIGHT=24 \
  MAX_NUM_SEQS=24 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 \
  MAX_TOKENS=1024 \
  TEMPERATURE=0.7 \
  TOP_P=0.95 \
  HISTORY_TURNS=4 \
  EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID=vllm-generation-config-vllm-sft2-step1682-version36-passk8 \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 \
  TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
  VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$VLLM_PID_FILE" \
  EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
  exit_code=$?
  set -e

  if [[ "$exit_code" -eq 0 ]]; then
    validate_complete
    set_status complete "rows=$EXPECTED result=$RESULT_DIR"
    exit 0
  fi
  set_status retrying "attempt=$attempt exit=$exit_code rows=$(completed_rows)"
  sleep 60
done

set_status failed "exhausted retries rows=$(completed_rows)"
exit 1
