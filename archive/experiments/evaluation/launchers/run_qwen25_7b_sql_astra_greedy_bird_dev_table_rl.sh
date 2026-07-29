#!/usr/bin/env bash
set -euo pipefail

# One-off reproduction of SQL-ASTRA's disclosed Qwen2.5-7B-Instruct BIRD baseline input.
# This launcher is deployed into an isolated table_rl runtime and waits for GPU 0 to be released.

PROJECT_DIR="${PROJECT_DIR:-/home/dengyan/tabular_rl_outputs/sql_astra_baseline_runtime_20260729}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations}"
LOG_ROOT="${LOG_ROOT:-/home/dengyan/tabular_rl_outputs/logs}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8038}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen2.5-7B-Instruct}"
SCHEMA_METADATA_JSON="${SCHEMA_METADATA_JSON:-$PROJECT_DIR/data/bird/dev_20240627/dev_tables.json}"
RUNNER="$PROJECT_DIR/src/eval/run_bird_direct_sql_passk_local_gpu.sh"
RUN_STEM="qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set"
QUEUE_LOG="$LOG_ROOT/$RUN_STEM.queue.log"
SMOKE_RESULT="$OUTPUT_ROOT/${RUN_STEM}_smoke32"
FULL_RESULT="$OUTPUT_ROOT/${RUN_STEM}_dev1534"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec >>"$QUEUE_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

gpu_memory_used_mib() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits |
    tr -d '[:space:]'
}

echo "$(timestamp) queued SQL-ASTRA prompt-profile baseline on GPU $GPU_ID"
while true; do
  used="$(gpu_memory_used_mib)"
  if test -n "$used" && test "$used" -lt 2048; then
    break
  fi
  echo "$(timestamp) waiting for GPU $GPU_ID; memory.used=${used:-unknown} MiB"
  sleep 30
done

COMMON_ENV=(
  PROJECT_DIR="$PROJECT_DIR"
  PYTHON_BIN="$PYTHON_BIN"
  BASE_MODEL="$BASE_MODEL"
  GPU_ID="$GPU_ID"
  PORT="$PORT"
  N_SAMPLES=1
  PASS_K=1
  WORKERS=6
  MAX_TOKENS=2048
  TEMPERATURE=0
  TOP_P=1
  REPETITION_PENALTY=1
  PROMPT_PROFILE=sql-astra-appendix-v1
  SCHEMA_VALUE_COUNT=2
  SCHEMA_METADATA_JSON="$SCHEMA_METADATA_JSON"
  EXECUTION_TIMEOUT_SECONDS=10
  MAX_MODEL_LEN=8192
  MAX_NUM_SEQS=16
  MAX_NUM_BATCHED_TOKENS=16384
)

echo "$(timestamp) GPU free; starting smoke-32"
env "${COMMON_ENV[@]}" \
  N=32 \
  SERVED_MODEL="${RUN_STEM}_smoke32" \
  RESULT_DIR="$SMOKE_RESULT" \
  VLLM_LOG="$LOG_ROOT/${RUN_STEM}.smoke32.vllm.log" \
  VLLM_PID_FILE="$LOG_ROOT/${RUN_STEM}.smoke32.vllm.pid" \
  EVAL_LOG="$LOG_ROOT/${RUN_STEM}.smoke32.eval.log" \
  bash "$RUNNER"

"$PYTHON_BIN" -c \
  'import json,sys; from pathlib import Path; rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]; assert len(rows)==32; assert not any(r.get("failure_type") in {"api_error","context_overflow","incomplete_api_response"} for r in rows)' \
  "$SMOKE_RESULT/all.jsonl"

echo "$(timestamp) smoke passed; starting full BIRD dev-1534"
env "${COMMON_ENV[@]}" \
  N=1534 \
  SERVED_MODEL="$RUN_STEM" \
  RESULT_DIR="$FULL_RESULT" \
  VLLM_LOG="$LOG_ROOT/${RUN_STEM}.vllm.log" \
  VLLM_PID_FILE="$LOG_ROOT/${RUN_STEM}.vllm.pid" \
  EVAL_LOG="$LOG_ROOT/${RUN_STEM}.eval.log" \
  bash "$RUNNER"

echo "$(timestamp) SQL-ASTRA prompt-profile baseline complete: $FULL_RESULT"
