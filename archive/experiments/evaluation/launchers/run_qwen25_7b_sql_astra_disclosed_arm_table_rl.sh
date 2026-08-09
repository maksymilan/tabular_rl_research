#!/usr/bin/env bash
set -euo pipefail

# Controlled full-dev arm for the SQL-ASTRA Qwen2.5-7B-Instruct baseline reproduction.
# The caller supplies a unique label and decoding values; all other factors stay frozen.

PROJECT_DIR="${PROJECT_DIR:-/home/dengyan/tabular_rl_outputs/sql_astra_reproduction_matrix_20260805}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/sql_astra_reproduction_matrix_20260805}"
LOG_ROOT="${LOG_ROOT:-/home/dengyan/tabular_rl_outputs/logs/sql_astra_reproduction_matrix_20260805}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
PORT="${PORT:?PORT is required}"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required}"
TEMPERATURE="${TEMPERATURE:?TEMPERATURE is required}"
TOP_P="${TOP_P:?TOP_P is required}"
SMOKE_N="${SMOKE_N:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen2.5-7B-Instruct}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/home/dengyan/tabular_rl_outputs/bird_dev_eval/data}"
TASKS_JSON="${TASKS_JSON:-$EVAL_DATA_ROOT/eval_inputs/bird_dev_20240627.jsonl}"
SCHEMA_METADATA_JSON="${SCHEMA_METADATA_JSON:-$EVAL_DATA_ROOT/bird/dev_20240627/dev_tables.json}"
RUNNER="$PROJECT_DIR/src/eval/run_bird_direct_sql_passk_local_gpu.sh"
RUN_STEM="qwen2.5-7b-instruct_sql-astra-disclosed-single-turn-v1_${RUN_LABEL}_bird-set"
QUEUE_LOG="$LOG_ROOT/$RUN_STEM.queue.log"
FULL_RESULT="$OUTPUT_ROOT/${RUN_STEM}_dev1534"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec >>"$QUEUE_LOG" 2>&1

test -f "$TASKS_JSON" || {
  echo "Missing BIRD task file: $TASKS_JSON" >&2
  exit 2
}
test -f "$SCHEMA_METADATA_JSON" || {
  echo "Missing BIRD schema metadata: $SCHEMA_METADATA_JSON" >&2
  exit 2
}

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

gpu_memory_used_mib() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits |
    tr -d '[:space:]'
}

wait_for_gpu() {
  local stable=0
  local used
  while test "$stable" -lt 10; do
    used="$(gpu_memory_used_mib)"
    if test -n "$used" && test "$used" -lt 512; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 6
  done
}

run_eval() {
  local count="$1"
  local suffix="$2"
  local result_dir="$3"
  env \
    PROJECT_DIR="$PROJECT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    BASE_MODEL="$BASE_MODEL" \
    TASKS_JSON="$TASKS_JSON" \
    GPU_ID="$GPU_ID" \
    PORT="$PORT" \
    N_SAMPLES=1 \
    PASS_K=1 \
    WORKERS=12 \
    MAX_TOKENS=2048 \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    REPETITION_PENALTY=1 \
    PROMPT_PROFILE=sql-astra-disclosed-single-turn-v1 \
    SCHEMA_VALUE_COUNT=2 \
    SCHEMA_METADATA_JSON="$SCHEMA_METADATA_JSON" \
    EXECUTION_TIMEOUT_SECONDS=10 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=24 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    N="$count" \
    SERVED_MODEL="${RUN_STEM}_${suffix}" \
    RESULT_DIR="$result_dir" \
    VLLM_LOG="$LOG_ROOT/${RUN_STEM}.${suffix}.vllm.log" \
    VLLM_PID_FILE="$LOG_ROOT/${RUN_STEM}.${suffix}.vllm.pid" \
    EVAL_LOG="$LOG_ROOT/${RUN_STEM}.${suffix}.eval.log" \
    bash "$RUNNER"
}

echo "$(timestamp) queued $RUN_LABEL on GPU $GPU_ID: temperature=$TEMPERATURE top_p=$TOP_P"
wait_for_gpu

if test "$SMOKE_N" -gt 0; then
  SMOKE_RESULT="$OUTPUT_ROOT/${RUN_STEM}_smoke${SMOKE_N}"
  echo "$(timestamp) starting smoke-$SMOKE_N"
  run_eval "$SMOKE_N" "smoke${SMOKE_N}" "$SMOKE_RESULT"
  "$PYTHON_BIN" -c \
    'import json,sys; from pathlib import Path; rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]; assert len(rows)==int(sys.argv[2]); assert not any(r.get("failure_type") in {"api_error","context_overflow","incomplete_api_response"} for r in rows)' \
    "$SMOKE_RESULT/all.jsonl" "$SMOKE_N"
fi

echo "$(timestamp) starting full BIRD-dev"
run_eval 1534 "dev1534" "$FULL_RESULT"
echo "$(timestamp) completed $RUN_LABEL: $FULL_RESULT"
