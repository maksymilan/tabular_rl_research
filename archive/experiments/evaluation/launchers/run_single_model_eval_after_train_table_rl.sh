#!/usr/bin/env bash
# Wait for one training job, then evaluate epoch-2 and epoch-4 adapters sequentially on one GPU.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT_DIR/src/eval/run_qwen35_plan_eval_table_rl.sh}

TRAIN_PID=${TRAIN_PID:?set TRAIN_PID}
MODEL_LABEL=${MODEL_LABEL:?set MODEL_LABEL}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
EPOCH2_LORA=${EPOCH2_LORA:?set EPOCH2_LORA}
EPOCH4_LORA=${EPOCH4_LORA:?set EPOCH4_LORA}
SERVED_PREFIX=${SERVED_PREFIX:?set SERVED_PREFIX}
GPU_ID=${GPU_ID:?set GPU_ID}
VLLM_PORT=${VLLM_PORT:?set VLLM_PORT}
MAX_MODEL_LEN=${MAX_MODEL_LEN:?set MAX_MODEL_LEN}

MAX_STEPS=${MAX_STEPS:-30}
MAX_CONSECUTIVE_ERRORS=${MAX_CONSECUTIVE_ERRORS:-3}
MAX_TOKENS=${MAX_TOKENS:-768}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-3}
N_DEV=${N_DEV:-1034}
TABLE_OUTPUT_ROWS=${TABLE_OUTPUT_ROWS:-10}

RUN_ID=${RUN_ID:-${SERVED_PREFIX}_eval_after_train_$(date +%Y%m%d_%H%M%S)}
LOG="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT"
cd "$PROJECT_DIR"

require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "[$(date)] missing required adapter directory: $path"
    exit 1
  fi
}

run_one() {
  local epoch_label="$1"
  local lora_dir="$2"
  local served_model="${SERVED_PREFIX}_${epoch_label}"
  local result_dir="$OUT_ROOT/tool_zero_shot_dev1034_${epoch_label}"

  echo "[$(date)] starting $MODEL_LABEL $epoch_label eval on gpu=$GPU_ID"
  MODEL_DIR="$MODEL_DIR" \
  GPU_ID="$GPU_ID" VLLM_PORT="$VLLM_PORT" \
  LORA_DIR="$lora_dir" \
  SERVED_MODEL="$served_model" \
  RESULT_DIR="$result_dir" \
  RUN_ID="${served_model}_tool_zero_dev1034_$(date +%Y%m%d_%H%M%S)" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_STEPS="$MAX_STEPS" \
  MAX_CONSECUTIVE_ERRORS="$MAX_CONSECUTIVE_ERRORS" MAX_TOKENS="$MAX_TOKENS" \
  ROLLOUT_WORKERS="$ROLLOUT_WORKERS" N_DEV="$N_DEV" TABLE_OUTPUT_ROWS="$TABLE_OUTPUT_ROWS" \
  "$RUN_SCRIPT"

  echo "[$(date)] finished $MODEL_LABEL $epoch_label eval"
  test -f "$result_dir/summary.json" && cat "$result_dir/summary.json"
}

{
  echo "[$(date)] run_id=$RUN_ID"
  echo "[$(date)] waiting for $MODEL_LABEL training pid=$TRAIN_PID"
  while ps -p "$TRAIN_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date)] $MODEL_LABEL training pid=$TRAIN_PID exited"

  require_dir "$EPOCH2_LORA"
  require_dir "$EPOCH4_LORA"
  run_one epoch2 "$EPOCH2_LORA"
  run_one epoch4 "$EPOCH4_LORA"
  echo "[$(date)] $MODEL_LABEL epoch2/epoch4 evaluations finished"
} >> "$LOG" 2>&1 &

echo "run_id=$RUN_ID"
echo "pid=$!"
echo "log=$LOG"
