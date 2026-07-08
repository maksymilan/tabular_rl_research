#!/usr/bin/env bash
# Wait for the Qwen2.5-7B v9 training job, then evaluate epoch-2 and epoch-4
# adapters concurrently on the two table_rl GPUs.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
TRAIN_PID=${TRAIN_PID:?set TRAIN_PID to the launcher pid}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen2.5-7B-Instruct}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT_DIR/src/eval/run_qwen35_plan_eval_table_rl.sh}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/data/results/qwen2.5_7b_sft_v9_ready_988_epoch4}

EPOCH4_LORA=${EPOCH4_LORA:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-spider-v9-ready-988-epoch4-qlora}
EPOCH2_LORA=${EPOCH2_LORA:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-spider-v9-ready-988-epoch2-qlora}

RUN_ID=${RUN_ID:-qwen25_7b_v9_eval_after_train_$(date +%Y%m%d_%H%M%S)}
LOG="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

{
  echo "[$(date)] waiting for training pid=$TRAIN_PID"
  while ps -p "$TRAIN_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date)] training pid=$TRAIN_PID exited"

  if [ ! -d "$EPOCH2_LORA" ]; then
    echo "[$(date)] missing epoch2 adapter: $EPOCH2_LORA"
    exit 1
  fi
  if [ ! -d "$EPOCH4_LORA" ]; then
    echo "[$(date)] missing epoch4 adapter: $EPOCH4_LORA"
    exit 1
  fi

  echo "[$(date)] starting epoch2 and epoch4 evaluations"

  MODEL_DIR="$MODEL_DIR" \
  GPU_ID=0 VLLM_PORT=8010 \
  LORA_DIR="$EPOCH2_LORA" \
  SERVED_MODEL=qwen25_7b_v9_e2 \
  RESULT_DIR="$OUT_ROOT/tool_zero_shot_dev1034_epoch2" \
  RUN_ID=qwen25_7b_v9_e2_tool_zero_dev1034_$(date +%Y%m%d_%H%M%S) \
  MAX_MODEL_LEN=8192 MAX_STEPS=20 MAX_CONSECUTIVE_ERRORS=3 MAX_TOKENS=768 \
  ROLLOUT_WORKERS=3 N_DEV=1034 TABLE_OUTPUT_ROWS=0 \
  "$RUN_SCRIPT" &
  PID_E2=$!

  MODEL_DIR="$MODEL_DIR" \
  GPU_ID=1 VLLM_PORT=8011 \
  LORA_DIR="$EPOCH4_LORA" \
  SERVED_MODEL=qwen25_7b_v9_e4 \
  RESULT_DIR="$OUT_ROOT/tool_zero_shot_dev1034_epoch4" \
  RUN_ID=qwen25_7b_v9_e4_tool_zero_dev1034_$(date +%Y%m%d_%H%M%S) \
  MAX_MODEL_LEN=8192 MAX_STEPS=20 MAX_CONSECUTIVE_ERRORS=3 MAX_TOKENS=768 \
  ROLLOUT_WORKERS=3 N_DEV=1034 TABLE_OUTPUT_ROWS=0 \
  "$RUN_SCRIPT" &
  PID_E4=$!

  set +e
  wait "$PID_E2"
  status_e2=$?
  wait "$PID_E4"
  status_e4=$?
  set -e

  echo "[$(date)] epoch2 eval status=$status_e2"
  test -f "$OUT_ROOT/tool_zero_shot_dev1034_epoch2/summary.json" && cat "$OUT_ROOT/tool_zero_shot_dev1034_epoch2/summary.json"
  echo "[$(date)] epoch4 eval status=$status_e4"
  test -f "$OUT_ROOT/tool_zero_shot_dev1034_epoch4/summary.json" && cat "$OUT_ROOT/tool_zero_shot_dev1034_epoch4/summary.json"

  if [ "$status_e2" -ne 0 ] || [ "$status_e4" -ne 0 ]; then
    exit 1
  fi
} >> "$LOG" 2>&1 &

echo "run_id=$RUN_ID"
echo "pid=$!"
echo "log=$LOG"
