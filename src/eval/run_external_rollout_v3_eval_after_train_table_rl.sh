#!/usr/bin/env bash
# Wait for both external-rollout v3 SFT jobs, then evaluate epoch-2 and epoch-4 adapters.
# Evaluations run in two waves to avoid competing with an unfinished training process.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT_DIR/src/eval/run_qwen35_plan_eval_table_rl.sh}

TRAIN_PID_7B=${TRAIN_PID_7B:?set TRAIN_PID_7B}
TRAIN_PID_9B=${TRAIN_PID_9B:?set TRAIN_PID_9B}

MODEL_DIR_7B=${MODEL_DIR_7B:-/home/dengyan/models/Qwen2.5-7B-Instruct}
MODEL_DIR_9B=${MODEL_DIR_9B:-/home/dengyan/models/Qwen3.5-9B}

OUT_ROOT_7B=${OUT_ROOT_7B:-$PROJECT_DIR/data/results/qwen2.5_7b_sft_external_rollout_v3_epoch4}
OUT_ROOT_9B=${OUT_ROOT_9B:-$PROJECT_DIR/data/results/qwen3.5_9b_sft_external_rollout_v3_4k_epoch4}

EPOCH4_LORA_7B=${EPOCH4_LORA_7B:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch4-qlora}
EPOCH2_LORA_7B=${EPOCH2_LORA_7B:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch2-qlora}
EPOCH4_LORA_9B=${EPOCH4_LORA_9B:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-external-rollout-v3-success831-4k-epoch4-qlora}
EPOCH2_LORA_9B=${EPOCH2_LORA_9B:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-external-rollout-v3-success831-4k-epoch2-qlora}

RUN_ID=${RUN_ID:-external_rollout_v3_eval_after_train_$(date +%Y%m%d_%H%M%S)}
LOG="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT_7B" "$OUT_ROOT_9B"
cd "$PROJECT_DIR"

wait_for_pid() {
  local pid="$1"
  local label="$2"
  echo "[$(date)] waiting for $label training pid=$pid"
  while ps -p "$pid" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date)] $label training pid=$pid exited"
}

require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "[$(date)] missing required adapter directory: $path"
    exit 1
  fi
}

run_pair() {
  local model_label="$1"
  local model_dir="$2"
  local out_root="$3"
  local epoch2_lora="$4"
  local epoch4_lora="$5"
  local served_prefix="$6"
  local max_model_len="$7"

  require_dir "$epoch2_lora"
  require_dir "$epoch4_lora"

  echo "[$(date)] starting $model_label epoch2 and epoch4 evaluations"

  MODEL_DIR="$model_dir" \
  GPU_ID=0 VLLM_PORT=8010 \
  LORA_DIR="$epoch2_lora" \
  SERVED_MODEL="${served_prefix}_e2" \
  RESULT_DIR="$out_root/tool_zero_shot_dev1034_epoch2" \
  RUN_ID="${served_prefix}_e2_tool_zero_dev1034_$(date +%Y%m%d_%H%M%S)" \
  MAX_MODEL_LEN="$max_model_len" MAX_STEPS=30 MAX_CONSECUTIVE_ERRORS=3 MAX_TOKENS=768 \
  ROLLOUT_WORKERS=3 N_DEV=1034 TABLE_OUTPUT_ROWS=10 \
  "$RUN_SCRIPT" &
  local pid_e2=$!

  MODEL_DIR="$model_dir" \
  GPU_ID=1 VLLM_PORT=8011 \
  LORA_DIR="$epoch4_lora" \
  SERVED_MODEL="${served_prefix}_e4" \
  RESULT_DIR="$out_root/tool_zero_shot_dev1034_epoch4" \
  RUN_ID="${served_prefix}_e4_tool_zero_dev1034_$(date +%Y%m%d_%H%M%S)" \
  MAX_MODEL_LEN="$max_model_len" MAX_STEPS=30 MAX_CONSECUTIVE_ERRORS=3 MAX_TOKENS=768 \
  ROLLOUT_WORKERS=3 N_DEV=1034 TABLE_OUTPUT_ROWS=10 \
  "$RUN_SCRIPT" &
  local pid_e4=$!

  set +e
  wait "$pid_e2"
  local status_e2=$?
  wait "$pid_e4"
  local status_e4=$?
  set -e

  echo "[$(date)] $model_label epoch2 eval status=$status_e2"
  test -f "$out_root/tool_zero_shot_dev1034_epoch2/summary.json" && cat "$out_root/tool_zero_shot_dev1034_epoch2/summary.json"
  echo "[$(date)] $model_label epoch4 eval status=$status_e4"
  test -f "$out_root/tool_zero_shot_dev1034_epoch4/summary.json" && cat "$out_root/tool_zero_shot_dev1034_epoch4/summary.json"

  if [ "$status_e2" -ne 0 ] || [ "$status_e4" -ne 0 ]; then
    exit 1
  fi
}

{
  echo "[$(date)] run_id=$RUN_ID"
  wait_for_pid "$TRAIN_PID_7B" "7B"
  wait_for_pid "$TRAIN_PID_9B" "9B"
  run_pair "7B" "$MODEL_DIR_7B" "$OUT_ROOT_7B" "$EPOCH2_LORA_7B" "$EPOCH4_LORA_7B" "qwen25_7b_ext_v3" 8192
  run_pair "9B" "$MODEL_DIR_9B" "$OUT_ROOT_9B" "$EPOCH2_LORA_9B" "$EPOCH4_LORA_9B" "qwen35_ext_v3_4k" 4096
  echo "[$(date)] all external-rollout v3 evaluations finished"
} >> "$LOG" 2>&1 &

echo "run_id=$RUN_ID"
echo "pid=$!"
echo "log=$LOG"
