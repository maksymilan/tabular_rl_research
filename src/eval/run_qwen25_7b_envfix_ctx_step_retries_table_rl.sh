#!/usr/bin/env bash
# Sequentially rerun the 7B epoch2/epoch4 context_overflow + execution_error buckets.
#
# Control-variable intent: only relax context length and max trajectory steps.
# Keep consecutive tool-error retry budget, table observations, and generation length at
# the same settings as the normal eval.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
GPU_ID=${GPU_ID:-0}
VLLM_PORT=${VLLM_PORT:-8030}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen2.5-7B-Instruct}
BASE_RESULT_DIR=${BASE_RESULT_DIR:-/home/dengyan/tabular_rl_project/data/results/qwen2.5_7b_sft_external_rollout_v3_epoch4}

COMMON_ENV=(
  MODEL_DIR="$MODEL_DIR"
  GPU_ID="$GPU_ID"
  VLLM_PORT="$VLLM_PORT"
  MAX_MODEL_LEN=16384
  GPU_MEMORY_UTILIZATION=0.95
  MAX_STEPS=50
  MAX_CONSECUTIVE_ERRORS=3
  MAX_TOKENS=768
  TABLE_OUTPUT_ROWS=10
  ROLLOUT_WORKERS=2
)

cd "$PROJECT_DIR"

env "${COMMON_ENV[@]}" \
  LORA_DIR=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch2-qlora \
  SERVED_MODEL=qwen25_7b_ext_v3_epoch2_retry_envfix_ctx_step \
  RESULT_DIR="$BASE_RESULT_DIR/tool_zero_shot_dev1034_epoch2_retry_envfix_ctx_step" \
  INDICES_FILE="$BASE_RESULT_DIR/tool_zero_shot_dev1034_epoch2/retry_buckets_ctx_exec/combined.indices.json" \
  RUN_ID=qwen25_7b_epoch2_retry_envfix_ctx_step \
  bash src/eval/run_failure_bucket_retry_table_rl.sh

env "${COMMON_ENV[@]}" \
  LORA_DIR=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch4-qlora \
  SERVED_MODEL=qwen25_7b_ext_v3_epoch4_retry_envfix_ctx_step \
  RESULT_DIR="$BASE_RESULT_DIR/tool_zero_shot_dev1034_epoch4_retry_envfix_ctx_step" \
  INDICES_FILE="$BASE_RESULT_DIR/tool_zero_shot_dev1034_epoch4/retry_buckets_ctx_exec/combined.indices.json" \
  RUN_ID=qwen25_7b_epoch4_retry_envfix_ctx_step \
  bash src/eval/run_failure_bucket_retry_table_rl.sh
