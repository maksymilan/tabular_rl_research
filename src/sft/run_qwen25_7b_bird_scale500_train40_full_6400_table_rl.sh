#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/qwen2.5_7b_qlora_bird_scale500_train40_rolling4_full_6400_table_rl.yaml}
GPU_ID=${GPU_ID:-0}
RUN_ID=${RUN_ID:-bird_scale500_train40_full_6400_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="$OUTPUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
PID_FILE="$LOG_DIR/$RUN_ID.pid"

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"

echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/bird_scale500_train40_full_6400_latest.run_id"

echo "run_id=$RUN_ID"
echo "pid=$$"
echo "log=$LOG"
echo "config=$CONFIG"

exec /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train "$CONFIG" > "$LOG" 2>&1
