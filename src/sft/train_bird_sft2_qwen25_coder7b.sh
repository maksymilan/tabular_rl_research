#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_sft2_qwen25_coder7b_cp560_batch2_single_gpu_qlora_6400.yaml}
RUN_ID=${RUN_ID:-bird_sft2_qwen25_coder7b_cp560_batch2_$(date +%Y%m%d_%H%M%S)}
GPU_IDS=${GPU_IDS:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
LOG_DIR="$OUTPUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
PID_FILE="$LOG_DIR/$RUN_ID.pid"

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NPROC_PER_NODE
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="/home/dengyan/miniconda3/envs/sft/bin:${PATH}"

echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/bird_sft2_qwen25_coder7b_cp560_batch2_latest.run_id"
echo "run_id=$RUN_ID"
echo "pid=$$"
echo "gpus=$GPU_IDS"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "log=$LOG"
echo "config=$CONFIG"

exec /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train "$CONFIG" > "$LOG" 2>&1
