#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dengyan/tabular_rl_project
PYTHON_ENV=/home/dengyan/miniconda3/envs/sft
GPU_ID=${GPU_ID:-7}
CONFIG=${CONFIG:-$ROOT/scripts/sft/configs/qwen2.5_7b_qlora_sft.yaml}
LOG=${LOG:-$ROOT/logs/qwen2.5_7b_spider_v1_qlora.log}
GPU_LOG=${GPU_LOG:-$ROOT/logs/qwen2.5_7b_spider_v1_qlora_gpu.csv}
PID_FILE=${PID_FILE:-$ROOT/logs/qwen2.5_7b_spider_v1_qlora.pid}

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Training is already running with PID $old_pid" >&2
    exit 1
  fi
fi

used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
if (( used_mib > 512 )); then
  echo "GPU $GPU_ID is not idle: ${used_mib} MiB is already in use" >&2
  exit 1
fi

: > "$LOG"
: > "$GPU_LOG"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_ENV/bin/llamafactory-cli" train "$CONFIG" \
  > "$LOG" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$PID_FILE"

nohup bash -c '
  pid=$1
  gpu=$2
  output=$3
  while kill -0 "$pid" 2>/dev/null; do
    timestamp=$(date -Iseconds)
    stats=$(nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | sed -n "$((gpu + 1))p")
    printf "%s,%s\n" "$timestamp" "$stats" >> "$output"
    sleep 30
  done
' bash "$train_pid" "$GPU_ID" "$GPU_LOG" \
  > /dev/null 2>&1 < /dev/null &

echo "training_pid=$train_pid"
echo "gpu=$GPU_ID"
echo "log=$LOG"
echo "gpu_log=$GPU_LOG"
