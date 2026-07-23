#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=${1:?usage: train_model.sh qwen25-coder|omnisql GPU_ID}
GPU_ID=${2:?usage: train_model.sh qwen25-coder|omnisql GPU_ID}
PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_experiments/omnisql_sft12_ablation}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k}

case "$MODEL_KEY" in
  qwen25-coder)
    CONFIG="$PROJECT_DIR/reproductions/omnisql_sft12_ablation/configs/qwen25_coder_7b_sft12_1k_qlora.yaml"
    OUTPUT_DIR="$OUTPUT_ROOT/checkpoints/qwen25_coder_7b"
    ;;
  omnisql)
    CONFIG="$PROJECT_DIR/reproductions/omnisql_sft12_ablation/configs/omnisql_7b_sft12_1k_qlora.yaml"
    OUTPUT_DIR="$OUTPUT_ROOT/checkpoints/omnisql_7b"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_DIR/trainer_state.json" ]]; then
  echo "completed trainer state already exists: $OUTPUT_DIR/trainer_state.json" >&2
  exit 3
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty output directory: $OUTPUT_DIR" >&2
  exit 4
fi

memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
if (( memory_used > 256 )); then
  echo "GPU $GPU_ID is not idle: ${memory_used} MiB used" >&2
  exit 5
fi

RUN_ID="${MODEL_KEY}_sft12_1k_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
PID_FILE="$LOG_DIR/$RUN_ID.pid"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"

echo "$$" > "$PID_FILE"
{
  echo "run_id=$RUN_ID"
  echo "pid=$$"
  echo "model_key=$MODEL_KEY"
  echo "gpu=$GPU_ID"
  echo "config=$CONFIG"
  echo "output_dir=$OUTPUT_DIR"
  echo "project_commit=$(git rev-parse HEAD 2>/dev/null || echo exported-worktree)"
  echo "dataset_sha256=$(sha256sum data/sft/bird_sft12_ablation_1k.jsonl | awk '{print $1}')"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader -i "$GPU_ID"
} > "$LOG_DIR/$RUN_ID.manifest.txt"

exec /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train "$CONFIG" > "$LOG" 2>&1
