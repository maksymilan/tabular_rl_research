#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
GENERAL_CONFIG=${GENERAL_CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen25_7b_qlora_6400.yaml}
CODER_CONFIG=${CODER_CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen25_coder_7b_qlora_6400.yaml}
GENERAL_GPU_ID=${GENERAL_GPU_ID:?GENERAL_GPU_ID must identify an idle GPU}
CODER_GPU_ID=${CODER_GPU_ID:?CODER_GPU_ID must identify an idle GPU}
MODE=${MODE:-train}
RUN_TAG=${RUN_TAG:-external_teacher_dual_7b_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="$OUTPUT_ROOT/logs/$RUN_TAG"

if [[ "$GENERAL_GPU_ID" == "$CODER_GPU_ID" ]]; then
  echo "GENERAL_GPU_ID and CODER_GPU_ID must be different" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "train" ]]; then
  echo "MODE must be smoke or train" >&2
  exit 2
fi
for path in "$GENERAL_CONFIG" "$CODER_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "missing config: $path" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
cd "$PROJECT_DIR"

general_overrides=()
coder_overrides=()
if [[ "$MODE" == "smoke" ]]; then
  general_overrides=(
    "max_steps=5"
    "num_train_epochs=1.0"
    "save_strategy=no"
    "output_dir=$OUTPUT_ROOT/checkpoints/smoke-qwen2.5-7b-bird-external-teacher-fixed1000-6400"
    "overwrite_output_dir=true"
  )
  coder_overrides=(
    "max_steps=5"
    "num_train_epochs=1.0"
    "save_strategy=no"
    "output_dir=$OUTPUT_ROOT/checkpoints/smoke-qwen2.5-coder-7b-bird-external-teacher-fixed1000-6400"
    "overwrite_output_dir=true"
  )
fi

CUDA_VISIBLE_DEVICES="$GENERAL_GPU_ID" \
  /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train \
  "$GENERAL_CONFIG" "${general_overrides[@]}" \
  >"$LOG_DIR/general_7b.log" 2>&1 &
general_pid=$!

CUDA_VISIBLE_DEVICES="$CODER_GPU_ID" \
  /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train \
  "$CODER_CONFIG" "${coder_overrides[@]}" \
  >"$LOG_DIR/coder_7b.log" 2>&1 &
coder_pid=$!

printf '%s\n' "$general_pid" >"$LOG_DIR/general_7b.pid"
printf '%s\n' "$coder_pid" >"$LOG_DIR/coder_7b.pid"
printf 'mode=%s\ngeneral_gpu=%s\ncoder_gpu=%s\ngeneral_pid=%s\ncoder_pid=%s\n' \
  "$MODE" "$GENERAL_GPU_ID" "$CODER_GPU_ID" "$general_pid" "$coder_pid" \
  >"$LOG_DIR/run.env"

status=0
wait "$general_pid" || status=$?
wait "$coder_pid" || status=$?
exit "$status"
