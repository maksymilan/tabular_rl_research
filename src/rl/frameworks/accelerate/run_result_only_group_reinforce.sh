#!/usr/bin/env bash
# Single-GPU hardware-compatible result-only RL baseline. No vLLM, Ray, FSDP, or process rewards.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/sft/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch4-qlora}
EXAMPLES_JSON=${EXAMPLES_JSON:-}
SELECTION=${SELECTION:-}
OUTPUT_DIR=${OUTPUT_DIR:-/home/dengyan/tabular_rl_outputs/checkpoints/accelerate-qwen25-7b-result-only-group-reinforce}

mkdir -p /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_DIR/src/rl:$PROJECT_DIR/src/eval:$PROJECT_DIR/src/harness:$PROJECT_DIR/src/sft:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
data_args=()
if [ -n "$EXAMPLES_JSON" ]; then
  data_args+=(--examples-json "$EXAMPLES_JSON")
elif [ -n "$SELECTION" ]; then
  data_args+=(--selection "$SELECTION")
fi

# With neither selector, task_data.load_result_only_task_records uses all Spider train examples.
"$PYTHON" src/rl/frameworks/accelerate/group_reinforce.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "${data_args[@]}" \
  "$@"
