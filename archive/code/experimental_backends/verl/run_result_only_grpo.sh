#!/usr/bin/env bash
# Launch the Verl adapter for the closed-loop, result-only GRPO baseline.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
VERL_ENV=${VERL_ENV:-/home/dengyan/miniconda3/envs/verl-table/bin/python}
# Keep the framework checkout outside the synchronised project directory.  The server's project
# mirror is periodically refreshed, while experiment dependencies must remain stable for resume.
VERL_ROOT=${VERL_ROOT:-/home/dengyan/tabular_rl_outputs/third_party/verl}
CONFIG_DIR=${CONFIG_DIR:-$PROJECT_DIR/src/rl/frameworks/verl/configs}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
RUN_ID=${RUN_ID:-verl_qwen25_7b_result_only_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

export HF_HUB_OFFLINE=1
# Verl's colocated vLLM engine uses CUDA memory pools, which are incompatible with PyTorch
# ``expandable_segments``.  Do not inherit this SFT-only setting into the RL process.
unset PYTORCH_CUDA_ALLOC_CONF
# The two RTX 3090s are not peer-accessible.  FSDP can still shard across them through host
# staging, but NCCL must not attempt the unavailable P2P transport during module broadcast.
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
# The base model stays FSDP-sharded; the small rank-16 adapter is replicated with averaged gradients.
export TABLE_RL_REPLICATED_LORA=1
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_DIR/src/rl:$PROJECT_DIR/src/eval:$PROJECT_DIR/src/harness:$PROJECT_DIR/src/sft:${PYTHONPATH:-}"

cd "$VERL_ROOT"
"$VERL_ENV" "$PROJECT_DIR/src/rl/frameworks/verl/patch_host_fallback.py" --verl-root "$VERL_ROOT"
"$VERL_ENV" -m verl.trainer.main_ppo --config-path "$CONFIG_DIR" \
  --config-name verl_qwen25_7b_result_only_grpo \
  trainer.experiment_name="$RUN_ID" \
  "$@" 2>&1 | tee "$LOG_DIR/$RUN_ID.log"
