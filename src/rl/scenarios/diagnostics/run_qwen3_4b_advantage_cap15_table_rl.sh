#!/usr/bin/env bash
# Mechanism validation: symmetric advantage-magnitude cap c=1.5 with SAAM.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_advantage_cap15_saam60_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new60.jsonl"
export PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
export PYTHON="$PYTHON_ENV/bin/python"
export VLLM_PORT=${VLLM_PORT:-18416}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51516}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
train_args=(
  --seed 20260916 --expected-records 60
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca
  --save-steps 1 --save-total-limit 10
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096
  --replicated-base-storage 4bit --gradient-checkpointing
  --attn-implementation sdpa --old-policy-logprob-source actor
  --max-new-tokens 2048 --max-context-tokens 16384
)
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  export OUTPUT_DIR="$RUN_ROOT/train"
  CUDA_VISIBLE_DEVICES="" bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" --preflight-only "${train_args[@]}"
  exit 0
fi
export TRAIN_GPU=${TRAIN_GPU:-0} VLLM_GPU=${VLLM_GPU:-1}
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" "${train_args[@]}"
