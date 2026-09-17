#!/usr/bin/env bash
# Authorized isolated 4B first non-timeout error cap control.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_first_error_capped_saam60_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new60.jsonl"
export VLLM_PORT=${VLLM_PORT:-18374}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51474}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260912 --expected-records 60 \
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca \
  --save-steps 1 --save-total-limit 10 \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096 \
  --replicated-base-storage 4bit --gradient-checkpointing \
  --attn-implementation sdpa --old-policy-logprob-source actor \
  --max-new-tokens 2048 --max-context-tokens 16384
