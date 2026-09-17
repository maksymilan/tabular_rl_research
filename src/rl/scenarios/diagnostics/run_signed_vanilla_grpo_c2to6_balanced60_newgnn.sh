#!/usr/bin/env bash
# Frozen checkpoint-6380 c=2..6 signed-binary vanilla-GRPO diagnostic.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set isolated RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_8b_atomic_v26_signed_vanilla_grpo_c2to6_balanced60_newgnn.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-8B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data/rl_inputs/qwen3_v26_checkpoint6380_k8_correct2to6_balanced60_20260911.jsonl"
export VLLM_PORT=${VLLM_PORT:-18310}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51410}
export MAX_MODEL_LEN=16384
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260911 --expected-records 60 \
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca \
  --save-steps 1 --save-total-limit 10 \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 8192 \
  --replicated-base-storage 4bit
