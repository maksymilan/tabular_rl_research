#!/usr/bin/env bash
# Confirmed 60-task signed-result hybrid arm; composes shared lifecycle helpers.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set isolated RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_8b_atomic_v26_binary_saam_legal_reason_hybrid_balanced60_newgnn.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-8B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data/rl_inputs/qwen3_8b_atomic_v26_smc_balanced60_train_v1.jsonl"
export VLLM_PORT=${VLLM_PORT:-18295}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51395}
export MAX_MODEL_LEN=16384
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260901 --expected-records 60 \
  --expected-examples-sha256 b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4 \
  --save-steps 1 --save-total-limit 10 \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 8192 \
  --replicated-base-storage 4bit
