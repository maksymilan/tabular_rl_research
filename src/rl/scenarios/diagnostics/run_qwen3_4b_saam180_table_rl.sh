#!/usr/bin/env bash
# Isolated 4B data-scaling arm: the 2026-09-14 correctness-only SAAM recipe on a
# 180-question cohort (frozen 60 + 120 fresh 2-6 BIRD candidates).
# Mechanism, reward, optimizer, rollout and decode are identical to the 60-question
# arm; only the cohort size changes (6 updates = one pass over 180 prompts).
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_saam180_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new180.jsonl"
export TRAIN_GPU=${TRAIN_GPU:?set TRAIN_GPU (idle card for the trainer)}
export VLLM_GPU=${VLLM_GPU:?set VLLM_GPU (idle card for online vLLM, must differ)}
export VLLM_PORT=${VLLM_PORT:-18382}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51482}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260914 --expected-records 180 \
  --expected-examples-sha256 d55ccccdf05b55cf597b1dba29e16040f5da218e95d45bae88305953a9416432 \
  --save-steps 1 --save-total-limit 10 \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096 \
  --replicated-base-storage 4bit --gradient-checkpointing \
  --attn-implementation sdpa --old-policy-logprob-source actor \
  --max-new-tokens 2048 --max-context-tokens 16384
