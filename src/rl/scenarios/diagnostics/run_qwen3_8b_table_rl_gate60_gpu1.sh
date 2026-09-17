#!/usr/bin/env bash
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_gate60_table_rl_gpu1.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-8B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl"
export PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
export PYTHON="$PYTHON_ENV/bin/python"
export VLLM_PORT=${VLLM_PORT:-18394}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51494}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
train_args=(
  --seed 20260915 --expected-records 60
  --expected-examples-sha256 47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5
  --save-steps 1 --save-total-limit 3
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
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" "${train_args[@]}"
