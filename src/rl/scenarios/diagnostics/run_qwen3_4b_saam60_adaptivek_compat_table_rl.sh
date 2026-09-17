#!/usr/bin/env bash
# Isolated 4B arm that switches on BOTH newly implemented mechanisms on top of the
# 60-question / 6-update SAAM recipe:
#   * --adaptive-group-size-max 16 : all-correct / all-wrong K=8 groups get one extra round of
#     8 samples (mixed groups untouched; advantages are computed inside the merged group);
#   * --carrier-repair             : runtime-only transport repair of carrier-shaped slips
#     (unclosed <think>, markdown fence, extra prose around the action object). Strict carrier
#     semantics stay for every other arm; both switches are recorded in the run manifest.
# This is a combined-effect arm, not a single-variable contrast: read it as "do the two new
# mechanisms move the point estimate", then isolate whichever one looks promising.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_saam60_adaptivek_compat_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new60.jsonl"
export TRAIN_GPU=${TRAIN_GPU:?set TRAIN_GPU (idle card for the trainer)}
export VLLM_GPU=${VLLM_GPU:?set VLLM_GPU (idle card for online vLLM, must differ)}
export VLLM_PORT=${VLLM_PORT:-18422}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51522}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260914 --expected-records 60 \
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca \
  --save-steps 1 --save-total-limit 10 \
  --adaptive-group-size-max 16 --carrier-repair \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096 \
  --replicated-base-storage 4bit --gradient-checkpointing \
  --attn-implementation sdpa --old-policy-logprob-source actor \
  --max-new-tokens 2048 --max-context-tokens 16384
