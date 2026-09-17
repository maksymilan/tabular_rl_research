#!/usr/bin/env bash
# Continue the 180-question SAAM arm for ONE MORE EPOCH (updates 6 -> 12) from its own
# checkpoint-6, into a fresh run root, so the completed 1-epoch run stays untouched.
#
# Resume contract (all three verified in the code and with a CPU-only preflight):
#   * the resume checkpoint must be an IMMEDIATE CHILD of the output directory, and that
#     directory must already contain the run's own run_manifest.json + implementation_lock.json
#     (configuration/experiment_config.require_resume_base_model_identity), so the caller stages
#     a copy of the original run's checkpoint-6 + manifest + lock + source snapshot into the new
#     run root before launching;
#   * the implementation lock binds the experiment config FILE hash, the SFT adapter hash and the
#     implementation sources, so this launcher passes the ORIGINAL 180-question config unchanged
#     and ADAPTER_PATH stays the frozen SFT adapter; the extra updates come from the CLI override
#     --optimizer-steps 12 (`parser.set_defaults` means the config only supplies defaults);
#   * everything else (cohort, seed, K, reward, credit, span, LR, decode) stays identical, so this
#     is the same recipe continuing for a second pass over its own 180 questions.
# Everything else (cohort, seed, K, reward, credit, span, LR, decode) is unchanged, so this is
# "the same recipe trained for a second pass over its own 180 questions".
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to the staged checkpoint-6 inside RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_saam180_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new180.jsonl"
export TRAIN_GPU=${TRAIN_GPU:?set TRAIN_GPU (idle card for the trainer)}
export VLLM_GPU=${VLLM_GPU:?set VLLM_GPU (idle card for online vLLM, must differ)}
export VLLM_PORT=${VLLM_PORT:-18430}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51530}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
mkdir -p "$RUN_ROOT/train"
exec bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" \
  --seed 20260914 --expected-records 180 \
  --expected-examples-sha256 d55ccccdf05b55cf597b1dba29e16040f5da218e95d45bae88305953a9416432 \
  --resume-from-checkpoint "$RESUME_CHECKPOINT" \
  --optimizer-steps 12 \
  --save-steps 1 --save-total-limit 20 \
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096 \
  --replicated-base-storage 4bit --gradient-checkpointing \
  --attn-implementation sdpa --old-policy-logprob-source actor \
  --max-new-tokens 2048 --max-context-tokens 16384
