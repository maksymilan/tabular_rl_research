#!/usr/bin/env bash
# Pre-registered alpha=0.5 control; shared launchers own GPU lifecycle and evaluation.
set -Eeuo pipefail
export PROJECT_DIR=${PROJECT_DIR:?set isolated PROJECT_DIR}
export RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
export EXPERIMENT_CONFIG="$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_later_error_half_saam60_table_rl.yaml"
export MODEL_PATH=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
export ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380
export EXAMPLES_JSON="$PROJECT_DIR/data_new60.jsonl"
export PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
export PYTHON="$PYTHON_ENV/bin/python"
export VLLM_PORT=${VLLM_PORT:-18384}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51484}
export MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=0
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
train_args=(
  --seed 20260912 --expected-records 60
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca
  --save-steps 1 --save-total-limit 10
  --transition-micro-batch-size 1 --transition-micro-batch-tokens 4096
  --replicated-base-storage 4bit --gradient-checkpointing
  --attn-implementation sdpa --old-policy-logprob-source actor
  --max-new-tokens 2048 --max-context-tokens 16384
)
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  export OUTPUT_DIR="$RUN_ROOT/train"
  CUDA_VISIBLE_DEVICES="" bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --preflight-only "${train_args[@]}"
  exit 0
fi

# The frozen matched controller is staged before launch; no evaluation code is edited in flight.
export RUN_DIR=${EVAL_RUN_DIR:?set a fresh EVAL_RUN_DIR with staged controller}
[[ -f "$RUN_DIR/controller/launcher.sh" ]]
source "$PROJECT_DIR/src/rl/frameworks/launcher/launch_common.sh"
trap 'code=$?; if (( code != 0 )); then set_status "$RUN_ROOT" failed "pipeline exit=$code; inspect training/evaluation logs"; fi' EXIT
bash "$PROJECT_DIR/src/rl/frameworks/launcher/run_trl_gpu_pair.sh" "${train_args[@]}"

# Snapshot verifies global_step/max_steps and stable, nonempty checkpoint files.
"$PYTHON" -m rl.scenarios.evaluation.preserve_diagnostic_checkpoint \
  --source "$RUN_ROOT/train/checkpoint-4" --destination "$RUN_ROOT/eval_checkpoint-4" \
  --expected-step 4 --expected-max-steps 4 --timeout-seconds 60
export ADAPTER="$RUN_ROOT/eval_checkpoint-4"
export SOURCE_CHECKPOINT="$RUN_ROOT/train/checkpoint-4"
export CHECKPOINT_GLOBAL_STEP=4
export MODEL="$MODEL_PATH"
export PYTHON_BIN="$PYTHON"
export GPU0="$TRAIN_GPU" GPU1="$VLLM_GPU"
export PORT0=${EVAL_PORT0:-18386} PORT1=${EVAL_PORT1:-18387}
export MAX_TOKENS=2048 ERROR_FEEDBACK_VERSION=actionable-error-v1
export VLLM_MAX_MODEL_LEN=16384 VLLM_BATCH_TOKENS=16384 VLLM_MAX_NUM_SEQS=24 EVAL_WORKERS=24
export VLLM_MEMORY_UTILIZATION=0.90
set_status "$RUN_ROOT" evaluating "checkpoint-4 verified; matched BIRD-dev1534 on two independent replicas"
bash "$RUN_DIR/controller/launcher.sh"
set_status "$RUN_ROOT" evaluated_pending_audit "matched shards merged; paired comparison and fresh replay still required"
