#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?usage: run_exp15_exact_prefix_expanded_train_newgnn.sh GPU_ID}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_exact_prefix_branch_20260802}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_branch_20260802}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
EXPECTED_ADAPTER_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e

DATA_DIR=${DATA_DIR:-$OUTPUT_ROOT/scale48/training}
DATASET=${DATASET:-$DATA_DIR/expanded_exp15_union.verified.jsonl}
AUDIT=${AUDIT:-$DATA_DIR/expanded_exp15_union.verification_audit.json}
CHECKPOINT_ROOT=/home/dengyan/tabular_rl_outputs/checkpoints
ARTIFACT=${ARTIFACT:-trl-transition-v26-exp15-exact-prefix-expanded-union-action-dpo-sft2-seed101-20260802}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-exp15_exact_prefix_expanded_union_action_dpo}
RUN_DIR=$CHECKPOINT_ROOT/$ARTIFACT
RETRY_DIR=$CHECKPOINT_ROOT/${ARTIFACT}-lr5e7
LOG_DIR=$OUTPUT_ROOT/logs
STATUS=${STATUS:-$LOG_DIR/expanded_exp15_train.status}
TRAIN_LOG=${TRAIN_LOG:-$LOG_DIR/expanded_exp15_train.log}
COMPLETION_AUDIT=${COMPLETION_AUDIT:-$DATA_DIR/training_completion_audit.json}
SELECTED_OUTPUT=${SELECTED_OUTPUT:-$DATA_DIR/selected_training_output.txt}
mkdir -p "$LOG_DIR"

set_status() {
  printf 'state=%s gpu=%s detail=%s time=%s\n' \
    "$1" "$GPU_ID" "$2" "$(date -Is)" >"$STATUS"
}
on_exit() {
  exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 ]]; then
    set_status failed "exit_code=$exit_code log=$TRAIN_LOG"
  fi
  exit "$exit_code"
}
trap on_exit EXIT

test -s "$DATASET"
test -s "$AUDIT"
actual_adapter_sha=$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | awk '{print $1}')
if [[ "$actual_adapter_sha" != "$EXPECTED_ADAPTER_SHA" ]]; then
  set_status failed "initial_adapter_hash_mismatch=$actual_adapter_sha"
  exit 4
fi

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | sed -n "$((GPU_ID + 1))p")
    if [[ -n "$used" && "$used" -le 512 ]]; then
      break
    fi
    set_status waiting_gpu "memory_used_mib=${used:-unknown}"
    sleep 30
  done
}

train_one() {
  learning_rate=$1
  output_dir=$2
  experiment_name=$3
  if [[ -e "$output_dir" ]]; then
    set_status failed "refusing_existing_output=$output_dir"
    exit 4
  fi
  wait_for_gpu
  set_status training "name=$experiment_name lr=$learning_rate beta=0.1"
  cd "$RUNTIME"
  extra_args=()
  if [[ "${SEQUENTIAL_PAIR_SCORING:-0}" == 1 ]]; then
    extra_args+=(--sequential-pair-scoring)
  fi
  if [[ "${MEMORY_SAFE_DPO_BACKWARD:-0}" == 1 ]]; then
    extra_args+=(--memory-safe-dpo-backward)
  fi
  if [[ "${SELECTED_TOOL_LOGITS:-0}" == 1 ]]; then
    extra_args+=(--selected-tool-logits)
  fi
  if [[ -n "${ATTENTION_IMPLEMENTATION:-}" ]]; then
    extra_args+=(--attention-implementation "$ATTENTION_IMPLEMENTATION")
  fi
  if [[ "${CPU_OFFLOAD_OPTIMIZER_STATE:-0}" == 1 ]]; then
    extra_args+=(--cpu-offload-optimizer-state)
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
    "$PYTHON_BIN" src/rl/action_dpo/train_fixed_prefix_action_dpo.py \
      --dataset "$DATASET" \
      --verification-audit "$AUDIT" \
      --model-path "$MODEL_PATH" \
      --adapter-path "$ADAPTER_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --beta 0.1 \
      --learning-rate "$learning_rate" \
      --epochs 1 \
      --gradient-clip 1.0 \
      --max-length 8192 \
      --seed 101 \
      "${extra_args[@]}"
}

exec >>"$TRAIN_LOG" 2>&1
train_one 1e-6 "$RUN_DIR" "$EXPERIMENT_NAME"
clip_retry=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(int(float(json.load(open(sys.argv[1]))["gradient_clip_fraction"]) > 0.20))' \
    "$RUN_DIR/run_manifest.json"
)
SELECTED_DIR=$RUN_DIR
SELECTED_LR=1e-6
if [[ "$clip_retry" == 1 ]]; then
  train_one 5e-7 "$RETRY_DIR" "${EXPERIMENT_NAME}_lr5e7"
  SELECTED_DIR=$RETRY_DIR
  SELECTED_LR=5e-7
fi

set_status validating "selected_lr=$SELECTED_LR"
"$PYTHON_BIN" src/rl/action_dpo/validate_exp15_training_run.py \
  --run-dir "$SELECTED_DIR" \
  --dataset "$DATASET" \
  --verification-audit "$AUDIT" \
  --expected-initial-adapter-sha256 "$EXPECTED_ADAPTER_SHA" \
  --expected-learning-rate "$SELECTED_LR" \
  --output-audit "$COMPLETION_AUDIT"
printf '%s\n' "$SELECTED_DIR" >"$SELECTED_OUTPUT"
set_status complete "selected=$SELECTED_DIR lr=$SELECTED_LR"
trap - EXIT
