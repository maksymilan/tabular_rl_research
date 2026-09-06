#!/usr/bin/env bash
# Wait for strict branch generation, build the controlled dataset, train, then evaluate.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_strict4_20260803}
GLOBAL_OUTPUT_ROOT=${GLOBAL_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GLOBAL_OUTPUT_ROOT/exp15_exact_prefix_branch_20260802}
STRICT_ROOT=${STRICT_ROOT:-$OUTPUT_ROOT/strict4_seed2603}
GENERATION_DIR=$STRICT_ROOT/generation_turn1plus
TRAINING_DIR=$STRICT_ROOT/training
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-$GLOBAL_OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
SOURCE_POOL=$OUTPUT_ROOT/input/validated_trajectories.jsonl
SOURCE_MANIFEST=$OUTPUT_ROOT/input/manifest.json
ORIGINAL_DATASET=$OUTPUT_ROOT/input/original_exp15/fixed_prefix_train.verified.jsonl
ORIGINAL_AUDIT=$OUTPUT_ROOT/input/original_exp15/verification_audit.json
CANDIDATE_NAME=exp15_exact_prefix_strict4_stage_balanced_action_dpo
ARTIFACT=trl-transition-v26-exp15-exact-prefix-strict4-stage-balanced-action-dpo-sft2-seed101-20260803
STATUS=$OUTPUT_ROOT/logs/strict4_seed2603_pipeline.status
RUN_LOG=$OUTPUT_ROOT/logs/strict4_seed2603_pipeline.run.log
LOCK=$STATUS.lock
SINGLE_GPU_ID=${SINGLE_GPU_ID:-6}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$TRAINING_DIR" "$STRICT_ROOT/evaluation"
exec >>"$RUN_LOG" 2>&1
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '%s pipeline lock already held; state=%s\n' "$(timestamp)" "$(state_of "$STATUS")"
  exit 0
fi

set_status waiting_generation "shard=0 gpu=$SINGLE_GPU_ID; shard=1 queued_on_same_gpu"
while true; do
  shard_state=$(state_of "$OUTPUT_ROOT/logs/strict4_seed2603_shard0.status")
  if [[ "$shard_state" == failed ]]; then
    set_status failed "generation_shard=0"
    exit 3
  fi
  [[ "$shard_state" == complete ]] && break
  sleep 30
done

if [[ "$(state_of "$OUTPUT_ROOT/logs/strict4_seed2603_shard1.status")" != complete ]]; then
  set_status generating_second_shard "shard=1 gpu=$SINGLE_GPU_ID serial_after_shard0"
  bash "$RUNTIME/src/rl/experiments/run_exp15_exact_prefix_strict4_seed2603_newgnn.sh" \
    1 "$SINGLE_GPU_ID"
fi

cd "$RUNTIME"
COMMON_ARGS=(
  --source-pool "$SOURCE_POOL"
  --source-manifest "$SOURCE_MANIFEST"
  --question-ids-dataset "$ORIGINAL_DATASET"
  --model-path "$MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --output-dir "$GENERATION_DIR"
  --limit-trajectories 0
  --anchors-per-trajectory 8
  --min-anchor-turn-index 1
  --candidate-count 4
  --candidate-draws 12
  --continuation-count 4
  --max-pairs-per-anchor 2
  --temperature 0.7
  --top-p 0.95
  --max-steps 30
  --max-new-tokens 1024
  --max-context-tokens 8192
  --history-turns 4
  --seed 2603
)

set_status replay_validating "exact_source_prefixes_and_suffixes"
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
  "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
    "${COMMON_ARGS[@]}" --replay-validate-only

set_status finalizing_generation "strict4_views"
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
  "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
    "${COMMON_ARGS[@]}" --finalize-only

set_status building_strict_union "positive=4/4 negative=0/4 turn>=1 same_32_questions"
"$PYTHON_BIN" src/rl/action_dpo/build_strict_exact_prefix_training_set.py \
  --original-dataset "$ORIGINAL_DATASET" \
  --original-audit "$ORIGINAL_AUDIT" \
  --expanded-dataset "$GENERATION_DIR/exact_prefix_pairs.online_consistent.jsonl" \
  --expanded-manifest "$GENERATION_DIR/manifest.json" \
  --expanded-file-key pairs_online_consistent \
  --source-replay-audit "$GENERATION_DIR/source_replay_audit.json" \
  --output-dataset "$TRAINING_DIR/strict_exp15_union.verified.jsonl" \
  --output-audit "$TRAINING_DIR/strict_exp15_union.verification_audit.json" \
  --min-online-trials 4 \
  --min-turn-index 1 \
  --max-new-pairs-per-question-stage 1 \
  --max-new-pairs-per-question 2 \
  --min-new-pairs 8 \
  --min-new-questions 6

chmod 444 \
  "$GENERATION_DIR/plan.json" \
  "$GENERATION_DIR/source_replay_audit.json" \
  "$GENERATION_DIR/manifest.json" \
  "$GENERATION_DIR/exact_prefix_pairs.jsonl" \
  "$GENERATION_DIR/exact_prefix_pairs.online_supported.jsonl" \
  "$GENERATION_DIR/exact_prefix_pairs.online_consistent.jsonl" \
  "$TRAINING_DIR/strict_exp15_union.verified.jsonl" \
  "$TRAINING_DIR/strict_exp15_union.verification_audit.json"

set_status training "gpu=$SINGLE_GPU_ID questions=32 optimizer_steps=32"
DATA_DIR="$TRAINING_DIR" \
RUNTIME="$RUNTIME" \
DATASET="$TRAINING_DIR/strict_exp15_union.verified.jsonl" \
AUDIT="$TRAINING_DIR/strict_exp15_union.verification_audit.json" \
ARTIFACT="$ARTIFACT" \
EXPERIMENT_NAME="$CANDIDATE_NAME" \
STATUS="$OUTPUT_ROOT/logs/strict4_seed2603_train.status" \
TRAIN_LOG="$OUTPUT_ROOT/logs/strict4_seed2603_train.run.log" \
COMPLETION_AUDIT="$TRAINING_DIR/training_completion_audit.json" \
SELECTED_OUTPUT="$TRAINING_DIR/selected_training_output.txt" \
  bash "$RUNTIME/src/rl/experiments/run_exp15_exact_prefix_expanded_train_newgnn.sh" \
    "$SINGLE_GPU_ID"

SELECTED_DIR=$(sed -n '1p' "$TRAINING_DIR/selected_training_output.txt")
test -d "$SELECTED_DIR/final"
ADAPTER_SHA256=$(sha256sum "$SELECTED_DIR/final/adapter_model.safetensors" | awk '{print $1}')
cp "$OUTPUT_ROOT/evaluation/original_exp15_decision_requirements.json" \
  "$STRICT_ROOT/evaluation/original_exp15_decision_requirements.json"

set_status evaluating "gpu=$SINGLE_GPU_ID serial=fixed_prefix_then_full_dev"
OUTPUT_ROOT="$GLOBAL_OUTPUT_ROOT" \
EXPERIMENT_ROOT="$STRICT_ROOT" \
LAUNCH_RUNTIME="$RUNTIME" \
CANDIDATE_NAME="$CANDIDATE_NAME" \
ADAPTER="$SELECTED_DIR/final" \
ADAPTER_SHA256="$ADAPTER_SHA256" \
PREFIX_GPU_ID="$SINGLE_GPU_ID" GREEDY_GPU_ID="$SINGLE_GPU_ID" \
SERIAL_EVALUATION=1 PORT=18087 \
STATUS="$OUTPUT_ROOT/logs/strict4_seed2603_evaluation.status" \
RUN_LOG="$OUTPUT_ROOT/logs/strict4_seed2603_evaluation.run.log" \
  bash "$RUNTIME/src/rl/experiments/run_exp15_expanded_stage1_evaluation_newgnn.sh"

set_status complete "decision=$STRICT_ROOT/evaluation/decision_requirements.json"
trap - EXIT
