#!/usr/bin/env bash
# Expand Exp15 to previously uncovered questions, train from SFT2, then evaluate.
set -euo pipefail

GPU_ID=${GPU_ID:-1}
GLOBAL_OUTPUT_ROOT=${GLOBAL_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$GLOBAL_OUTPUT_ROOT/rl_runtime_exp15_branch_dynamic_20260803}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GLOBAL_OUTPUT_ROOT/exp15_question_expansion71_20260804}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-$GLOBAL_OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
SOURCE_DIR=${SOURCE_DIR:-$GLOBAL_OUTPUT_ROOT/phase8_dense_stage2_20260802/balanced_mixed120_rank_seed101}
SOURCE_POOL=$SOURCE_DIR/validated_trajectories.jsonl
SOURCE_MANIFEST=$SOURCE_DIR/manifest.json

INPUT_DIR=$OUTPUT_ROOT/input
GENERATION_DIR=$OUTPUT_ROOT/generation
TRAINING_DIR=$OUTPUT_ROOT/training
EVALUATION_DIR=$OUTPUT_ROOT/evaluation
LOG_DIR=$OUTPUT_ROOT/logs
QUESTION_FILTER=$INPUT_DIR/new71_questions.jsonl
QUESTION_FILTER_AUDIT=$INPUT_DIR/new71_questions.selection_audit.json
UNION_DATASET=$TRAINING_DIR/expanded_strict_union.verified.jsonl
UNION_AUDIT=$TRAINING_DIR/expanded_strict_union.verification_audit.json
STATUS=$OUTPUT_ROOT/pipeline.status
RUN_LOG=$OUTPUT_ROOT/pipeline.log
LOCK=$OUTPUT_ROOT/pipeline.lock

ORIGINAL_DATASET=$INPUT_DIR/original36.verified.jsonl
ORIGINAL_AUDIT=$INPUT_DIR/original36.verification_audit.json
CONSISTENT28_DATASET=$INPUT_DIR/consistent28.verified.jsonl
CONSISTENT28_AUDIT=$INPUT_DIR/consistent28.verification_audit.json
PRIOR60_DATASET=$GLOBAL_OUTPUT_ROOT/exp15_exact_prefix_12h_dynamic_20260803/data/exact_prefix_pairs.online_consistent.jsonl
PRIOR60_MANIFEST=$GLOBAL_OUTPUT_ROOT/exp15_exact_prefix_12h_dynamic_20260803/data/manifest.json
NEW_STRICT_DATASET=$GENERATION_DIR/exact_prefix_pairs.online_consistent.jsonl
NEW_MANIFEST=$GENERATION_DIR/manifest.json
ORIGINAL_DECISION_INPUT=$INPUT_DIR/original_exp15_decision_requirements.json

EXPECTED_SOURCE_POOL_SHA=60e08079f792c224c593eec824fb64011ea65cad04b7b3dcff27891b7c7721b0
EXPECTED_SOURCE_MANIFEST_SHA=53b83cbdd9ef27e416f401a2ac931c81cca849a39f8e78641ec3f660a4faf9f6
EXPECTED_ADAPTER_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
EXPECTED_ORIGINAL_SHA=1a6c3e9937b2a6d899d3f8636556f644bb3b9072bb0b8fee850d34617fd96dcc
EXPECTED_ORIGINAL_AUDIT_SHA=8a017c092891bee221d845868d65f6af41b82e9c9f9ebff9d977407f40f7f98a
EXPECTED_CONSISTENT28_SHA=451ffbd7343dd54d8f760e0640c514d0a693a3ef4657674de342124428c1dcf6
EXPECTED_PRIOR60_SHA=59bb04a16f7fe008d2e2a7f5c2a65de639059118eb954f653c5e1c538d91c38f

CANDIDATE_NAME=exp15_question_expanded_strict_action_dpo_20260804
ARTIFACT=trl-transition-v26-exp15-question-expanded-strict-action-dpo-sft2-seed101-20260804
TRAIN_STATUS=$LOG_DIR/train.status
TRAIN_LOG=$LOG_DIR/train.log
EVAL_STATUS=$LOG_DIR/evaluation.status
EVAL_LOG=$LOG_DIR/evaluation.log
SELECTED_OUTPUT=$TRAINING_DIR/selected_training_output.txt
COMPLETION_AUDIT=$TRAINING_DIR/training_completion_audit.json
PORT=${PORT:-18091}

if [[ "$GPU_ID" != 1 ]]; then
  printf 'this pipeline is authorized only for table_rl GPU 1\n' >&2
  exit 2
fi

mkdir -p "$INPUT_DIR" "$GENERATION_DIR" "$TRAINING_DIR" "$EVALUATION_DIR" "$LOG_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'another question-expansion pipeline holds %s\n' "$LOCK" >&2
  exit 3
fi

set_status() {
  printf 'state=%s gpu=%s detail=%s time=%s\n' "$1" "$GPU_ID" "$2" "$(date -Is)" >"$STATUS"
}

on_exit() {
  exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 ]]; then
    set_status failed "exit_code=$exit_code log=$RUN_LOG"
  fi
  exit "$exit_code"
}
trap on_exit EXIT
exec >>"$RUN_LOG" 2>&1

require_sha() {
  path=$1
  expected=$2
  label=$3
  test -f "$path" || { printf 'missing %s: %s\n' "$label" "$path" >&2; return 1; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    printf '%s hash mismatch expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    return 1
  }
}

set_status preflight "validating frozen inputs"
test -f "$RUNTIME/src/rl/action_dpo/prepare_exp15_question_expansion.py"
test -f "$RUNTIME/src/rl/action_dpo/build_multisource_strict_exp15_union.py"
test -f "$RUNTIME/src/rl/action_dpo/generate_exact_prefix_branches.py"
test -f "$RUNTIME/src/rl/experiments/run_exp15_exact_prefix_expanded_train_newgnn.sh"
test -f "$RUNTIME/src/rl/experiments/run_exp15_expanded_stage1_evaluation_newgnn.sh"
require_sha "$SOURCE_POOL" "$EXPECTED_SOURCE_POOL_SHA" source_pool
require_sha "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA" source_manifest
require_sha "$ADAPTER_PATH/adapter_model.safetensors" "$EXPECTED_ADAPTER_SHA" sft2_adapter
require_sha "$ORIGINAL_DATASET" "$EXPECTED_ORIGINAL_SHA" original36
require_sha "$ORIGINAL_AUDIT" "$EXPECTED_ORIGINAL_AUDIT_SHA" original36_audit
require_sha "$CONSISTENT28_DATASET" "$EXPECTED_CONSISTENT28_SHA" consistent28
require_sha "$PRIOR60_DATASET" "$EXPECTED_PRIOR60_SHA" prior_strict60
test -f "$CONSISTENT28_AUDIT"
test -f "$PRIOR60_MANIFEST"

cd "$RUNTIME"
if [[ ! -e "$QUESTION_FILTER" && ! -e "$QUESTION_FILTER_AUDIT" ]]; then
  set_status selecting_questions "eligible_pool=94 existing_union=50 expected_new=71"
  PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
    "$PYTHON_BIN" src/rl/action_dpo/prepare_exp15_question_expansion.py \
      --source-pool "$SOURCE_POOL" \
      --source-manifest "$SOURCE_MANIFEST" \
      --existing-dataset "$ORIGINAL_DATASET" \
      --existing-dataset "$CONSISTENT28_DATASET" \
      --existing-dataset "$PRIOR60_DATASET" \
      --output-dataset "$QUESTION_FILTER" \
      --output-audit "$QUESTION_FILTER_AUDIT" \
      --expected-existing-questions 50 \
      --expected-eligible-questions 94 \
      --expected-new-questions 71
  chmod 444 "$QUESTION_FILTER" "$QUESTION_FILTER_AUDIT"
elif [[ ! -s "$QUESTION_FILTER" || ! -s "$QUESTION_FILTER_AUDIT" ]]; then
  set_status failed "partial frozen question filter"
  exit 4
fi

stable=0
while [[ "$stable" -lt 3 ]]; do
  used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk '/^[0-9]+$/ {print}' | paste -sd, - || true)
  if [[ -n "$used" && "$used" -le 512 && -z "$pids" ]]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  set_status waiting_gpu_stability "samples=$stable/3 mib=${used:-unknown} pids=${pids:-none}"
  if [[ "$stable" -lt 3 ]]; then sleep 10; fi
done

COMMON_ARGS=(
  --source-pool "$SOURCE_POOL"
  --source-manifest "$SOURCE_MANIFEST"
  --question-ids-dataset "$QUESTION_FILTER"
  --model-path "$MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --candidate-count 4
  --candidate-draws 8
  --continuation-count 2
  --max-pairs-per-anchor 2
  --temperature 0.7
  --top-p 0.95
  --max-steps 30
  --max-new-tokens 1024
  --max-context-tokens 8192
  --history-turns 4
  --seed 40404
  --gpu-memory-utilization 0.82
  --scheduler dynamic
  --anchor-window 8
  --max-num-batched-tokens 8192
  --max-num-seqs 32
  --limit-trajectories 0
  --anchors-per-trajectory 30
)

if [[ ! -s "$NEW_MANIFEST" ]]; then
  set_status generating "new_questions=71 expected_anchors=512 scheduler=dynamic anchor_window=8 batched_tokens=8192 max_num_seqs=32 gpu_memory=0.82 benchmark_selected_profile"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
    "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
      "${COMMON_ARGS[@]}" \
      --output-dir "$GENERATION_DIR"
fi

manifest_status=$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$NEW_MANIFEST"
)
if [[ "$manifest_status" != generated_verified_diagnostic ]]; then
  set_status failed "generation_manifest_status=$manifest_status"
  exit 5
fi
test -s "$NEW_STRICT_DATASET"

new_summary=$(
  "$PYTHON_BIN" -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; print("pairs={} questions={}".format(len(rows), len({str(x["question_id"]) for x in rows})))' "$NEW_STRICT_DATASET"
)
set_status materializing_union "$new_summary existing_unique_pairs=116 existing_questions=50"
if [[ ! -e "$UNION_DATASET" && ! -e "$UNION_AUDIT" ]]; then
  PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
    "$PYTHON_BIN" src/rl/action_dpo/build_multisource_strict_exp15_union.py \
      --audited-source original36 "$ORIGINAL_DATASET" "$ORIGINAL_AUDIT" \
      --audited-source consistent28 "$CONSISTENT28_DATASET" "$CONSISTENT28_AUDIT" \
      --branch-source prior_strict60 "$PRIOR60_DATASET" "$PRIOR60_MANIFEST" \
      --branch-source new_question_strict "$NEW_STRICT_DATASET" "$NEW_MANIFEST" \
      --output-dataset "$UNION_DATASET" \
      --output-audit "$UNION_AUDIT" \
      --minimum-total-pairs 140 \
      --minimum-total-questions 75 \
      --minimum-last-source-questions 25
  chmod 444 "$UNION_DATASET" "$UNION_AUDIT"
elif [[ ! -s "$UNION_DATASET" || ! -s "$UNION_AUDIT" ]]; then
  set_status failed "partial frozen training union"
  exit 6
fi

union_summary=$(
  "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("pairs={} questions={} steps={}".format(d["verified_pairs"], d["questions"], d["optimizer_steps_expected"]))' "$UNION_AUDIT"
)
set_status training "$union_summary beta=0.1 lr=1e-6"
DATA_DIR="$TRAINING_DIR" RUNTIME="$RUNTIME" OUTPUT_ROOT="$OUTPUT_ROOT" \
DATASET="$UNION_DATASET" AUDIT="$UNION_AUDIT" ARTIFACT="$ARTIFACT" \
EXPERIMENT_NAME="$CANDIDATE_NAME" STATUS="$TRAIN_STATUS" TRAIN_LOG="$TRAIN_LOG" \
COMPLETION_AUDIT="$COMPLETION_AUDIT" SELECTED_OUTPUT="$SELECTED_OUTPUT" \
SEQUENTIAL_PAIR_SCORING=1 MEMORY_SAFE_DPO_BACKWARD=1 SELECTED_TOOL_LOGITS=1 \
ATTENTION_IMPLEMENTATION=flex_attention \
CPU_OFFLOAD_OPTIMIZER_STATE=1 \
TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash "$RUNTIME/src/rl/experiments/run_exp15_exact_prefix_expanded_train_newgnn.sh" "$GPU_ID"

SELECTED_DIR=$(sed -n '1p' "$SELECTED_OUTPUT")
test -f "$SELECTED_DIR/final/adapter_model.safetensors"
ADAPTER_SHA256=$(sha256sum "$SELECTED_DIR/final/adapter_model.safetensors" | awk '{print $1}')
if [[ -s "$ORIGINAL_DECISION_INPUT" ]]; then
  cp "$ORIGINAL_DECISION_INPUT" "$EVALUATION_DIR/original_exp15_decision_requirements.json"
fi

export RTX3090_EVAL_GPU_MEMORY_UTILIZATION=0.90
export RTX3090_EVAL_MAX_MODEL_LEN=8192
export RTX3090_EVAL_MAX_NUM_BATCHED_TOKENS=8192
export RTX3090_EVAL_MAX_INFLIGHT=24
export RTX3090_EVAL_MAX_NUM_SEQS=24
export RTX3090_GREEDY_WORKERS=24
export RTX3090_GREEDY_SAMPLE_WORKERS=1

set_status evaluating "gpu=$GPU_ID serial=prefix_then_dev1534 max_inflight=24 greedy_workers=24"
OUTPUT_ROOT="$GLOBAL_OUTPUT_ROOT" EXPERIMENT_ROOT="$OUTPUT_ROOT" \
LAUNCH_RUNTIME="$RUNTIME" SCORE_RUNTIME="$RUNTIME" CANDIDATE_NAME="$CANDIDATE_NAME" \
ADAPTER="$SELECTED_DIR/final" ADAPTER_SHA256="$ADAPTER_SHA256" \
PREFIX_GPU_ID="$GPU_ID" GREEDY_GPU_ID="$GPU_ID" SERIAL_EVALUATION=1 PORT="$PORT" \
STATUS="$EVAL_STATUS" RUN_LOG="$EVAL_LOG" \
  bash "$RUNTIME/src/rl/experiments/run_exp15_expanded_stage1_evaluation_newgnn.sh"

set_status complete "decision=$EVALUATION_DIR/decision_requirements.json comparison=$EVALUATION_DIR/comparison_vs_original_exp15.json"
trap - EXIT
