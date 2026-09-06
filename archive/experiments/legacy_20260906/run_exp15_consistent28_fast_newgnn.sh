#!/usr/bin/env bash
# Materialize the existing 28 non-overlap stable pairs, train, and evaluate serially on GPU6.
set -euo pipefail

GLOBAL_OUTPUT_ROOT=${GLOBAL_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$GLOBAL_OUTPUT_ROOT/rl_runtime_exp15_strict4_20260803}
SOURCE_ROOT=${SOURCE_ROOT:-$GLOBAL_OUTPUT_ROOT/exp15_exact_prefix_branch_20260802}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$SOURCE_ROOT/consistent28_fast_20260803}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
GPU_ID=${GPU_ID:-6}
PORT=${PORT:-18088}

TRAINING_DIR=$EXPERIMENT_ROOT/training
DATASET=$TRAINING_DIR/nonoverlap_online_consistent28.verified.jsonl
AUDIT=$TRAINING_DIR/nonoverlap_online_consistent28.verification_audit.json
ORIGINAL_DATASET=$SOURCE_ROOT/input/original_exp15/fixed_prefix_train.verified.jsonl
ORIGINAL_AUDIT=$SOURCE_ROOT/input/original_exp15/verification_audit.json
CONSISTENT_DATASET=$SOURCE_ROOT/scale48_stage5/exact_prefix_pairs.online_consistent.jsonl
CONSISTENT_MANIFEST=$SOURCE_ROOT/scale48_stage5/manifest.json
SOURCE_REPLAY_AUDIT=$SOURCE_ROOT/scale48_stage5/source_replay_audit.json

CANDIDATE_NAME=exp15_online_consistent28_only_action_dpo
ARTIFACT=trl-transition-v26-exp15-online-consistent28-only-action-dpo-sft2-seed101-20260803
STATUS=$SOURCE_ROOT/logs/consistent28_fast_pipeline.status
RUN_LOG=$SOURCE_ROOT/logs/consistent28_fast_pipeline.run.log
TRAIN_STATUS=$SOURCE_ROOT/logs/consistent28_fast_train.status
TRAIN_LOG=$SOURCE_ROOT/logs/consistent28_fast_train.run.log
EVAL_STATUS=$SOURCE_ROOT/logs/consistent28_fast_evaluation.status
EVAL_LOG=$SOURCE_ROOT/logs/consistent28_fast_evaluation.run.log
SELECTED_OUTPUT=$TRAINING_DIR/selected_training_output.txt
COMPLETION_AUDIT=$TRAINING_DIR/training_completion_audit.json
LOCK=$STATUS.lock

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$SOURCE_ROOT/logs" "$TRAINING_DIR" "$EXPERIMENT_ROOT/evaluation"
exec >>"$RUN_LOG" 2>&1
exec 9>"$LOCK"
flock -n 9 || exit 0

if [[ "$GPU_ID" != 6 ]]; then
  set_status failed "this user-authorized run is restricted to GPU6, got=$GPU_ID"
  exit 2
fi

cd "$RUNTIME"
if [[ ! -e "$DATASET" && ! -e "$AUDIT" ]]; then
  set_status materializing "existing_only=1 consistent=31 overlap=3 expected_pairs=28"
  "$PYTHON_BIN" src/rl/action_dpo/build_nonoverlap_online_consistent_set.py \
    --original-dataset "$ORIGINAL_DATASET" \
    --original-audit "$ORIGINAL_AUDIT" \
    --consistent-dataset "$CONSISTENT_DATASET" \
    --consistent-manifest "$CONSISTENT_MANIFEST" \
    --source-replay-audit "$SOURCE_REPLAY_AUDIT" \
    --output-dataset "$DATASET" \
    --output-audit "$AUDIT" \
    --min-online-trials 2 \
    --expected-pairs 28 \
    --expected-questions 19
  chmod 444 "$DATASET" "$AUDIT"
elif [[ ! -s "$DATASET" || ! -s "$AUDIT" ]]; then
  set_status failed "partial materialization requires audit"
  exit 4
fi

stable=0
while [[ "$stable" -lt 10 ]]; do
  used=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
  pids=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk '/^[0-9]+$/ {print}' | paste -sd, - || true)
  if [[ -n "$used" && "$used" -le 512 && -z "$pids" ]]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  set_status waiting_gpu_stability "gpu=6 samples=$stable/10 mib=${used:-unknown} pids=${pids:-none}"
  if [[ "$stable" -lt 10 ]]; then sleep 30; fi
done

set_status training "gpu=6 pairs=28 questions=19 beta=0.1 lr=1e-6"
DATA_DIR="$TRAINING_DIR" \
RUNTIME="$RUNTIME" \
OUTPUT_ROOT="$SOURCE_ROOT" \
DATASET="$DATASET" \
AUDIT="$AUDIT" \
ARTIFACT="$ARTIFACT" \
EXPERIMENT_NAME="$CANDIDATE_NAME" \
STATUS="$TRAIN_STATUS" \
TRAIN_LOG="$TRAIN_LOG" \
COMPLETION_AUDIT="$COMPLETION_AUDIT" \
SELECTED_OUTPUT="$SELECTED_OUTPUT" \
  bash "$RUNTIME/src/rl/experiments/run_exp15_exact_prefix_expanded_train_newgnn.sh" 6

SELECTED_DIR=$(sed -n '1p' "$SELECTED_OUTPUT")
test -f "$SELECTED_DIR/final/adapter_model.safetensors"
ADAPTER_SHA256=$(sha256sum "$SELECTED_DIR/final/adapter_model.safetensors" | awk '{print $1}')
cp "$SOURCE_ROOT/evaluation/original_exp15_decision_requirements.json" \
  "$EXPERIMENT_ROOT/evaluation/original_exp15_decision_requirements.json"

# Exact settings selected by the frozen RTX3090 full-dev saturation profile.
export RTX3090_EVAL_GPU_MEMORY_UTILIZATION=0.90
export RTX3090_EVAL_MAX_MODEL_LEN=8192
export RTX3090_EVAL_MAX_NUM_BATCHED_TOKENS=8192
export RTX3090_EVAL_MAX_INFLIGHT=24
export RTX3090_EVAL_MAX_NUM_SEQS=24
export RTX3090_GREEDY_WORKERS=24
export RTX3090_GREEDY_SAMPLE_WORKERS=1

set_status evaluating "gpu=6 serial=prefix_then_dev1534 greedy_workers=24 max_inflight=24"
OUTPUT_ROOT="$GLOBAL_OUTPUT_ROOT" \
EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
LAUNCH_RUNTIME="$RUNTIME" \
CANDIDATE_NAME="$CANDIDATE_NAME" \
ADAPTER="$SELECTED_DIR/final" \
ADAPTER_SHA256="$ADAPTER_SHA256" \
PREFIX_GPU_ID=6 GREEDY_GPU_ID=6 SERIAL_EVALUATION=1 PORT="$PORT" \
STATUS="$EVAL_STATUS" RUN_LOG="$EVAL_LOG" \
  bash "$RUNTIME/src/rl/experiments/run_exp15_expanded_stage1_evaluation_newgnn.sh"

set_status complete "decision=$EXPERIMENT_ROOT/evaluation/decision_requirements.json"
trap - EXIT
