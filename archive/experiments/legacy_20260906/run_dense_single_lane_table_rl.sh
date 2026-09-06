#!/usr/bin/env bash
# Recoverable single-candidate lane used when one parallel child exited before its view froze.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
VIEWS_STATUS=${VIEWS_STATUS:-$O/logs/dense_outcome_views_table_rl_20260802.status}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
for variable in CANDIDATE_NAME EXPERIMENT_CONFIG ARTIFACT POOL_DIR GPU_ID PORT STATUS RUN_LOG; do
  [[ -n "${!variable:-}" ]] || { printf 'missing %s\n' "$variable" >&2; exit 2; }
done
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT
mkdir -p "$O/logs"
exec 9>"$O/logs/${ARTIFACT}.lane.lock"; flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

while [[ ! -f "$POOL_DIR/manifest.json" ]]; do
  view_state=$(state_of "$VIEWS_STATUS")
  if [[ "$view_state" == failed || "$view_state" == blocked ]]; then
    set_status blocked "dense view state=$view_state"; exit 3
  fi
  set_status waiting_view "candidate=$CANDIDATE_NAME"; sleep 15
done

train_status="$O/logs/stage1_${CANDIDATE_NAME}.train.status"
set_status training_or_waiting_gpu "candidate=$CANDIDATE_NAME gpu=$GPU_ID"
EXPERIMENT_NAME="$CANDIDATE_NAME" EXPERIMENT_CONFIG="$TR/src/rl/configs/experiments/$EXPERIMENT_CONFIG" \
ARTIFACT="$ARTIFACT" STATUS="$train_status" \
RUN_LOG="$O/logs/stage1_${CANDIDATE_NAME}.train.log" TRAIN_RUNTIME="$TR" \
OUTPUT_ROOT="$O" POOL_DIR="$POOL_DIR" GPU_ID="$GPU_ID" \
  bash "$TR/src/rl/experiments/run_fixed_pool_controlled_train_table_rl.sh"
[[ "$(state_of "$train_status")" == complete ]] || exit 5

diag_status="$O/logs/stage1_${CANDIDATE_NAME}.diagnostics.status"
set_status diagnostics "candidate=$CANDIDATE_NAME gpu=$GPU_ID full_dev=1534"
CANDIDATE_NAME="$CANDIDATE_NAME" ADAPTER="$O/checkpoints/$ARTIFACT/final" \
STATUS="$diag_status" RUN_LOG="$O/logs/stage1_${CANDIDATE_NAME}.diagnostics.log" \
TRAIN_RUNTIME="$TR" EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" \
DIAGNOSTIC_GPU_ID="$GPU_ID" PORT="$PORT" \
  bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
[[ "$(state_of "$diag_status")" == complete ]] || exit 5
set_status complete "candidate=$CANDIDATE_NAME train+full_dev_greedy complete"
