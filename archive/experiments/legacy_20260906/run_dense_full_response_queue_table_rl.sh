#!/usr/bin/env bash
# Two independent SFT2-initialized dense full-response experiments and full-dev greedy evaluations.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
ROOT=${PHASE_ROOT:-$O/phase8_controlled_20260801}
VIEWS_STATUS=${VIEWS_STATUS:-$O/logs/dense_outcome_views_table_rl_20260802.status}
STATUS=${STATUS:-$O/logs/dense_full_response_queue_table_rl_20260802.status}
RUN_LOG=${RUN_LOG:-$O/logs/dense_full_response_queue_table_rl_20260802.log}
LOCK=${LOCK:-$O/logs/dense_full_response_queue_table_rl_20260802.lock}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}

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
mkdir -p "$O/logs" "$ROOT"
exec 9>"$LOCK"; flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

run_lane() {
  local name=$1 config=$2 artifact=$3 pool=$4 gpu=$5 port=$6
  local train_status="$O/logs/stage1_${name}.train.status"
  local diag_status="$O/logs/stage1_${name}.diagnostics.status"
  while [[ ! -f "$pool/manifest.json" ]]; do
    local view_state
    view_state=$(state_of "$VIEWS_STATUS")
    if [[ "$view_state" == failed || "$view_state" == blocked ]]; then
      return 3
    fi
    sleep 15
  done
  EXPERIMENT_NAME="$name" EXPERIMENT_CONFIG="$TR/src/rl/configs/experiments/$config" \
  ARTIFACT="$artifact" STATUS="$train_status" \
  RUN_LOG="$O/logs/stage1_${name}.train.log" TRAIN_RUNTIME="$TR" \
  OUTPUT_ROOT="$O" POOL_DIR="$pool" GPU_ID="$gpu" \
    bash "$TR/src/rl/experiments/run_fixed_pool_controlled_train_table_rl.sh"
  [[ "$(state_of "$train_status")" == complete ]] || return 5
  CANDIDATE_NAME="$name" ADAPTER="$O/checkpoints/$artifact/final" \
  STATUS="$diag_status" RUN_LOG="$O/logs/stage1_${name}.diagnostics.log" \
  TRAIN_RUNTIME="$TR" EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" \
  DIAGNOSTIC_GPU_ID="$gpu" PORT="$port" \
    bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
  [[ "$(state_of "$diag_status")" == complete ]]
}

uniform_pool="$ROOT/balanced_mixed60_dense_uniform_seed101"
strategic_pool="$ROOT/balanced_mixed60_dense_strategic_seed101"
uniform_artifact=trl-transition-v26-exp16-dense-uniform-full-response-sft2-60xk4-seed101-20260802
strategic_artifact=trl-transition-v26-exp17-dense-strategic-full-response-sft2-60xk4-seed101-20260802

set_status parallel_lanes "each lane waits only for its own manifest; Exp16 gpu1; Exp17 gpu0"
run_lane exp16_dense_uniform_full_response exp16_dense_uniform_full_response.yaml \
  "$uniform_artifact" "$uniform_pool" 1 18083 &
pid16=$!
run_lane exp17_dense_strategic_full_response exp17_dense_strategic_full_response.yaml \
  "$strategic_artifact" "$strategic_pool" 0 18082 &
pid17=$!
set +e
wait "$pid16"; code16=$?
wait "$pid17"; code17=$?
set -e
if [[ "$code16" -ne 0 || "$code17" -ne 0 ]]; then
  set_status failed "lane failures exp16=$code16 exp17=$code17"
  exit 5
fi
set_status complete "Exp16 and Exp17 trained from SFT2 and full-dev greedy diagnostics complete"
