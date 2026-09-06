#!/usr/bin/env bash
# Conditional scale/replication queue for exactly one winning dense-credit method.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_dense_stage2_20260802}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
STAGE1_ROOT=${STAGE1_ROOT:-$O/phase8_controlled_20260801}
ROOT=${STAGE2_ROOT:-$O/phase8_dense_stage2_20260802}
STATUS=${STATUS:-$O/logs/dense_stage2_scale120_queue_table_rl_20260802.status}
RUN_LOG=${RUN_LOG:-$O/logs/dense_stage2_scale120_queue_table_rl_20260802.log}
LOCK=${LOCK:-$O/logs/dense_stage2_scale120_queue_table_rl_20260802.lock}
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

uniform_status="$O/logs/stage1_exp16_dense_uniform_full_response.diagnostics.status"
strategic_status="$O/logs/stage1_exp17_dense_strategic_full_response.diagnostics.status"
uniform_req="$STAGE1_ROOT/exp16_dense_uniform_full_response/decision_requirements.json"
strategic_req="$STAGE1_ROOT/exp17_dense_strategic_full_response/decision_requirements.json"
if [[ "$(state_of "$uniform_status")" != complete || \
      "$(state_of "$strategic_status")" != complete || \
      ! -f "$uniform_req" || ! -f "$strategic_req" ]]; then
  set_status waiting_stage1 "Exp16=$(state_of "$uniform_status") Exp17=$(state_of "$strategic_status")"
  exit 0
fi

selection="$ROOT/stage1_winner_selection.json"
"$PY" "$TR/src/rl/experiments/select_dense_stage2_winner.py" \
  --uniform "$uniform_req" --strategic "$strategic_req" --output "$selection"
selected=$(
  "$PY" - "$selection" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p.get("selected") or "")
PY
)
if [[ -z "$selected" ]]; then
  set_status complete_no_scaleup "neither dense method passed requirements with positive full-dev delta"
  exit 0
fi

prepare_status="$O/logs/dense_stage2_mixed120_prepare_table_rl_20260802.status"
set_status preparing_mixed120 "selected=$selected candidate960 then 40/40/40 mixed"
OUTPUT_ROOT="$O" TRAIN_RUNTIME="$TR" STAGE2_ROOT="$ROOT" \
STATUS="$prepare_status" RUN_LOG="$O/logs/dense_stage2_mixed120_prepare_table_rl_20260802.log" \
  bash "$TR/src/rl/experiments/prepare_dense_stage2_mixed120_table_rl.sh"
prepare_state=$(state_of "$prepare_status")
if [[ "$prepare_state" == blocked ]]; then
  set_status blocked "mixed120 preparation blocked; see=$prepare_status"
  exit 0
elif [[ "$prepare_state" != complete ]]; then
  set_status failed "mixed120 preparation state=$prepare_state"
  exit 5
fi

train_experiment="exp18_dense_${selected}_full_response_scale120"
config="${train_experiment}.yaml"
pool="$ROOT/balanced_mixed120_dense_${selected}_seed101"
[[ -f "$pool/manifest.json" ]] || { set_status failed "missing frozen selected pool=$pool"; exit 5; }

run_seed() {
  local seed=$1 gpu=$2 port=$3
  local candidate="${train_experiment}_seed${seed}"
  local artifact="trl-transition-v26-${train_experiment//_/-}-sft2-120xk4-trainseed${seed}-20260802"
  local train_status="$O/logs/stage2_${candidate}.train.status"
  local diag_status="$O/logs/stage2_${candidate}.diagnostics.status"
  EXPERIMENT_NAME="$train_experiment" \
  EXPERIMENT_CONFIG="$TR/src/rl/configs/experiments/$config" \
  ARTIFACT="$artifact" STATUS="$train_status" \
  RUN_LOG="$O/logs/stage2_${candidate}.train.log" TRAIN_RUNTIME="$TR" \
  OUTPUT_ROOT="$O" POOL_DIR="$pool" GPU_ID="$gpu" TRAIN_SEED="$seed" \
  EXPECTED_POOL_TASKS=120 EXPECTED_POOL_TRAJECTORIES=480 \
  EXPECTED_POOL_STATUS=frozen_dense_ready \
    bash "$TR/src/rl/experiments/run_fixed_pool_controlled_train_table_rl.sh"
  [[ "$(state_of "$train_status")" == complete ]] || return 5
  CANDIDATE_NAME="$candidate" ADAPTER="$O/checkpoints/$artifact/final" \
  STATUS="$diag_status" RUN_LOG="$O/logs/stage2_${candidate}.diagnostics.log" \
  TRAIN_RUNTIME="$TR" EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" \
  DECISION_ROOT="$ROOT" DIAGNOSTIC_GPU_ID="$gpu" PORT="$port" \
    bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
  [[ "$(state_of "$diag_status")" == complete ]]
}

set_status training_replications "selected=$selected seeds=101,202,303 independent SFT2; gpu0 lane=101+303 gpu1 lane=202"
(run_seed 101 0 18182; run_seed 303 0 18182) & lane0=$!
run_seed 202 1 18183 & lane1=$!
set +e
wait "$lane0"; code0=$?
wait "$lane1"; code1=$?
set -e
if [[ "$code0" -ne 0 || "$code1" -ne 0 ]]; then
  set_status failed "replication lane failures gpu0=$code0 gpu1=$code1"
  exit 5
fi

summary="$ROOT/stage2_scale120_seed_summary.json"
"$PY" "$TR/src/rl/experiments/summarize_dense_stage2_seeds.py" \
  --selection "$selection" \
  --requirement "101:$ROOT/${train_experiment}_seed101/decision_requirements.json" \
  --requirement "202:$ROOT/${train_experiment}_seed202/decision_requirements.json" \
  --requirement "303:$ROOT/${train_experiment}_seed303/decision_requirements.json" \
  --output "$summary"
summary_state=$(
  "$PY" - "$summary" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
set_status complete "selected=$selected scale120 seeds complete summary=$summary_state finalist_k4=deferred"
