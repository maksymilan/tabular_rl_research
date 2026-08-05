#!/usr/bin/env bash
# Convert the generated SFT2 K4 pool into an immutable, counterfactually gated pool.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_rank_score_v3_20260731}
POOL_DIR=${POOL_DIR:-$OUTPUT_ROOT/phase8_controlled_20260801/fixed_pool_60_seed101}
CF_DIR=${CF_DIR:-$POOL_DIR/counterfactual_suite}
GENERATION_STATUS=${GENERATION_STATUS:-$OUTPUT_ROOT/logs/fixed_pool60_generation_table_rl_20260801.status}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/fixed_pool60_validation_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/fixed_pool60_validation_table_rl_20260801.log}
LOCK=${LOCK:-$OUTPUT_ROOT/logs/fixed_pool60_validation_table_rl_20260801.lock}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
PROCESS_CONFIG=${PROCESS_CONFIG:-$RUNTIME/src/rl/configs/simple_process_reward_no_backslice.json}
PRIOR_AUDIT=${PRIOR_AUDIT:-$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/process_rl_mandatory_gates.audit.json}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf 'missing\n'; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs"
exec 8>"$LOCK"
if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"

if [[ -f "$POOL_DIR/manifest.json" ]]; then
  "$PYTHON_BIN" - "$POOL_DIR/manifest.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["schema_version"] == "table-agent-fixed-rollout-pool-v1"
assert p["status"] == "frozen_passed"
assert p["tasks"] == 60 and p["trajectories"] == 240 and p["group_size"] == 4
PY
  set_status complete "already frozen tasks=60 trajectories=240"
  exit 0
fi

while true; do
  generation=$(state_of "$GENERATION_STATUS")
  case "$generation" in
    complete) break ;;
    failed) set_status blocked "generation failed; audit before validation"; exit 3 ;;
    *) set_status waiting_generation "state=$generation"; sleep 60 ;;
  esac
done

[[ -f "$POOL_DIR/manifest.pending.json" ]]
[[ -f "$POOL_DIR/trajectories.jsonl" ]]
[[ "$(wc -l < "$POOL_DIR/trajectories.jsonl")" -eq 240 ]]

if [[ ! -f "$CF_DIR/counterfactual_suite.passed.json" ]]; then
  if [[ -e "$CF_DIR/quality_audit.json" ]]; then
    set_status failed "partial counterfactual output exists; fail closed for audit"
    exit 4
  fi
  set_status building_counterfactuals "correct trajectories only; two DBs per task"
  "$PYTHON_BIN" src/rl/fixed_pool/build_counterfactual_suite_for_fixed_pool.py \
    --tasks "$POOL_DIR/tasks.jsonl" \
    --trajectories "$POOL_DIR/trajectories.jsonl" \
    --output-dir "$CF_DIR" \
    --prior-grounding-audit "$PRIOR_AUDIT" \
    --workers 2
fi

set_status validating_process_credit "replay correct trajectories on counterfactual DBs"
"$PYTHON_BIN" src/rl/fixed_pool/validate_counterfactual_pool.py \
  --trajectories "$POOL_DIR/trajectories.jsonl" \
  --process-reward-config "$PROCESS_CONFIG" \
  --counterfactual-manifest "$CF_DIR/counterfactual_suite.passed.json" \
  --output-dir "$POOL_DIR" \
  --workers 4

set_status freezing "hashing 60xK4 pool and making training inputs read-only"
"$PYTHON_BIN" src/rl/fixed_pool/freeze_pool_manifest.py \
  --pool-dir "$POOL_DIR" \
  --process-reward-config "$PROCESS_CONFIG" \
  --counterfactual-manifest "$CF_DIR/counterfactual_suite.passed.json"

set_status complete "frozen_passed tasks=60 trajectories=240 online_resampling_forbidden=1"
