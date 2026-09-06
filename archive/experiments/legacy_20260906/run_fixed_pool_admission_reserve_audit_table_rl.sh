#!/usr/bin/env bash
# Build counterfactual suites and audit reserve-task admission without weakening the gate.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
C=${CANDIDATE_DIR:-$O/phase8_controlled_20260801/fixed_pool_60_seed101_admission_reserve_v1}
CF=$C/counterfactual_suite
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
PROCESS_CONFIG=${PROCESS_CONFIG:-$R/src/rl/configs/simple_process_reward_no_backslice.json}
PRIOR_AUDIT=${PRIOR_AUDIT:-$O/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/process_rl_mandatory_gates.audit.json}
GENERATION_STATUS=$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.status
STATUS=$O/logs/fixed_pool_admission_reserve_audit_table_rl_20260801.status
RUN_LOG=$O/logs/fixed_pool_admission_reserve_audit_table_rl_20260801.log
LOCK=$O/logs/fixed_pool_admission_reserve_audit_table_rl_20260801.lock

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
exec 8>"$LOCK"
flock -n 8 || exit 0
exec >>"$RUN_LOG" 2>&1
cd "$R"

if [[ -f "$C/admission_summary.json" ]]; then
  "$PYTHON_BIN" - "$C/admission_summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed"
assert all(len(values) == 2 for values in p["selected_replacements"].values())
PY
  set_status complete "already audited replacements=6"
  exit 0
fi

while [[ "$(state_of "$GENERATION_STATUS")" != complete ]]; do
  state=$(state_of "$GENERATION_STATUS")
  if [[ "$state" == failed ]]; then set_status blocked "generation failed"; exit 3; fi
  set_status waiting_generation "state=$state"
  sleep 30
done

[[ "$(wc -l < "$C/trajectories.jsonl")" -eq 48 ]]
if [[ ! -f "$CF/counterfactual_suite.passed.json" ]]; then
  if [[ -e "$CF/quality_audit.json" ]]; then
    set_status failed "partial counterfactual output exists; audit before resume"
    exit 4
  fi
  set_status building_counterfactuals "reserve correct trajectories only"
  "$PYTHON_BIN" src/rl/scenarios/fixed_pool/build_counterfactual_suite_for_fixed_pool.py \
    --tasks "$C/tasks.jsonl" --trajectories "$C/trajectories.jsonl" \
    --output-dir "$CF" --prior-grounding-audit "$PRIOR_AUDIT" --workers 2
fi

set_status validating "workers=4 unchanged completeness gate"
set +e
"$PYTHON_BIN" src/rl/fixed_pool/validate_counterfactual_pool.py \
  --trajectories "$C/trajectories.jsonl" \
  --process-reward-config "$PROCESS_CONFIG" \
  --counterfactual-manifest "$CF/counterfactual_suite.passed.json" \
  --output-dir "$C" --workers 4
validation_exit=$?
set -e
if [[ "$validation_exit" -ne 0 && ! -f "$C/counterfactual_results.jsonl" ]]; then
  set_status failed "validation exit=$validation_exit without auditable results"
  exit "$validation_exit"
fi

set_status summarizing "require=2 causally admitted tasks per difficulty"
"$PYTHON_BIN" src/rl/scenarios/fixed_pool/summarize_admission_reserve.py \
  --tasks "$C/tasks.jsonl" --trajectories "$C/trajectories.jsonl" \
  --counterfactual-results "$C/counterfactual_results.jsonl" \
  --output "$C/admission_summary.json" --required-per-level 2
set_status complete "causal replacements=6 ready_for_pool_v2"
