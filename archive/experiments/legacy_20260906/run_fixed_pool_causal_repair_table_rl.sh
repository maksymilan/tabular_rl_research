#!/usr/bin/env bash
# Assemble, revalidate, and freeze the strict causal replacement pool.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ORIGINAL=$O/phase8_controlled_20260801/fixed_pool_60_seed101
RESERVE=$O/phase8_controlled_20260801/fixed_pool_60_seed101_admission_reserve_v1
POOL=$O/phase8_controlled_20260801/fixed_pool_60_seed101_causal_repair_v2
CF=$POOL/counterfactual_suite
MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
ADAPTER=$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
PROCESS_CONFIG=$R/src/rl/configs/simple_process_reward_no_backslice.json
PRIOR_AUDIT=$O/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/process_rl_mandatory_gates.audit.json
ADMISSION_STATUS=$O/logs/fixed_pool_admission_reserve_audit_table_rl_20260801.status
STATUS=$O/logs/fixed_pool_causal_repair_table_rl_20260801.status
RUN_LOG=$O/logs/fixed_pool_causal_repair_table_rl_20260801.log
LOCK=$O/logs/fixed_pool_causal_repair_table_rl_20260801.lock

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
on_exit() { local c=$?; trap - EXIT; if [[ "$c" -ne 0 ]]; then set_status failed "exit=$c see=$RUN_LOG"; fi; exit "$c"; }
trap on_exit EXIT
mkdir -p "$O/logs"
exec 8>"$LOCK"
flock -n 8 || exit 0
exec >>"$RUN_LOG" 2>&1
cd "$R"

if [[ -f "$POOL/manifest.json" ]]; then
  "$PY" - "$POOL/manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"] == "frozen_passed"
assert p["tasks"] == 60 and p["trajectories"] == 240 and p["group_size"] == 4
PY
  set_status complete "already frozen tasks=60 trajectories=240"
  exit 0
fi
while [[ "$(state_of "$ADMISSION_STATUS")" != complete ]]; do
  state=$(state_of "$ADMISSION_STATUS")
  if [[ "$state" == failed || "$state" == blocked ]]; then set_status blocked "reserve audit=$state"; exit 3; fi
  set_status waiting_admission "state=$state"
  sleep 30
done

if [[ ! -f "$POOL/tasks.jsonl" ]]; then
  set_status assembling "replace six failed groups in-place by difficulty"
  "$PY" src/rl/fixed_pool/assemble_repaired_fixed_pool.py \
    --original-pool "$ORIGINAL" --reserve-pool "$RESERVE" --output-dir "$POOL"
fi
if [[ ! -f "$POOL/manifest.pending.json" ]]; then
  set_status finalizing "assemble 60 groups and exact sequence 0..239"
  "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
    --model-path "$MODEL" --adapter-path "$ADAPTER" --tasks "$POOL/tasks.jsonl" \
    --output-dir "$POOL" --group-size 4 --temperature 0.7 --top-p 0.95 \
    --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
    --history-turns 4 --seed 101 --finalize-only
fi
if [[ ! -f "$CF/counterfactual_suite.passed.json" ]]; then
  set_status building_counterfactuals "workers=2 repaired pool"
  "$PY" src/rl/scenarios/fixed_pool/build_counterfactual_suite_for_fixed_pool.py \
    --tasks "$POOL/tasks.jsonl" --trajectories "$POOL/trajectories.jsonl" \
    --output-dir "$CF" --prior-grounding-audit "$PRIOR_AUDIT" --workers 2
fi
if [[ ! -f "$POOL/validation_summary.json" ]]; then
  set_status validating "workers=4 require every correct trajectory"
  "$PY" src/rl/fixed_pool/validate_counterfactual_pool.py \
    --trajectories "$POOL/trajectories.jsonl" \
    --process-reward-config "$PROCESS_CONFIG" \
    --counterfactual-manifest "$CF/counterfactual_suite.passed.json" \
    --output-dir "$POOL" --workers 4
fi
set_status freezing "strict repaired pool"
"$PY" src/rl/scenarios/fixed_pool/freeze_pool_manifest.py \
  --pool-dir "$POOL" --process-reward-config "$PROCESS_CONFIG" \
  --counterfactual-manifest "$CF/counterfactual_suite.passed.json"
set_status complete "frozen_passed tasks=60 trajectories=240 replacements=6"
