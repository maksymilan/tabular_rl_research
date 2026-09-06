#!/usr/bin/env bash
# Freeze one balanced mixed 60-task pool into independent Rank and Process views.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
CANDIDATE=${CANDIDATE_DIR:-$O/phase8_mixed_pool_search_20260801/candidate_480_seed101}
ROOT=${PHASE_ROOT:-$O/phase8_controlled_20260801}
RAW=${RAW_POOL_DIR:-$ROOT/balanced_mixed60_raw_seed101}
RANK=${RANK_POOL_DIR:-$ROOT/balanced_mixed60_rank_seed101}
PROCESS=${PROCESS_POOL_DIR:-$ROOT/balanced_mixed60_process_seed101}
STATUS=${STATUS:-$O/logs/balanced_mixed60_views_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$O/logs/balanced_mixed60_views_table_rl_20260801.log}
MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
SFT2=$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682
REWARD=$R/src/rl/configs/simple_process_reward_no_backslice.json
GROUNDING=${GROUNDING_AUDIT:-$O/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/process_rl_mandatory_gates.audit.json}
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
groups() { find "$CANDIDATE/groups" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d '[:space:]'; }
mkdir -p "$O/logs" "$ROOT"
exec 9>"$O/logs/balanced_mixed60_views_table_rl_20260801.lock"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1
if [[ -f "$RANK/manifest.json" && -f "$PROCESS/manifest.json" ]]; then
  set_status complete "rank=$RANK process=$PROCESS"
  exit 0
fi
if [[ "$(groups)" -ne 480 ]]; then
  set_status waiting_candidate "groups=$(groups)/480"
  exit 0
fi
cd "$R"
set_status selecting "candidate=480 required=20/20/20 mixed"
"$PY" src/rl/scenarios/fixed_pool/summarize_mixed_outcome_pool.py \
  --tasks "$CANDIDATE/tasks.jsonl" --groups-dir "$CANDIDATE/groups" \
  --summary "$CANDIDATE/mixed_outcome_summary.json" \
  --selected-tasks "$CANDIDATE/balanced_mixed60_tasks.jsonl" \
  --selection-manifest "$CANDIDATE/balanced_mixed60_selection_manifest.json" \
  --required-per-level 20 --group-size 4 --require-complete

if [[ ! -f "$RAW/manifest.pending.json" ]]; then
  if [[ ! -e "$RAW" ]]; then
    set_status materializing "raw mixed60 trajectories"
    "$PY" src/rl/scenarios/fixed_pool/prepare_mixed_pool_search.py \
      --tasks "$CANDIDATE/balanced_mixed60_tasks.jsonl" --output-dir "$RAW" \
      --reuse-groups "$CANDIDATE/groups" --group-size 4 --shards 1
    cp "$CANDIDATE/balanced_mixed60_tasks.jsonl" "$RAW/tasks.jsonl"
    cp "$CANDIDATE/balanced_mixed60_selection_manifest.json" "$RAW/task_selection_manifest.json"
  else
    [[ -f "$RAW/tasks.jsonl" && -f "$RAW/task_selection_manifest.json" ]] || {
      set_status failed "partial raw pool lacks immutable task metadata path=$RAW"; exit 4;
    }
    [[ "$(find "$RAW/groups" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d '[:space:]')" -eq 60 ]] || {
      set_status failed "partial raw pool does not contain 60 complete groups"; exit 4;
    }
  fi
  set_status materializing "finalize-only without SQLite task loading"
  "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
    --model-path "$MODEL" --adapter-path "$SFT2" --tasks "$RAW/tasks.jsonl" \
    --output-dir "$RAW" --group-size 4 --temperature 0.7 --top-p 0.95 \
    --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
    --history-turns 4 --seed 101 --finalize-only
fi

seed_view() {
  local target=$1
  if [[ ! -e "$target" ]]; then
    mkdir -p "$target"
    cp "$RAW/tasks.jsonl" "$RAW/trajectories.jsonl" "$RAW/manifest.pending.json" \
      "$RAW/task_selection_manifest.json" "$target/"
  fi
}

if [[ ! -f "$RANK/manifest.json" ]]; then
  seed_view "$RANK"
  set_status validating_rank "local features only; no counterfactual requirement"
  "$PY" src/rl/fixed_pool/validate_counterfactual_pool.py \
    --trajectories "$RANK/trajectories.jsonl" --process-reward-config "$REWARD" \
    --admission-mode rank-only --output-dir "$RANK" --workers 4
  "$PY" src/rl/scenarios/fixed_pool/freeze_pool_manifest.py \
    --pool-dir "$RANK" --process-reward-config "$REWARD" --admission-mode rank-only
fi

if [[ ! -f "$PROCESS/manifest.json" ]]; then
  seed_view "$PROCESS"
  CF="$PROCESS/counterfactual_suite"
  if [[ ! -f "$CF/counterfactual_suite.passed.json" && ! -f "$CF/counterfactual_suite.screening.json" ]]; then
    set_status building_consistency "process correct trajectories; failures will be screened"
    "$PY" src/rl/scenarios/fixed_pool/build_counterfactual_suite_for_fixed_pool.py \
      --tasks "$PROCESS/tasks.jsonl" --trajectories "$PROCESS/trajectories.jsonl" \
      --output-dir "$CF" --prior-grounding-audit "$GROUNDING" --workers 2 \
      --allow-partial-screening
  fi
  CFM="$CF/counterfactual_suite.passed.json"
  [[ -f "$CFM" ]] || CFM="$CF/counterfactual_suite.screening.json"
  set_status validating_process "counterfactual-screened manifest=$CFM"
  "$PY" src/rl/fixed_pool/validate_counterfactual_pool.py \
    --trajectories "$PROCESS/trajectories.jsonl" --process-reward-config "$REWARD" \
    --counterfactual-manifest "$CFM" --admission-mode process-screened \
    --output-dir "$PROCESS" --workers 4
  "$PY" src/rl/scenarios/fixed_pool/freeze_pool_manifest.py \
    --pool-dir "$PROCESS" --process-reward-config "$REWARD" \
    --counterfactual-manifest "$CFM" --admission-mode process-screened
fi
set_status complete "rank=$RANK process=$PROCESS"
