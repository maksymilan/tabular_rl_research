#!/usr/bin/env bash
# Rescore the immutable balanced mixed60 trajectories for Exp16/Exp17 without CF screening.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
ROOT=${PHASE_ROOT:-$O/phase8_controlled_20260801}
RAW=${RAW_POOL_DIR:-$ROOT/balanced_mixed60_raw_seed101}
FEATURE_SOURCE=${FEATURE_SOURCE_POOL_DIR:-$ROOT/balanced_mixed60_rank_seed101}
UNIFORM=${UNIFORM_POOL_DIR:-$ROOT/balanced_mixed60_dense_uniform_seed101}
STRATEGIC=${STRATEGIC_POOL_DIR:-$ROOT/balanced_mixed60_dense_strategic_seed101}
STATUS=${STATUS:-$O/logs/dense_outcome_views_table_rl_20260802.status}
RUN_LOG=${RUN_LOG:-$O/logs/dense_outcome_views_table_rl_20260802.log}
LOCK=${LOCK:-$O/logs/dense_outcome_views_table_rl_20260802.lock}
EXPECTED_TASKS=${EXPECTED_TASKS:-60}
EXPECTED_TRAJECTORIES=${EXPECTED_TRAJECTORIES:-$((EXPECTED_TASKS * 4))}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
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

for required in tasks.jsonl trajectories.jsonl manifest.pending.json task_selection_manifest.json; do
  [[ -f "$RAW/$required" ]] || {
    set_status waiting_raw "missing=$RAW/$required"
    exit 0
  }
done
[[ -f "$FEATURE_SOURCE/validated_trajectories.jsonl" \
  && -f "$FEATURE_SOURCE/manifest.json" ]] || {
  set_status waiting_feature_source "missing frozen rank view=$FEATURE_SOURCE"
  exit 0
}
cmp -s "$RAW/trajectories.jsonl" "$FEATURE_SOURCE/trajectories.jsonl" || {
  set_status failed "rank feature source does not share the immutable raw trajectories"; exit 4;
}
[[ "$(wc -l < "$RAW/tasks.jsonl" | tr -d '[:space:]')" -eq "$EXPECTED_TASKS" ]] || {
  set_status failed "raw tasks must be $EXPECTED_TASKS"; exit 4;
}
[[ "$(wc -l < "$RAW/trajectories.jsonl" | tr -d '[:space:]')" -eq "$EXPECTED_TRAJECTORIES" ]] || {
  set_status failed "raw trajectories must be $EXPECTED_TRAJECTORIES"; exit 4;
}

seed_view() {
  local target=$1
  if [[ ! -e "$target" ]]; then
    mkdir -p "$target"
    cp "$RAW/tasks.jsonl" "$RAW/trajectories.jsonl" "$RAW/manifest.pending.json" \
      "$RAW/task_selection_manifest.json" "$target/"
  fi
  for required in tasks.jsonl trajectories.jsonl manifest.pending.json task_selection_manifest.json; do
    [[ -f "$target/$required" ]] || {
      set_status failed "partial dense view missing=$target/$required"; exit 4;
    }
    cmp -s "$RAW/$required" "$target/$required" || {
      set_status failed "dense view differs from immutable raw source file=$required target=$target"; exit 4;
    }
  done
}

freeze_view() {
  local label target reward feature_source
  label=$1
  target=$2
  reward=$3
  feature_source=${4:-}
  if [[ -f "$target/manifest.json" ]]; then return 0; fi
  seed_view "$target"
  for partial in validated_trajectories.jsonl process_features.jsonl \
    counterfactual_results.jsonl transitions.jsonl validation_summary.json; do
    [[ ! -e "$target/$partial" ]] || {
      set_status failed "partial validation requires audit label=$label file=$target/$partial"; exit 4;
    }
  done
  set_status "validating_${label}" "dense-outcome no counterfactual trajectories=$EXPECTED_TRAJECTORIES"
  cd "$R"
  if [[ -n "$feature_source" ]]; then
    "$PY" src/rl/fixed_pool/rescore_fixed_pool_process_rewards.py \
      --trajectories "$target/trajectories.jsonl" \
      --feature-source "$feature_source" --process-reward-config "$reward" \
      --admission-mode dense-outcome --output-dir "$target"
  else
    "$PY" src/rl/fixed_pool/validate_counterfactual_pool.py \
      --trajectories "$target/trajectories.jsonl" \
      --process-reward-config "$reward" --admission-mode dense-outcome \
      --output-dir "$target" --workers 4
  fi
  "$PY" src/rl/fixed_pool/freeze_pool_manifest.py \
    --pool-dir "$target" --process-reward-config "$reward" \
    --admission-mode dense-outcome
}

freeze_view uniform "$UNIFORM" "$R/src/rl/configs/dense_outcome_uniform.json" \
  "$FEATURE_SOURCE/validated_trajectories.jsonl"
freeze_view strategic "$STRATEGIC" "$R/src/rl/configs/dense_outcome_strategic.json" \
  "$FEATURE_SOURCE/validated_trajectories.jsonl"
set_status complete "uniform=$UNIFORM strategic=$STRATEGIC source=$RAW"
