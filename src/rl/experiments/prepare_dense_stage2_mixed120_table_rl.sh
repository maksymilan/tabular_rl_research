#!/usr/bin/env bash
# Expand the deterministic mixed search, then freeze balanced mixed120 dense views.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_dense_stage2_20260802}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
BASE=${BASE_CANDIDATE_DIR:-$O/phase8_mixed_pool_search_20260801/candidate_480_seed101}
CANDIDATE=${CANDIDATE_DIR:-$O/phase8_mixed_pool_search_20260802/candidate_960_seed101}
ROOT=${STAGE2_ROOT:-$O/phase8_dense_stage2_20260802}
RAW=${RAW_POOL_DIR:-$ROOT/balanced_mixed120_raw_seed101}
RANK=${RANK_POOL_DIR:-$ROOT/balanced_mixed120_rank_seed101}
UNIFORM=${UNIFORM_POOL_DIR:-$ROOT/balanced_mixed120_dense_uniform_seed101}
STRATEGIC=${STRATEGIC_POOL_DIR:-$ROOT/balanced_mixed120_dense_strategic_seed101}
STATUS=${STATUS:-$O/logs/dense_stage2_mixed120_prepare_table_rl_20260802.status}
RUN_LOG=${RUN_LOG:-$O/logs/dense_stage2_mixed120_prepare_table_rl_20260802.log}
LOCK=${LOCK:-$O/logs/dense_stage2_mixed120_prepare_table_rl_20260802.lock}
MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
SFT2=$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682
REWARD=$R/src/rl/configs/simple_process_reward_no_backslice.json

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT
mkdir -p "$O/logs" "$CANDIDATE" "$ROOT"
exec 9>"$LOCK"; flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

if [[ -f "$UNIFORM/manifest.json" && -f "$STRATEGIC/manifest.json" ]]; then
  set_status complete "uniform=$UNIFORM strategic=$STRATEGIC"
  exit 0
fi
for required in tasks.jsonl task_selection_manifest.json; do
  [[ -f "$BASE/$required" ]] || { set_status failed "missing base candidate $BASE/$required"; exit 4; }
done

if [[ ! -f "$CANDIDATE/task_selection_manifest.json" ]]; then
  [[ ! -e "$CANDIDATE/tasks.jsonl" ]] || {
    set_status failed "candidate tasks exist without selection manifest"; exit 4;
  }
  mapfile -t sources < <("$PY" - "$BASE/task_selection_manifest.json" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["source"]:
    print(row["path"])
PY
  )
  [[ "${#sources[@]}" -ge 1 ]] || { set_status failed "base selection has no sources"; exit 4; }
  select_args=()
  for source in "${sources[@]}"; do select_args+=(--input "$source"); done
  set_status selecting_candidates "candidate=960 difficulty=320/320/320 seed=101"
  cd "$R"
  "$PY" src/rl/fixed_pool/select_stratified_bird_train.py \
    "${select_args[@]}" --output "$CANDIDATE/tasks.jsonl" \
    --manifest "$CANDIDATE/task_selection_manifest.json" --per-level 320 --seed 101
fi
"$PY" - "$CANDIDATE/tasks.jsonl" "$CANDIDATE/task_selection_manifest.json" <<'PY'
import json,sys
tasks=sum(1 for line in open(sys.argv[1]) if line.strip())
m=json.load(open(sys.argv[2]))
assert tasks == 960 and m["tasks"] == 960
assert m["difficulty_counts"] == {"challenging":320,"moderate":320,"simple":320}
PY

reuse="$BASE/groups"
if [[ -d "$CANDIDATE/groups" ]] && find "$CANDIDATE/groups" -maxdepth 1 -name '*.json' -print -quit | grep -q .; then
  reuse="$CANDIDATE/groups"
fi
set_status preparing_shards "candidate=960 reuse=$reuse shards=2"
cd "$R"
"$PY" src/rl/fixed_pool/prepare_mixed_pool_search.py \
  --tasks "$CANDIDATE/tasks.jsonl" --output-dir "$CANDIDATE" \
  --reuse-groups "$reuse" --group-size 4 --shards 2

run_shard() {
  local shard=$1 gpu=$2
  local ids="$CANDIDATE/shards/shard-${shard}.ids"
  local status="$O/logs/dense_stage2_candidate960_shard${shard}_gpu${gpu}.status"
  local log="$O/logs/dense_stage2_candidate960_shard${shard}_gpu${gpu}.log"
  if [[ ! -s "$ids" ]]; then
    printf '%s\tcomplete\tassigned=0\n' "$(timestamp)" >"$status"
    return 0
  fi
  RUNTIME="$R" HOST_ROLE=table_rl GPU_ID="$gpu" TASKS="$CANDIDATE/tasks.jsonl" \
  TASK_ID_FILE="$ids" OUTPUT_DIR="$CANDIDATE" STATUS="$status" RUN_LOG="$log" \
  OUTPUT_ROOT="$O" bash "$R/src/rl/experiments/run_mixed_pool_search_shard.sh"
}

set_status generating "candidate=960 gpu0/1 disjoint shards"
run_shard 00 0 & pid0=$!
run_shard 01 1 & pid1=$!
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
if [[ "$code0" -ne 0 || "$code1" -ne 0 ]]; then
  set_status failed "generation shards failed gpu0=$code0 gpu1=$code1"
  exit 5
fi

set_status validating_groups "candidate=960 exact K4/sequence/sample_index"
"$PY" src/rl/fixed_pool/validate_mixed_pool_groups.py \
  --tasks "$CANDIDATE/tasks.jsonl" --groups-dir "$CANDIDATE/groups" \
  --output "$CANDIDATE/group_integrity_audit.json" --group-size 4

set +e
"$PY" src/rl/fixed_pool/summarize_mixed_outcome_pool.py \
  --tasks "$CANDIDATE/tasks.jsonl" --groups-dir "$CANDIDATE/groups" \
  --summary "$CANDIDATE/mixed_outcome_summary.json" \
  --selected-tasks "$CANDIDATE/balanced_mixed120_tasks.jsonl" \
  --selection-manifest "$CANDIDATE/balanced_mixed120_selection_manifest.json" \
  --required-per-level 40 --group-size 4 --require-complete
summary_code=$?
set -e
if [[ "$summary_code" -eq 2 ]]; then
  set_status blocked "candidate960 complete but fewer than 40 mixed tasks in at least one difficulty"
  exit 0
elif [[ "$summary_code" -ne 0 ]]; then
  exit "$summary_code"
fi

if [[ ! -f "$RAW/manifest.pending.json" ]]; then
  if [[ ! -e "$RAW" ]]; then
    set_status materializing "balanced mixed120 raw K4"
    "$PY" src/rl/fixed_pool/prepare_mixed_pool_search.py \
      --tasks "$CANDIDATE/balanced_mixed120_tasks.jsonl" --output-dir "$RAW" \
      --reuse-groups "$CANDIDATE/groups" --group-size 4 --shards 1
    cp "$CANDIDATE/balanced_mixed120_tasks.jsonl" "$RAW/tasks.jsonl"
    cp "$CANDIDATE/balanced_mixed120_selection_manifest.json" \
      "$RAW/task_selection_manifest.json"
  else
    for required in tasks.jsonl task_selection_manifest.json groups; do
      [[ -e "$RAW/$required" ]] || { set_status failed "partial raw pool missing=$required"; exit 4; }
    done
  fi
  "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
    --model-path "$MODEL" --adapter-path "$SFT2" --tasks "$RAW/tasks.jsonl" \
    --output-dir "$RAW" --group-size 4 --temperature 0.7 --top-p 0.95 \
    --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
    --history-turns 4 --seed 101 --finalize-only
fi

if [[ ! -f "$RANK/manifest.json" ]]; then
  if [[ ! -e "$RANK" ]]; then
    mkdir -p "$RANK"
    cp "$RAW/tasks.jsonl" "$RAW/trajectories.jsonl" "$RAW/manifest.pending.json" \
      "$RAW/task_selection_manifest.json" "$RANK/"
  else
    [[ -f "$RANK/tasks.jsonl" && -f "$RANK/trajectories.jsonl" ]] || {
      set_status failed "partial rank120 pool requires audit"; exit 4;
    }
  fi
  set_status validating_rank "mixed120 local features; no counterfactual requirement"
  "$PY" src/rl/fixed_pool/validate_counterfactual_pool.py \
    --trajectories "$RANK/trajectories.jsonl" --process-reward-config "$REWARD" \
    --admission-mode rank-only --output-dir "$RANK" --workers 4
  "$PY" src/rl/fixed_pool/freeze_pool_manifest.py \
    --pool-dir "$RANK" --process-reward-config "$REWARD" --admission-mode rank-only
fi

set_status building_dense_views "mixed120 uniform+strategic no consistency screening"
OUTPUT_ROOT="$O" TRAIN_RUNTIME="$R" PHASE_ROOT="$ROOT" \
RAW_POOL_DIR="$RAW" FEATURE_SOURCE_POOL_DIR="$RANK" \
UNIFORM_POOL_DIR="$UNIFORM" STRATEGIC_POOL_DIR="$STRATEGIC" \
STATUS="$O/logs/dense_stage2_mixed120_views_table_rl_20260802.status" \
RUN_LOG="$O/logs/dense_stage2_mixed120_views_table_rl_20260802.log" \
LOCK="$O/logs/dense_stage2_mixed120_views_table_rl_20260802.lock" \
EXPECTED_TASKS=120 EXPECTED_TRAJECTORIES=480 \
  bash "$R/src/rl/experiments/prepare_dense_outcome_views_table_rl.sh"
[[ -f "$UNIFORM/manifest.json" && -f "$STRATEGIC/manifest.json" ]] || {
  set_status failed "dense120 views did not freeze"; exit 5;
}
set_status complete "candidate=960 raw=$RAW uniform=$UNIFORM strategic=$STRATEGIC"
