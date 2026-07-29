#!/usr/bin/env bash
# Wait for the two non-empty pass@8 shards, select mixed tasks, audit rewards, and merge the pool.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/rl_runtime_sft2_result_only_v26_20260729
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs
PYTHON=/home/dengyan/miniconda3/envs/sft/bin/python
STATUS="$OUTPUT_ROOT/logs/bird_sft2_v26_train_next50_passk8_prepare_process_pool.status"
LOG="$OUTPUT_ROOT/logs/bird_sft2_v26_train_next50_passk8_prepare_process_pool.log"
FIRST_POOL="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"
COMBINED_POOL="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first100_passk8_mixed_outcomes_nonempty.json"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

rows() {
  local path=$1
  if [[ -f "$path" ]]; then
    wc -l <"$path" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$LOG" 2>&1
cd "$RUNTIME"
export PYTHONPATH="$RUNTIME/src/rl:$RUNTIME/src/eval:$RUNTIME/src/harness:$RUNTIME/src/sft"

for shard in 00 01; do
  expected=24
  if [[ "$shard" == "01" ]]; then expected=25; fi
  result="$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_bird_train_next50_nonempty_shard${shard}_passk8/all.jsonl"
  worker="bird_sft2_v26_train_next50_passk8_shard${shard}"
  while [[ "$(rows "$result")" -ne "$expected" ]]; do
    if ! tmux has-session -t "$worker" 2>/dev/null; then
      worker_status="$OUTPUT_ROOT/logs/bird_sft2_v26_train_next50_nonempty_shard${shard}_passk8.status"
      set_status failed \
        "shard=$shard rows=$(rows "$result")/$expected worker_missing status=$(cat "$worker_status" 2>/dev/null || true)"
      exit 4
    fi
    set_status waiting \
      "shard00=$(rows "$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_bird_train_next50_nonempty_shard00_passk8/all.jsonl")/24 shard01=$(rows "$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_bird_train_next50_nonempty_shard01_passk8/all.jsonl")/25"
    sleep 60
  done
  worker_status="$OUTPUT_ROOT/logs/bird_sft2_v26_train_next50_nonempty_shard${shard}_passk8.status"
  while tmux has-session -t "$worker" 2>/dev/null; do
    sleep 5
  done
  if ! grep -q $'\tcomplete\t' "$worker_status"; then
    set_status failed "shard=$shard rows=$expected worker_exit status=$(cat "$worker_status" 2>/dev/null || true)"
    exit 5
  fi
done

for shard in 00 01; do
  tasks="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard${shard}.json"
  result_root="$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_bird_train_next50_nonempty_shard${shard}_passk8"
  selected="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard${shard}_passk8_mixed.json"
  selected_nonempty="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard${shard}_passk8_mixed_verified_nonempty.json"
  filter_audit="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard${shard}_passk8_mixed_verified_nonempty.audit.json"
  reward_audit="$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_train_next50_nonempty_shard${shard}_passk8_simple_process_audit"

  set_status selecting "shard=$shard"
  "$PYTHON" src/rl/select_passk_rl_examples.py \
    --input "$result_root/all.jsonl" \
    --tasks "$tasks" \
    --output "$selected" \
    --target-pass-k 8 \
    --mode mixed_attempt_outcomes \
    --no-shuffle
  "$PYTHON" src/rl/filter_empty_reference_tasks.py \
    --input "$selected" \
    --output "$selected_nonempty" \
    --audit-output "$filter_audit"

  selected_count="$("$PYTHON" - "$selected_nonempty" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["examples"]))
PY
)"
  if [[ "$selected_count" -gt 0 ]]; then
    set_status auditing "shard=$shard selected=$selected_count"
    "$PYTHON" src/rl/audit_passk_process_rewards.py \
      --passk-all "$result_root/all.jsonl" \
      --tasks-json "$tasks" \
      --selected-json "$selected_nonempty" \
      --config-json src/rl/configs/simple_process_reward.json \
      --output-dir "$reward_audit"
  fi
done

set_status merging "first50_plus_next50"
"$PYTHON" src/rl/merge_rl_task_artifacts.py \
  --input "$FIRST_POOL" \
  --input "$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard00_passk8_mixed_verified_nonempty.json" \
  --input "$RUNTIME/data/rl/bird_sft2_step1682_version26_train_next50_nonempty_shard01_passk8_mixed_verified_nonempty.json" \
  --output "$COMBINED_POOL"

combined_count="$("$PYTHON" - "$COMBINED_POOL" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["count"])
PY
)"
set_status awaiting_process_gates \
  "combined_nonempty_mixed_tasks=$combined_count result_only_rl=disabled process_rl_not_started"
