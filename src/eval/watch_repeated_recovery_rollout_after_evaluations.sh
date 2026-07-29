#!/usr/bin/env bash
set -euo pipefail

# Wait for the two declared evaluation artifacts, then use the released GPUs to run two disjoint
# greedy BIRD-train shards with the SFT2 adapter.  This server-side handoff deliberately stops
# after rollout completion: repeat-failure selection and the external-teacher diagnostic use local
# api.md credentials and are launched by the Codex monitor after the artifacts are synchronized.

PROJECT_DIR="${PROJECT_DIR:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}"
ADAPTER="${ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}"
RUNNER="${RUNNER:-$PROJECT_DIR/src/eval/run_bird_lora_tool_passk_local_gpu.sh}"
EXPECTED_EVAL_TASKS="${EXPECTED_EVAL_TASKS:-1534}"
EXPECTED_SHARD_TASKS="${EXPECTED_SHARD_TASKS:-1500}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CONCURRENCY="${CONCURRENCY:-24}"
MAX_RUNNER_RESUME_ATTEMPTS="${MAX_RUNNER_RESUME_ATTEMPTS:-3}"

GPU0_EVAL_SESSION="${GPU0_EVAL_SESSION:-bird_sft2_v36_full_greedy}"
GPU1_EVAL_SESSION="${GPU1_EVAL_SESSION:-bird_cp560_overnight_eval}"
GPU0_EVAL_RESULT="${GPU0_EVAL_RESULT:-$PROJECT_DIR/data/results/qwen25_coder7b_sft2_step1682_tool_version36_dev1534_greedy1_bird_set}"
GPU1_EVAL_RESULT="${GPU1_EVAL_RESULT:-$OUTPUT_ROOT/eval_runtime_20260726/data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk8_bird_set}"

TASK_ROOT="${TASK_ROOT:-$PROJECT_DIR/data/eval_inputs/bird_train_batch2_student_rollout_disjoint3000_shards}"
TASK_SHARD0="${TASK_SHARD0:-$TASK_ROOT/bird_train_batch2_student_rollout_disjoint3000.shard-00-of-02.jsonl}"
TASK_SHARD1="${TASK_SHARD1:-$TASK_ROOT/bird_train_batch2_student_rollout_disjoint3000.shard-01-of-02.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/data/results/repeated_recovery}"
RESULT_SHARD0="${RESULT_SHARD0:-$RESULT_ROOT/qwen25_coder7b_sft2_step1682_train_repeat_rollout_shard00_staged_dbsync_bird_set}"
RESULT_SHARD1="${RESULT_SHARD1:-$RESULT_ROOT/qwen25_coder7b_sft2_step1682_train_repeat_rollout_shard01_staged_dbsync_bird_set}"

STATUS_FILE="${STATUS_FILE:-$OUTPUT_ROOT/logs/bird_sft2_repeated_recovery_rollout.status}"
WATCH_LOG="${WATCH_LOG:-$OUTPUT_ROOT/logs/bird_sft2_repeated_recovery_rollout.watch.log}"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_ROOT/logs/bird_sft2_repeated_recovery_rollout.lock}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  local stage="$1"
  local detail="$2"
  printf '%s\t%s\t%s\n' "$(timestamp)" "$stage" "$detail" >"$STATUS_FILE"
}

row_count() {
  local result_dir="$1"
  if test -f "$result_dir/all.jsonl"; then
    wc -l <"$result_dir/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

port_in_use() {
  local port="$1"
  ss -ltnH "sport = :$port" | grep -q .
}

validate_result() {
  local result_dir="$1"
  local expected="$2"
  local expected_samples="$3"
  "$PYTHON_BIN" - "$result_dir/all.jsonl" "$expected" "$expected_samples" <<'PY'
import json
import sys

path, expected_text, expected_samples_text = sys.argv[1:]
expected = int(expected_text)
expected_samples = int(expected_samples_text)
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
assert len(rows) == expected, (path, len(rows), expected)
indices = [int(row["example_index"]) for row in rows]
assert len(set(indices)) == expected, (path, "duplicate example_index")
infra = []
for row in rows:
    samples = row.get("samples") or [row]
    assert len(samples) == expected_samples, (
        row.get("example_index"),
        len(samples),
        expected_samples,
    )
    for sample in samples:
        if sample.get("failure_type") in {
            "api_error",
            "provider_carrier_error",
            "transport_error",
        }:
            infra.append((row.get("example_index"), sample.get("failure_type")))
assert not infra, (path, "infrastructure failures", infra[:20])
PY
}

missing_task_db_count() {
  "$PYTHON_BIN" - "$TASK_SHARD0" "$TASK_SHARD1" <<'PY'
import json
import os
import sys

missing = set()
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            db_path = task.get("db_path")
            if not db_path or not os.path.isfile(db_path):
                missing.add((task.get("db_id"), db_path))
print(len(missing))
PY
}

for path in "$PYTHON_BIN" "$RUNNER" "$TASK_SHARD0" "$TASK_SHARD1"; do
  test -e "$path" || die "required path is missing: $path"
done
test -x "$PYTHON_BIN" || die "Python is not executable: $PYTHON_BIN"
test -x "$RUNNER" || die "rollout runner is not executable: $RUNNER"
test -f "$ADAPTER/adapter_config.json" || die "adapter config is missing: $ADAPTER"
test -f "$ADAPTER/adapter_model.safetensors" || die "adapter weights are missing: $ADAPTER"
[[ "$MAX_RUNNER_RESUME_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] ||
  die "MAX_RUNNER_RESUME_ATTEMPTS must be a positive integer"

mkdir -p "$(dirname "$STATUS_FILE")" "$RESULT_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another repeated-recovery rollout watcher holds $LOCK_FILE"
exec >>"$WATCH_LOG" 2>&1
cd "$PROJECT_DIR"

printf '%s watcher started; waiting for both evaluation chains\n' "$(timestamp)"
while :; do
  eval0_rows="$(row_count "$GPU0_EVAL_RESULT")"
  eval1_rows="$(row_count "$GPU1_EVAL_RESULT")"
  eval0_live=0
  eval1_live=0
  tmux has-session -t "$GPU0_EVAL_SESSION" 2>/dev/null && eval0_live=1
  tmux has-session -t "$GPU1_EVAL_SESSION" 2>/dev/null && eval1_live=1
  set_status waiting_evaluations \
    "gpu0_rows=$eval0_rows/$EXPECTED_EVAL_TASKS gpu0_tmux=$eval0_live gpu1_rows=$eval1_rows/$EXPECTED_EVAL_TASKS gpu1_tmux=$eval1_live"
  if test "$eval0_rows" -eq "$EXPECTED_EVAL_TASKS" &&
     test "$eval1_rows" -eq "$EXPECTED_EVAL_TASKS" &&
     test "$eval0_live" -eq 0 &&
     test "$eval1_live" -eq 0; then
    break
  fi
  sleep "$POLL_SECONDS"
done

validate_result "$GPU0_EVAL_RESULT" "$EXPECTED_EVAL_TASKS" 1
validate_result "$GPU1_EVAL_RESULT" "$EXPECTED_EVAL_TASKS" 8

while port_in_use 18048 || port_in_use 18037; do
  set_status waiting_ports "evaluation service cleanup is still in progress"
  sleep "$POLL_SECONDS"
done

run_shard() {
  local shard="$1"
  local gpu="$2"
  local port="$3"
  local result_dir="$4"
  local served_model="$5"

  if test "$(row_count "$result_dir")" -eq "$EXPECTED_SHARD_TASKS"; then
    validate_result "$result_dir" "$EXPECTED_SHARD_TASKS" 1
    printf '%s shard already complete: %s\n' "$(timestamp)" "$result_dir"
    return 0
  fi
  PROJECT_DIR="$PROJECT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_MODEL="$BASE_MODEL" \
  ADAPTER="$ADAPTER" \
  SERVED_MODEL="$served_model" \
  GPU_ID="$gpu" \
  PORT="$port" \
  RESULT_DIR="$result_dir" \
  EXAMPLES_JSON="$shard" \
  N="$EXPECTED_SHARD_TASKS" \
  N_SAMPLES=1 \
  PASS_K=1 \
  WORKERS="$CONCURRENCY" \
  SAMPLE_WORKERS=1 \
  MAX_INFLIGHT="$CONCURRENCY" \
  MAX_NUM_SEQS="$CONCURRENCY" \
  MAX_NUM_BATCHED_TOKENS=8192 \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 \
  MAX_TOKENS=1024 \
  TEMPERATURE=0 \
  TOP_P=1 \
  HISTORY_TURNS=4 \
  EVAL_ENABLE_THINKING=0 \
  ALLOW_MISSING_TASK_DATABASES=1 \
  SERVER_CONFIG_ID="vllm-generation-config-vllm-${served_model}" \
  TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  VLLM_LOG="$OUTPUT_ROOT/logs/${served_model}.vllm.log" \
  VLLM_PID_FILE="$OUTPUT_ROOT/logs/${served_model}.vllm.pid" \
  EVAL_LOG="$OUTPUT_ROOT/logs/${served_model}.eval.log" \
    bash "$RUNNER"
}

run_both_shards() {
  local phase="$1"
  set_status student_rollout \
    "phase=$phase concurrency=$CONCURRENCY result_root=$RESULT_ROOT"
  printf '%s launching two SFT2 train rollouts phase=%s\n' "$(timestamp)" "$phase"
  set +e
  run_shard \
    "$TASK_SHARD0" 0 18049 "$RESULT_SHARD0" \
    qwen25-coder7b-sft2-step1682-train-repeat-shard00 &
  pid0=$!
  run_shard \
    "$TASK_SHARD1" 1 18050 "$RESULT_SHARD1" \
    qwen25-coder7b-sft2-step1682-train-repeat-shard01 &
  pid1=$!
  wait "$pid0"
  status0=$?
  wait "$pid1"
  status1=$?
  set -e
  if test "$status0" -ne 0 || test "$status1" -ne 0; then
    set_status failed \
      "phase=$phase shard0_status=$status0 shard1_status=$status1; partial results preserved"
    printf 'ERROR: one or both train rollout shards failed during %s\n' "$phase" >&2
    return 1
  fi
}

run_phase_with_resume() {
  local phase="$1"
  local attempt=1
  while ! run_both_shards "$phase"; do
    if test "$attempt" -ge "$MAX_RUNNER_RESUME_ATTEMPTS"; then
      die "rollout phase $phase still failed after $attempt attempts"
    fi
    attempt=$((attempt + 1))
    set_status retrying_student_rollout \
      "phase=$phase attempt=$attempt/$MAX_RUNNER_RESUME_ATTEMPTS; completed rows preserved"
    printf '%s retrying rollout phase=%s attempt=%s/%s from immutable row checkpoints\n' \
      "$(timestamp)" "$phase" "$attempt" "$MAX_RUNNER_RESUME_ATTEMPTS"
    sleep "$POLL_SECONDS"
  done
}

run_phase_with_resume available_databases

while :; do
  missing_databases="$(missing_task_db_count)"
  if test "$missing_databases" -eq 0; then
    break
  fi
  set_status waiting_train_databases \
    "missing_unique_db_paths=$missing_databases; completed available tasks are preserved"
  sleep "$POLL_SECONDS"
done

run_phase_with_resume complete_after_sync

validate_result "$RESULT_SHARD0" "$EXPECTED_SHARD_TASKS" 1
validate_result "$RESULT_SHARD1" "$EXPECTED_SHARD_TASKS" 1
"$PYTHON_BIN" - "$RESULT_SHARD0/all.jsonl" "$RESULT_SHARD1/all.jsonl" <<'PY'
import json
import sys

sets = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        sets.append({int(json.loads(line)["example_index"]) for line in handle if line.strip()})
assert not (sets[0] & sets[1]), sorted(sets[0] & sets[1])[:20]
assert len(sets[0] | sets[1]) == 3000, len(sets[0] | sets[1])
PY

set_status rollouts_complete \
  "shard0=$EXPECTED_SHARD_TASKS shard1=$EXPECTED_SHARD_TASKS waiting_for_local_repeat_selection_and_teacher_pilot"
printf '%s repeated-recovery student rollouts complete; local teacher handoff is ready\n' \
  "$(timestamp)"
