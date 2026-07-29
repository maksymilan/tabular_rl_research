#!/usr/bin/env bash
set -euo pipefail

# Server-side watcher: after the single-GPU batch-2 SFT completes successfully, use the released
# GPU 0 to run one full BIRD-dev greedy tool evaluation. This script is intentionally self-hosted
# in tmux so a sleeping/disconnected client cannot pause the handoff.

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
TRAIN_PID="${TRAIN_PID:-2669312}"
TRAIN_CONFIG_BASENAME="${TRAIN_CONFIG_BASENAME:-bird_sft2_qwen25_coder7b_cp560_batch2_single_gpu_qlora_6400.yaml}"
EXPECTED_STEPS="${EXPECTED_STEPS:-1682}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora}"
TRAINER_LOG="${TRAINER_LOG:-$TRAIN_OUTPUT_DIR/trainer_log.jsonl}"
FINAL_ADAPTER="${FINAL_ADAPTER:-$TRAIN_OUTPUT_DIR/checkpoint-$EXPECTED_STEPS}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
GPU_ID="${GPU_ID:-0}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-3}"
EXPECTED_TASKS="${EXPECTED_TASKS:-1534}"
PORT="${PORT:-18039}"
CONCURRENCY="${CONCURRENCY:-24}"
SERVED_MODEL="${SERVED_MODEL:-qwen25-coder7b-sft2-batch2-step1682-tool-dev1534-greedy1}"
RESULT_DIR="${RESULT_DIR:-data/results/qwen25_coder7b_sft2_batch2_step1682_tool_dev1534_greedy1_bird_set}"
RUNNER="${RUNNER:-$PROJECT_DIR/src/eval/run_bird_lora_tool_passk_local_gpu.sh}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_ROOT/logs/bird_sft2_batch2_then_greedy.status}"
WATCH_LOG="${WATCH_LOG:-$OUTPUT_ROOT/logs/bird_sft2_batch2_then_greedy.log}"
VLLM_LOG="${VLLM_LOG:-$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.pid}"
EVAL_LOG="${EVAL_LOG:-$OUTPUT_ROOT/logs/$SERVED_MODEL.eval.log}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  local stage="$1"
  local detail="$2"
  printf '%s\t%s\t%s\n' "$(timestamp)" "$stage" "$detail" >"$STATUS_FILE"
}

completed_rows() {
  if test -f "$RESULT_DIR/all.jsonl"; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

training_pid_matches() {
  test -r "/proc/$TRAIN_PID/cmdline" || return 1
  tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" | grep -Fq "$TRAIN_CONFIG_BASENAME"
}

training_log_complete() {
  test -s "$TRAINER_LOG" || return 1
  "$PYTHON_BIN" - "$TRAINER_LOG" "$EXPECTED_STEPS" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
last = None
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            last = json.loads(line)
assert last is not None
assert int(last.get("current_steps", -1)) == expected, last.get("current_steps")
assert int(last.get("total_steps", -1)) == expected, last.get("total_steps")
PY
}

adapter_complete() {
  test -f "$FINAL_ADAPTER/adapter_config.json" || return 1
  test -f "$FINAL_ADAPTER/adapter_model.safetensors" || return 1
  test -f "$FINAL_ADAPTER/trainer_state.json" || return 1
  "$PYTHON_BIN" - "$FINAL_ADAPTER/trainer_state.json" "$EXPECTED_STEPS" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
with open(path, encoding="utf-8") as handle:
    state = json.load(handle)
assert int(state.get("global_step", -1)) == expected, state.get("global_step")
assert int(state.get("max_steps", -1)) == expected, state.get("max_steps")
PY
}

validate_result() {
  "$PYTHON_BIN" - "$RESULT_DIR/all.jsonl" "$EXPECTED_TASKS" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
records = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))
assert len(records) == expected, len(records)
indices = [record.get("example_index") for record in records]
assert len(set(indices)) == expected, len(set(indices))
infra = [
    record.get("example_index")
    for record in records
    if record.get("failure_type") in {
        "api_error",
        "provider_carrier_error",
        "context_overflow",
    }
]
assert not infra, infra[:20]
PY
}

for name in TRAIN_PID EXPECTED_STEPS GPU_ID GPU_IDLE_MEMORY_MIB POLL_SECONDS \
  MAX_EVAL_ATTEMPTS EXPECTED_TASKS PORT CONCURRENCY; do
  value="${!name}"
  is_uint "$value" || die "$name must be a non-negative integer, got $value"
done
test "$POLL_SECONDS" -gt 0 || die "POLL_SECONDS must be positive"
test "$MAX_EVAL_ATTEMPTS" -gt 0 || die "MAX_EVAL_ATTEMPTS must be positive"
test -x "$PYTHON_BIN" || die "evaluation Python is not executable: $PYTHON_BIN"
test -x "$RUNNER" || die "local-GPU evaluation runner is not executable: $RUNNER"
test -d "$BASE_MODEL" || die "base model is missing: $BASE_MODEL"

mkdir -p "$(dirname "$STATUS_FILE")" "$(dirname "$WATCH_LOG")"
exec >>"$WATCH_LOG" 2>&1
cd "$PROJECT_DIR"

printf '%s watcher started train_pid=%s expected_steps=%s gpu=%s\n' \
  "$(timestamp)" "$TRAIN_PID" "$EXPECTED_STEPS" "$GPU_ID"

if test "$(completed_rows)" -eq "$EXPECTED_TASKS"; then
  validate_result
  set_status completed "greedy already complete rows=$EXPECTED_TASKS"
  printf '%s greedy result already complete; nothing to launch\n' "$(timestamp)"
  exit 0
fi

set_status waiting_training "pid=$TRAIN_PID expected_step=$EXPECTED_STEPS"
while training_pid_matches; do
  if test -s "$TRAINER_LOG"; then
    progress="$(
      "$PYTHON_BIN" - "$TRAINER_LOG" <<'PY'
import json
import sys

last = None
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            last = json.loads(line)
if last:
    print(f"{last.get('current_steps')}/{last.get('total_steps')}")
else:
    print("unknown")
PY
    )"
    set_status waiting_training "progress=$progress pid=$TRAIN_PID"
  fi
  sleep "$POLL_SECONDS"
done

printf '%s training process exited; validating final artifacts\n' "$(timestamp)"
if ! training_log_complete; then
  set_status failed "training exited before trainer_log reached $EXPECTED_STEPS"
  die "training process exited without a complete trainer log"
fi

artifact_deadline=$((SECONDS + 600))
until adapter_complete; do
  if test "$SECONDS" -ge "$artifact_deadline"; then
    set_status failed "final adapter validation timed out: $FINAL_ADAPTER"
    die "final adapter did not become complete within 600 seconds"
  fi
  sleep 5
done
printf '%s final adapter validated: %s\n' "$(timestamp)" "$FINAL_ADAPTER"

set_status waiting_gpu "gpu=$GPU_ID memory_threshold_mib=$GPU_IDLE_MEMORY_MIB"
while :; do
  gpu_memory="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
      awk -v row="$((GPU_ID + 1))" 'NR == row {gsub(/ /, ""); print; exit}'
  )"
  is_uint "$gpu_memory" || die "could not read GPU $GPU_ID memory usage"
  if test "$gpu_memory" -le "$GPU_IDLE_MEMORY_MIB"; then
    break
  fi
  set_status waiting_gpu "gpu=$GPU_ID memory_mib=$gpu_memory"
  sleep "$POLL_SECONDS"
done

if ss -ltnH "sport = :$PORT" | grep -q .; then
  set_status failed "port $PORT already in use"
  die "evaluation port is already in use: $PORT"
fi

attempt=1
while test "$attempt" -le "$MAX_EVAL_ATTEMPTS"; do
  set_status greedy "attempt=$attempt concurrency=$CONCURRENCY rows=$(completed_rows)"
  printf '%s starting greedy attempt=%s adapter=%s\n' \
    "$(timestamp)" "$attempt" "$FINAL_ADAPTER"
  set +e
  PROJECT_DIR="$PROJECT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_MODEL="$BASE_MODEL" \
  ADAPTER="$FINAL_ADAPTER" \
  SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID="$GPU_ID" \
  PORT="$PORT" \
  RESULT_DIR="$RESULT_DIR" \
  N="$EXPECTED_TASKS" \
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
  SERVER_CONFIG_ID=vllm-generation-config-vllm-sft2-batch2-step1682-tool-dev1534-greedy1 \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 \
  TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
  VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$VLLM_PID_FILE" \
  EVAL_LOG="$EVAL_LOG" \
    bash "$RUNNER"
  status=$?
  set -e
  if test "$status" -eq 0; then
    validate_result
    set_status completed "rows=$EXPECTED_TASKS result=$RESULT_DIR"
    printf '%s greedy complete rows=%s result=%s\n' \
      "$(timestamp)" "$EXPECTED_TASKS" "$RESULT_DIR"
    exit 0
  fi
  printf '%s greedy attempt=%s failed status=%s; preserving partial output\n' \
    "$(timestamp)" "$attempt" "$status" >&2
  attempt=$((attempt + 1))
  if test "$attempt" -le "$MAX_EVAL_ATTEMPTS"; then
    sleep "$POLL_SECONDS"
  fi
done

set_status failed "greedy exhausted $MAX_EVAL_ATTEMPTS attempts rows=$(completed_rows)"
die "greedy evaluation exhausted all attempts"
