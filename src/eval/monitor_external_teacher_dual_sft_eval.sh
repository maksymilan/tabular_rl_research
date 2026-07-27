#!/usr/bin/env bash
set -euo pipefail

# Monitor the paired external-teacher SFT jobs. Once both training processes
# finish successfully, validate the final adapters, wait for both GPUs to be
# released, and launch matched bird-set pass@K evaluations in parallel.

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

REMOTE="${REMOTE:-table_rl}"
PROJECT_DIR="${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
REMOTE_TRAIN_PYTHON="${REMOTE_TRAIN_PYTHON:-/home/dengyan/miniconda3/envs/sft/bin/python}"
RUN_ENV="${RUN_ENV:-${REMOTE_OUTPUT_ROOT}/logs/bird_external_teacher_fixed1000_dual_7b_train_20260724/run.env}"
GENERAL_OUTPUT_DIR="${GENERAL_OUTPUT_DIR:-${REMOTE_OUTPUT_ROOT}/checkpoints/qwen2.5-7b-bird-external-teacher-fixed1000-full-prefix-6400-qlora}"
CODER_OUTPUT_DIR="${CODER_OUTPUT_DIR:-${REMOTE_OUTPUT_ROOT}/checkpoints/qwen2.5-coder-7b-bird-external-teacher-fixed1000-full-prefix-6400-qlora}"
GENERAL_BASE_MODEL="${GENERAL_BASE_MODEL:-/home/dengyan/models/Qwen2.5-7B-Instruct}"
CODER_BASE_MODEL="${CODER_BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}"
GENERAL_SERVED_MODEL="${GENERAL_SERVED_MODEL:-qwen25_7b_external_teacher_fixed1000_epoch2_tool_passk4_dev}"
CODER_SERVED_MODEL="${CODER_SERVED_MODEL:-qwen25_coder_7b_external_teacher_fixed1000_epoch2_tool_passk4_dev}"
EXPECTED_MAX_STEPS="${EXPECTED_MAX_STEPS:-516}"
GENERAL_GPU="${GENERAL_GPU:-0}"
CODER_GPU="${CODER_GPU:-1}"
GENERAL_REMOTE_PORT="${GENERAL_REMOTE_PORT:-8030}"
CODER_REMOTE_PORT="${CODER_REMOTE_PORT:-8031}"
GENERAL_LOCAL_PORT="${GENERAL_LOCAL_PORT:-18030}"
CODER_LOCAL_PORT="${CODER_LOCAL_PORT:-18031}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-1024}"
MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-3}"
RESUME_EXISTING_EVAL_SERVICES="${RESUME_EXISTING_EVAL_SERVICES:-0}"
RUNNER="${RUNNER:-${PROJECT_DIR}/src/eval/run_bird_lora_tool_passk_table_rl.sh}"
GENERAL_EVAL_LOG="${GENERAL_EVAL_LOG:-/tmp/bird_external_teacher_general_epoch2_eval.log}"
CODER_EVAL_LOG="${CODER_EVAL_LOG:-/tmp/bird_external_teacher_coder_epoch2_eval.log}"
GENERAL_RESULT_DIR="${GENERAL_RESULT_DIR:-data/results/qwen2.5_7b_external_teacher_fixed1000_epoch2_passk4_dev1534_bird_set}"
CODER_RESULT_DIR="${CODER_RESULT_DIR:-data/results/qwen2.5_coder_7b_external_teacher_fixed1000_epoch2_passk4_dev1534_bird_set}"

for name in EXPECTED_MAX_STEPS GENERAL_GPU CODER_GPU GENERAL_REMOTE_PORT \
  CODER_REMOTE_PORT GENERAL_LOCAL_PORT CODER_LOCAL_PORT POLL_SECONDS \
  GPU_IDLE_MEMORY_MIB MAX_EVAL_ATTEMPTS; do
  value="${!name}"
  is_uint "$value" || die "${name} must be a non-negative integer, got: ${value}"
done
test "$POLL_SECONDS" -gt 0 || die "POLL_SECONDS must be greater than zero"
test "$MAX_EVAL_ATTEMPTS" -gt 0 || die "MAX_EVAL_ATTEMPTS must be greater than zero"
case "$RESUME_EXISTING_EVAL_SERVICES" in
  0 | 1) ;;
  *) die "RESUME_EXISTING_EVAL_SERVICES must be 0 or 1" ;;
esac
test "$GENERAL_GPU" -ne "$CODER_GPU" || die "the paired evaluation requires distinct GPUs"
test "$GENERAL_REMOTE_PORT" -ne "$CODER_REMOTE_PORT" ||
  die "the paired evaluation requires distinct remote ports"
test "$GENERAL_LOCAL_PORT" -ne "$CODER_LOCAL_PORT" ||
  die "the paired evaluation requires distinct local ports"
test -x "$RUNNER" || die "evaluation runner is not executable: ${RUNNER}"

cd "$PROJECT_DIR"

training_pids="$(
  ssh "$REMOTE" "
    set -eu
    test -r '$RUN_ENV'
    . '$RUN_ENV'
    printf '%s %s\n' \"\$general_pid\" \"\$coder_pid\"
  "
)"
set -- $training_pids
test "$#" -eq 2 || die "could not read exactly two training pids from ${RUN_ENV}: ${training_pids}"
general_pid="$1"
coder_pid="$2"
is_uint "$general_pid" || die "invalid general training pid: ${general_pid}"
is_uint "$coder_pid" || die "invalid coder training pid: ${coder_pid}"

printf 'Monitoring training pids on %s: general=%s coder=%s\n' \
  "$REMOTE" "$general_pid" "$coder_pid"

last_status=""
ssh_failures=0
while :; do
  if status="$(
    ssh "$REMOTE" "
      general_live=0
      coder_live=0
      kill -0 '$general_pid' 2>/dev/null && general_live=1
      kill -0 '$coder_pid' 2>/dev/null && coder_live=1
      printf '%s %s\n' \"\$general_live\" \"\$coder_live\"
    "
  )"; then
    ssh_failures=0
  else
    ssh_failures=$((ssh_failures + 1))
    printf 'Transient SSH status-check failure (%s); retrying\n' "$ssh_failures" >&2
    sleep "$POLL_SECONDS"
    continue
  fi

  if test "$status" != "$last_status"; then
    printf '%s training_live general=%s coder=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S %Z')" $status
    last_status="$status"
  fi

  set -- $status
  test "$#" -eq 2 || die "unexpected remote process status: ${status}"
  if test "$1" -eq 0 && test "$2" -eq 0; then
    break
  fi
  sleep "$POLL_SECONDS"
done

general_adapter="${GENERAL_OUTPUT_DIR}/checkpoint-${EXPECTED_MAX_STEPS}"
coder_adapter="${CODER_OUTPUT_DIR}/checkpoint-${EXPECTED_MAX_STEPS}"
trainer_state_check='import json,sys; d=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert d.get("global_step")==expected, d.get("global_step"); assert d.get("max_steps")==expected, d.get("max_steps")'

printf 'Training processes exited; validating final adapters at step %s\n' "$EXPECTED_MAX_STEPS"
for adapter in "$general_adapter" "$coder_adapter"; do
  ssh "$REMOTE" "
    set -eu
    test -f '$adapter/adapter_config.json'
    test -f '$adapter/adapter_model.safetensors'
    test -f '$adapter/trainer_state.json'
    '$REMOTE_TRAIN_PYTHON' -c '$trainer_state_check' \
      '$adapter/trainer_state.json' '$EXPECTED_MAX_STEPS'
  " || die "final checkpoint validation failed: ${adapter}"
done

if test "$RESUME_EXISTING_EVAL_SERVICES" -eq 1; then
  printf 'Resume mode: runner will validate and adopt exact matching evaluation services\n'
else
  printf 'Final adapters are complete; waiting for GPUs %s and %s to be idle\n' \
    "$GENERAL_GPU" "$CODER_GPU"
  last_memory=""
  while :; do
    memory="$(
      ssh "$REMOTE" "
        set -eu
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
          awk -v general='$((GENERAL_GPU + 1))' -v coder='$((CODER_GPU + 1))' '
            {gsub(/ /, \"\")}
            NR == general {general_memory=\$0}
            NR == coder {coder_memory=\$0}
            END {printf \"%s %s\n\", general_memory, coder_memory}
          '
      "
    )"
    set -- $memory
    test "$#" -eq 2 || die "could not read memory usage for both GPUs: ${memory}"
    general_memory="$1"
    coder_memory="$2"
    is_uint "$general_memory" || die "invalid GPU memory value: ${general_memory}"
    is_uint "$coder_memory" || die "invalid GPU memory value: ${coder_memory}"

    if test "$memory" != "$last_memory"; then
      printf '%s gpu_memory_mib general=%s coder=%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$general_memory" "$coder_memory"
      last_memory="$memory"
    fi
    if test "$general_memory" -le "$GPU_IDLE_MEMORY_MIB" &&
      test "$coder_memory" -le "$GPU_IDLE_MEMORY_MIB"; then
      break
    fi
    sleep "$POLL_SECONDS"
  done
fi

run_eval_with_resume() {
  local label="$1"
  local base_model="$2"
  local adapter="$3"
  local served_model="$4"
  local gpu_id="$5"
  local remote_port="$6"
  local local_port="$7"
  local result_dir="$8"
  local log_file="$9"
  local attempt=1
  local status=0

  while test "$attempt" -le "$MAX_EVAL_ATTEMPTS"; do
    printf '%s evaluation attempt %s/%s\n' "$label" "$attempt" "$MAX_EVAL_ATTEMPTS"
    set +e
    REMOTE="$REMOTE" \
      PROJECT_DIR="$PROJECT_DIR" \
      REMOTE_OUTPUT_ROOT="$REMOTE_OUTPUT_ROOT" \
      BASE_MODEL="$base_model" \
      ADAPTER="$adapter" \
      SERVED_MODEL="$served_model" \
      GPU_ID="$gpu_id" \
      REMOTE_PORT="$remote_port" \
      LOCAL_PORT="$local_port" \
      RESULT_DIR="$result_dir" \
      LOCAL_EVAL_LOG="$log_file" \
      EXISTING_SERVICE_POLICY=adopt \
      "$RUNNER"
    status=$?
    set -e

    if test "$status" -eq 0; then
      return 0
    fi
    if test "$status" -ne 75; then
      printf '%s evaluation failed non-transiently with status %s\n' "$label" "$status" >&2
      return "$status"
    fi
    attempt=$((attempt + 1))
    if test "$attempt" -le "$MAX_EVAL_ATTEMPTS"; then
      printf '%s evaluation had a transient service failure; resuming after %ss\n' \
        "$label" "$POLL_SECONDS" >&2
      sleep "$POLL_SECONDS"
    fi
  done
  return "$status"
}

printf 'Both GPUs are idle; launching matched bird-set pass@K evaluations\n'

run_eval_with_resume \
  general \
  "$GENERAL_BASE_MODEL" \
  "$general_adapter" \
  "$GENERAL_SERVED_MODEL" \
  "$GENERAL_GPU" "$GENERAL_REMOTE_PORT" "$GENERAL_LOCAL_PORT" \
  "$GENERAL_RESULT_DIR" "$GENERAL_EVAL_LOG" &
general_eval_pid=$!

run_eval_with_resume \
  coder \
  "$CODER_BASE_MODEL" \
  "$coder_adapter" \
  "$CODER_SERVED_MODEL" \
  "$CODER_GPU" "$CODER_REMOTE_PORT" "$CODER_LOCAL_PORT" \
  "$CODER_RESULT_DIR" "$CODER_EVAL_LOG" &
coder_eval_pid=$!

set +e
wait "$general_eval_pid"
general_status=$?
wait "$coder_eval_pid"
coder_status=$?
set -e

printf 'Evaluation statuses: general=%s coder=%s\n' "$general_status" "$coder_status"
test "$general_status" -eq 0 || exit "$general_status"
test "$coder_status" -eq 0 || exit "$coder_status"
printf 'Both bird-set pass@K evaluations completed successfully\n'
