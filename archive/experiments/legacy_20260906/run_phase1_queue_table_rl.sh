#!/usr/bin/env bash
# Wait for the two matched read-only evaluations, then train and evaluate Exp 1.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-"$OUTPUT_ROOT/rl_runtime_trl_transition_v26_20260729"}
EVAL_RUNTIME=${EVAL_RUNTIME:-"$OUTPUT_ROOT/eval_runtime_version36_20260728"}
SFT_STATUS=${SFT_STATUS:-"$OUTPUT_ROOT/logs/phase0_sft2_equal300_passk4_20260729.status"}
PROCESS_STATUS=${PROCESS_STATUS:-"$OUTPUT_ROOT/logs/phase1_process_current_equal300_passk4_20260729.status"}
TRAIN_STATUS=${TRAIN_STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.status"}
EVAL_STATUS=${EVAL_STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_equal300_passk4_20260729.status"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/phase1_experiment_queue_20260729.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/phase1_experiment_queue_20260729.log"}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$QUEUE_STATUS"; }
on_exit() {
  local exit_status=$?
  trap - EXIT
  if [[ "$exit_status" -ne 0 ]]; then
    set_status failed "exit=$exit_status see=$QUEUE_LOG"
  fi
  exit "$exit_status"
}
trap on_exit EXIT

state_of() {
  if [[ -f "$1" ]]; then
    awk -F '\t' 'NR==1 {print $2}' "$1"
  else
    printf 'missing\n'
  fi
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$QUEUE_LOG" 2>&1

set_status waiting "phase0_sft_and_existing_process_k4"
for _ in {1..720}; do
  sft_state=$(state_of "$SFT_STATUS")
  process_state=$(state_of "$PROCESS_STATUS")
  if [[ "$sft_state" == "failed" || "$process_state" == "failed" ]]; then
    set_status failed "upstream sft=$sft_state process=$process_state"
    exit 1
  fi
  if [[ "$sft_state" == "complete" && "$process_state" == "complete" ]]; then
    break
  fi
  sleep 30
done
if [[ "$(state_of "$SFT_STATUS")" != "complete" || "$(state_of "$PROCESS_STATUS")" != "complete" ]]; then
  set_status failed "upstream timeout"
  exit 1
fi

set_status training "phase1_result_only"
cd "$TRAIN_RUNTIME"
bash src/rl/experiments/run_phase1_result_only_trl_table_rl.sh
if [[ "$(state_of "$TRAIN_STATUS")" != "complete" ]]; then
  set_status failed "training did not complete"
  exit 1
fi

set_status evaluating "phase1_result_only_equal300_k4"
cd "$EVAL_RUNTIME"
bash src/rl/experiments/run_phase1_result_only_equal300_passk4_table_rl.sh
if [[ "$(state_of "$EVAL_STATUS")" != "complete" ]]; then
  set_status failed "result-only evaluation did not complete"
  exit 1
fi
set_status complete "phase0/process/result-only matched K4 artifacts ready"
