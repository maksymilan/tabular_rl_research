#!/usr/bin/env bash
# Sequential Exp3 -> Exp7 training and matched K=4 evaluation.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
EVAL_RUNTIME=${EVAL_RUNTIME:-"$OUTPUT_ROOT/eval_runtime_version36_20260728"}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-"$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/counterfactual_suite_v2.passed.json"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.log"}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$QUEUE_STATUS"; }
state_of() {
  if [[ -f "$1" ]]; then
    awk -F '\t' 'NR==1 {print $2}' "$1"
  else
    printf 'missing\n'
  fi
}
on_exit() {
  local exit_status=$?
  trap - EXIT
  if [[ "$exit_status" -ne 0 ]]; then
    set_status failed "exit=$exit_status see=$QUEUE_LOG"
  fi
  exit "$exit_status"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$QUEUE_LOG" 2>&1

test -f "$COUNTERFACTUAL_SUITE_MANIFEST"
if pgrep -f "run_gated_process_ablation_queue_table_rl.sh" | grep -v "^$$$" >/dev/null 2>&1; then
  set_status failed "duplicate gated process queue detected"
  exit 1
fi

run_experiment() {
  local ordinal=$1
  local experiment_name=$2
  local config_name=$3
  local artifact_name=$4
  local train_status="$OUTPUT_ROOT/logs/${artifact_name}.train.status"
  local eval_status="$OUTPUT_ROOT/logs/${artifact_name}.eval.status"
  local checkpoint="$OUTPUT_ROOT/checkpoints/${artifact_name}"
  local result_dir="$EVAL_RUNTIME/data/results/${artifact_name}_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set"
  local served_model="${artifact_name}-version36-equal300-passk4-logprobs20"

  set_status training "ordinal=$ordinal experiment=$experiment_name"
  cd "$TRAIN_RUNTIME"
  EXPERIMENT_NAME="$experiment_name" \
  EXPERIMENT_CONFIG="$TRAIN_RUNTIME/src/rl/configs/experiments/$config_name" \
  COUNTERFACTUAL_SUITE_MANIFEST="$COUNTERFACTUAL_SUITE_MANIFEST" \
  OUTPUT_DIR="$checkpoint" \
  SMOKE_OUTPUT_DIR="$OUTPUT_ROOT/checkpoints/${artifact_name}-smoke" \
  STATUS="$train_status" \
  RUN_LOG="$OUTPUT_ROOT/logs/${artifact_name}.train.log" \
  VLLM_LOG="$OUTPUT_ROOT/logs/${artifact_name}.train.vllm.log" \
  VLLM_PORT=8031 \
  VLLM_GROUP_PORT=51231 \
    bash src/rl/experiments/run_gated_process_trl_table_rl.sh
  if [[ "$(state_of "$train_status")" != "complete" ]]; then
    set_status failed "training incomplete experiment=$experiment_name"
    exit 1
  fi

  set_status evaluating "ordinal=$ordinal experiment=$experiment_name"
  cd "$EVAL_RUNTIME"
  EXPERIMENT_NAME="$experiment_name" \
  ADAPTER="$checkpoint/final" \
  RESULT_DIR="$result_dir" \
  SERVED_MODEL="$served_model" \
  STATUS="$eval_status" \
  RUN_LOG="$OUTPUT_ROOT/logs/${artifact_name}.eval.run.log" \
  PORT=18058 \
    bash "$TRAIN_RUNTIME/src/rl/experiments/run_gated_process_equal300_passk4_table_rl.sh"
  if [[ "$(state_of "$eval_status")" != "complete" ]]; then
    set_status failed "evaluation incomplete experiment=$experiment_name"
    exit 1
  fi
}

run_experiment \
  3 phase1_process_no_backslice phase1_process_no_backslice.yaml \
  trl-transition-v26-phase1-process-no-backslice-lr1e6-23x4-gated-v2-20260730
run_experiment \
  4 phase1_process_tool_only phase1_process_tool_only.yaml \
  trl-transition-v26-phase1-process-tool-only-lr1e6-23x4-gated-v2-20260730
run_experiment \
  5 phase2_process_rank phase2_process_rank.yaml \
  trl-transition-v26-phase2-process-rank-lr1e6-23x4-gated-v2-20260730
run_experiment \
  6 phase2_process_no_normalize phase2_process_no_normalize.yaml \
  trl-transition-v26-phase2-process-no-normalize-lr1e6-23x4-gated-v2-20260730
run_experiment \
  7 phase3_process_strong_penalty phase3_process_strong_penalty.yaml \
  trl-transition-v26-phase3-process-strong-penalty-lr1e6-23x4-gated-v2-20260731

set_status complete "Exp3-Exp7 training and matched K4 evaluation complete"
