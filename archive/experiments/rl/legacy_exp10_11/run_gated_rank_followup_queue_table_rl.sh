#!/usr/bin/env bash
# Sequential Exp8 -> Exp9 rank follow-ups after the frozen Exp3 -> Exp7 queue.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
EVAL_RUNTIME=${EVAL_RUNTIME:-"$OUTPUT_ROOT/eval_runtime_version36_20260728"}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-"$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/counterfactual_suite_v2.passed.json"}
PREDECESSOR_QUEUE_STATUS=${PREDECESSOR_QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.status"}
PREDECESSOR_EVAL_STATUS=${PREDECESSOR_EVAL_STATUS:-"$OUTPUT_ROOT/logs/trl-transition-v26-phase3-process-strong-penalty-lr1e6-23x4-gated-v2-20260731.eval.status"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_20260731.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_20260731.log"}
QUEUE_LOCK=${QUEUE_LOCK:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_20260731.lock"}
HANDOFF_MARKER=${HANDOFF_MARKER:-"$OUTPUT_ROOT/logs/rank_followup_handed_off_to_newgnn_20260731"}

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
exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then
  set_status failed "duplicate gated rank follow-up queue detected"
  exit 1
fi
exec >>"$QUEUE_LOG" 2>&1

test -f "$COUNTERFACTUAL_SUITE_MANIFEST"
if [[ -f "$HANDOFF_MARKER" ]]; then
  set_status handed_off "Exp8-Exp9 assigned to NewGNN"
  exit 0
fi
if [[ "$(state_of "$PREDECESSOR_QUEUE_STATUS")" != complete ]] || \
   [[ "$(state_of "$PREDECESSOR_EVAL_STATUS")" != complete ]]; then
  set_status failed "Exp7 predecessor is not complete"
  exit 1
fi

run_experiment() {
  local ordinal=$1
  local experiment_name=$2
  local config_name=$3
  local artifact_name=$4
  local expected_rank_scope=$5
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
  if [[ "$(state_of "$train_status")" != complete ]]; then
    set_status failed "training incomplete experiment=$experiment_name"
    exit 1
  fi
  /home/dengyan/miniconda3/envs/trl-table/bin/python - \
    "$checkpoint/run_manifest.json" "$experiment_name" "$expected_rank_scope" <<'PY'
import json
import sys

manifest_path, experiment_name, expected_scope = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["experiment_config"]["experiment_name"] == experiment_name
assert manifest["experiment_config"]["process_reward_config"].endswith(
    "simple_process_reward_no_backslice.json"
)
assert manifest["trainable_part"] == "all"
assert manifest["rank_loss_coefficient"] == 0.5
assert manifest["rank_beta"] == 0.1
assert manifest["rank_score_tokens"] == "tool_only"
assert manifest["rank_update_scope"] == expected_scope
PY

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
  if [[ "$(state_of "$eval_status")" != complete ]]; then
    set_status failed "evaluation incomplete experiment=$experiment_name"
    exit 1
  fi
}

run_experiment \
  8 phase4_process_rank_action_only phase4_process_rank_action_only.yaml \
  trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-20260731 \
  full_trajectory
run_experiment \
  9 phase5_process_rank_conservative phase5_process_rank_conservative.yaml \
  trl-transition-v26-phase5-process-rank-conservative-lr1e6-23x4-gated-v2-20260731 \
  conservative_legal

set_status complete "Exp8-Exp9 training and matched K4 evaluation complete"
