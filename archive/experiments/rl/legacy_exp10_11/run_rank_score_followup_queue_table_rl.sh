#!/usr/bin/env bash
# Sequential Exp10 -> Exp11 score-masked rank follow-ups on table_rl.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-"$OUTPUT_ROOT/rl_runtime_rank_score_v3_20260731"}
EVAL_RUNTIME=${EVAL_RUNTIME:-"$OUTPUT_ROOT/eval_runtime_version36_20260728"}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
EXPECTED_ADAPTER_SHA256=${EXPECTED_ADAPTER_SHA256:-d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-"$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/counterfactual_suite_v2.passed.json"}
EXPECTED_COUNTERFACTUAL_SHA256=${EXPECTED_COUNTERFACTUAL_SHA256:-5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/rank_score_followup_queue_table_rl_20260731.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/rank_score_followup_queue_table_rl_20260731.log"}
QUEUE_LOCK=${QUEUE_LOCK:-"$OUTPUT_ROOT/logs/rank_score_followup_queue_table_rl_20260731.lock"}

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
  set_status failed "duplicate rank-score follow-up queue detected"
  exit 1
fi
exec >>"$QUEUE_LOG" 2>&1

test -f "$COUNTERFACTUAL_SUITE_MANIFEST"
test -f "$ADAPTER_PATH/adapter_model.safetensors"
test -d "$MODEL_PATH"
actual_adapter_sha=$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | awk '{print $1}')
actual_counterfactual_sha=$(sha256sum "$COUNTERFACTUAL_SUITE_MANIFEST" | awk '{print $1}')
if [[ "$actual_adapter_sha" != "$EXPECTED_ADAPTER_SHA256" ]]; then
  set_status failed "SFT2 adapter hash mismatch expected=$EXPECTED_ADAPTER_SHA256 actual=$actual_adapter_sha"
  exit 1
fi
if [[ "$actual_counterfactual_sha" != "$EXPECTED_COUNTERFACTUAL_SHA256" ]]; then
  set_status failed "counterfactual manifest hash mismatch expected=$EXPECTED_COUNTERFACTUAL_SHA256 actual=$actual_counterfactual_sha"
  exit 1
fi

run_experiment() {
  local ordinal=$1
  local experiment_name=$2
  local config_name=$3
  local artifact_name=$4
  local expected_reduction=$5
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
  VLLM_PORT=8033 \
  VLLM_GROUP_PORT=51233 \
  RUNTIME="$TRAIN_RUNTIME" \
    bash src/rl/experiments/run_gated_process_trl_table_rl.sh
  if [[ "$(state_of "$train_status")" != complete ]]; then
    set_status failed "training incomplete experiment=$experiment_name"
    exit 1
  fi
  /home/dengyan/miniconda3/envs/trl-table/bin/python - \
    "$checkpoint/run_manifest.json" "$experiment_name" "$expected_reduction" \
    "$MODEL_PATH" "$ADAPTER_PATH" <<'PY'
import json
import sys

manifest_path, experiment_name, expected_reduction, model_path, adapter_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["experiment_config"]["experiment_name"] == experiment_name
assert manifest["experiment_config"]["process_reward_config"].endswith(
    "simple_process_reward_no_backslice.json"
)
assert manifest["model_path"] == model_path
assert manifest["adapter_path"] == adapter_path
assert manifest["trainable_part"] == "all"
assert manifest["rank_loss_coefficient"] == 0.5
assert manifest["rank_beta"] == 0.1
assert manifest["rank_score_tokens"] == "tool_only"
assert manifest["rank_score_scope"] == "conservative_legal"
assert manifest["rank_score_reduction"] == expected_reduction
assert manifest["rank_update_scope"] == "conservative_legal"
PY

  set_status evaluating "ordinal=$ordinal experiment=$experiment_name"
  cd "$EVAL_RUNTIME"
  EXPERIMENT_NAME="$experiment_name" \
  ADAPTER="$checkpoint/final" \
  RESULT_DIR="$result_dir" \
  SERVED_MODEL="$served_model" \
  STATUS="$eval_status" \
  RUN_LOG="$OUTPUT_ROOT/logs/${artifact_name}.eval.run.log" \
  PORT=18063 \
    bash "$TRAIN_RUNTIME/src/rl/experiments/run_gated_process_equal300_passk4_table_rl.sh"
  if [[ "$(state_of "$eval_status")" != complete ]]; then
    set_status failed "evaluation incomplete experiment=$experiment_name"
    exit 1
  fi
}

run_experiment \
  10 phase6_process_rank_conservative_score_masked \
  phase6_process_rank_conservative_score_masked.yaml \
  trl-transition-v26-exp10-rank-conservative-score-masked-lr1e6-23x4-gated-v3-20260731 \
  sum_tokens
run_experiment \
  11 phase7_process_rank_conservative_action_mean \
  phase7_process_rank_conservative_action_mean.yaml \
  trl-transition-v26-exp11-rank-conservative-action-mean-lr1e6-23x4-gated-v3-20260731 \
  mean_action

set_status complete "Exp10-Exp11 training and matched K4 evaluation complete"
