#!/usr/bin/env bash
# Exp8 -> Exp9 on shared NewGNN GPUs 5/6, with stability checks before every phase.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
EVAL_RUNTIME=${EVAL_RUNTIME:-"$OUTPUT_ROOT/eval_runtime_version36_20260728"}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2_ADAPTER_PATH=${SFT2_ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
TRAIN_GPU_ID=${TRAIN_GPU_ID:-6}
VLLM_GPU_ID=${VLLM_GPU_ID:-7}
EVAL_GPU_ID=${EVAL_GPU_ID:-6}
STABLE_FREE_SECONDS=${STABLE_FREE_SECONDS:-300}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
READY_MARKER=${READY_MARKER:-"$OUTPUT_ROOT/logs/newgnn_rank_runtime_ready_20260731"}
SMOKE_MARKER=${SMOKE_MARKER:-"$OUTPUT_ROOT/logs/newgnn_sft2_runtime_smoke_20260731.passed"}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-"$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/counterfactual_suite_v2.passed.json"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_newgnn_20260731.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_newgnn_20260731.log"}
QUEUE_LOCK=${QUEUE_LOCK:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_newgnn_20260731.lock"}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$QUEUE_STATUS"; }
state_of() {
  if [[ -f "$1" ]]; then awk -F '\t' 'NR==1 {print $2}' "$1"; else printf 'missing\n'; fi
}
gpu_memory() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$(( $1 + 1 ))p" | tr -d '[:space:]'
}
wait_for_stable_free() {
  local label=$1
  shift
  local free_since=0 now used gpu all_free
  while true; do
    all_free=1
    for gpu in "$@"; do
      used=$(gpu_memory "$gpu")
      if [[ -z "$used" || "$used" -gt "$GPU_FREE_THRESHOLD_MIB" ]]; then
        all_free=0
      fi
    done
    now=$(date +%s)
    if [[ "$all_free" -eq 1 ]]; then
      if [[ "$free_since" -eq 0 ]]; then free_since=$now; fi
      if [[ $((now - free_since)) -ge "$STABLE_FREE_SECONDS" ]]; then return 0; fi
      set_status waiting_shared_gpu \
        "phase=$label stable_free=$((now - free_since))/$STABLE_FREE_SECONDS gpu_ids=$*"
    else
      free_since=0
      set_status waiting_shared_gpu "phase=$label occupied gpu_ids=$*"
    fi
    sleep 30
  done
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
  set_status failed "duplicate NewGNN rank follow-up queue detected"
  exit 1
fi
exec >>"$QUEUE_LOG" 2>&1

test -f "$READY_MARKER"
test -f "$SMOKE_MARKER"
test -x "$PYTHON_ENV/bin/python"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$SFT2_ADAPTER_PATH/adapter_model.safetensors"
test -f "$SFT2_ADAPTER_PATH/adapter_config.json"
test -f "$COUNTERFACTUAL_SUITE_MANIFEST"
if [[ "$TRAIN_GPU_ID" == "$VLLM_GPU_ID" ]]; then
  set_status failed "trainer and vLLM GPUs must be distinct"
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

  wait_for_stable_free "Exp${ordinal}_training" "$TRAIN_GPU_ID" "$VLLM_GPU_ID"
  set_status training "ordinal=$ordinal experiment=$experiment_name host=NewGNN"
  cd "$TRAIN_RUNTIME"
  EXPERIMENT_NAME="$experiment_name" \
  EXPERIMENT_CONFIG="$TRAIN_RUNTIME/src/rl/configs/experiments/$config_name" \
  COUNTERFACTUAL_SUITE_MANIFEST="$COUNTERFACTUAL_SUITE_MANIFEST" \
  MODEL_PATH="$MODEL_PATH" \
  ADAPTER_PATH="$SFT2_ADAPTER_PATH" \
  OUTPUT_DIR="$checkpoint" \
  SMOKE_OUTPUT_DIR="$OUTPUT_ROOT/checkpoints/${artifact_name}-smoke" \
  STATUS="$train_status" \
  RUN_LOG="$OUTPUT_ROOT/logs/${artifact_name}.train.log" \
  VLLM_LOG="$OUTPUT_ROOT/logs/${artifact_name}.train.vllm.log" \
  TRAIN_GPU_ID="$TRAIN_GPU_ID" \
  VLLM_GPU_ID="$VLLM_GPU_ID" \
  VLLM_PORT=8032 \
  VLLM_GROUP_PORT=51232 \
    bash src/rl/experiments/run_gated_process_trl_table_rl.sh
  if [[ "$(state_of "$train_status")" != complete ]]; then
    set_status failed "training incomplete experiment=$experiment_name"
    exit 1
  fi
  "$PYTHON_ENV/bin/python" - \
    "$checkpoint/run_manifest.json" "$experiment_name" "$expected_rank_scope" \
    "$SFT2_ADAPTER_PATH" <<'PY'
import json
import sys

manifest_path, experiment_name, expected_scope, expected_adapter = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["experiment_config"]["experiment_name"] == experiment_name
assert manifest["adapter_path"] == expected_adapter
assert manifest["trainable_part"] == "all"
assert manifest["rank_loss_coefficient"] == 0.5
assert manifest["rank_beta"] == 0.1
assert manifest["rank_score_tokens"] == "tool_only"
assert manifest["rank_update_scope"] == expected_scope
assert manifest["process_reward_config_sha256"]
PY

  wait_for_stable_free "Exp${ordinal}_evaluation" "$EVAL_GPU_ID"
  set_status evaluating "ordinal=$ordinal experiment=$experiment_name host=NewGNN"
  cd "$EVAL_RUNTIME"
  EXPERIMENT_NAME="$experiment_name" \
  ADAPTER="$checkpoint/final" \
  RESULT_DIR="$result_dir" \
  SERVED_MODEL="$served_model" \
  STATUS="$eval_status" \
  RUN_LOG="$OUTPUT_ROOT/logs/${artifact_name}.eval.run.log" \
  EVAL_GPU_ID="$EVAL_GPU_ID" \
  EVAL_PYTHON="$PYTHON_ENV/bin/python" \
  BASE_MODEL="$MODEL_PATH" \
  PORT=18059 \
    bash "$TRAIN_RUNTIME/src/rl/experiments/run_gated_process_equal300_passk4_table_rl.sh"
  if [[ "$(state_of "$eval_status")" != complete ]]; then
    set_status failed "evaluation incomplete experiment=$experiment_name"
    exit 1
  fi
}

run_experiment \
  8 phase4_process_rank_action_only phase4_process_rank_action_only.yaml \
  trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-newgnn-20260731 \
  full_trajectory
run_experiment \
  9 phase5_process_rank_conservative phase5_process_rank_conservative.yaml \
  trl-transition-v26-phase5-process-rank-conservative-lr1e6-23x4-gated-v2-newgnn-20260731 \
  conservative_legal

set_status complete "Exp8-Exp9 NewGNN training and matched K4 evaluation complete"
