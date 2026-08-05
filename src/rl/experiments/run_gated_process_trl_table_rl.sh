#!/usr/bin/env bash
# One gated 23-question × 4-rollout process-RL experiment using the TRL backend.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_process_gated_v2_20260730}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
EXAMPLES_JSON=${EXAMPLES_JSON:-"$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:?set EXPERIMENT_CONFIG}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-"$OUTPUT_ROOT/process_gate_v2_20260730/counterfactual_suite23_fullcopy_candidate_v2/counterfactual_suite_v2.passed.json"}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:?set SMOKE_OUTPUT_DIR}
STATUS=${STATUS:?set STATUS}
RUN_LOG=${RUN_LOG:?set RUN_LOG}
VLLM_LOG=${VLLM_LOG:?set VLLM_LOG}
VLLM_PORT=${VLLM_PORT:-8031}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51231}
TRAIN_GPU_ID=${TRAIN_GPU_ID:-0}
VLLM_GPU_ID=${VLLM_GPU_ID:-1}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}
HARDWARE_PROFILE=${HARDWARE_PROFILE:-$RUNTIME/src/rl/configs/hardware/rtx3090_24gb.sh}
test -f "$HARDWARE_PROFILE"
# shellcheck source=../configs/hardware/rtx3090_24gb.sh
source "$HARDWARE_PROFILE"
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-$RTX3090_FREE_MEMORY_THRESHOLD_MIB}

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

vllm_pid=""
vllm_pgid=""
cleanup() {
  local exit_status=$?
  local current_pgid=""
  trap - EXIT INT TERM
  current_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')
  if [[ -n "$vllm_pgid" ]] \
    && [[ "$vllm_pgid" != "$current_pgid" ]] \
    && pgrep -g "$vllm_pgid" >/dev/null 2>&1; then
    kill -TERM -- "-$vllm_pgid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! pgrep -g "$vllm_pgid" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
    if pgrep -g "$vllm_pgid" >/dev/null 2>&1; then
      kill -KILL -- "-$vllm_pgid" 2>/dev/null || true
    fi
  elif [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill -TERM "$vllm_pid" 2>/dev/null || true
  fi
  if [[ -n "$vllm_pid" ]]; then
    wait "$vllm_pid" 2>/dev/null || true
  fi
  if [[ "$exit_status" -ne 0 ]]; then
    set_status failed "exit=$exit_status see=$RUN_LOG"
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"

test -f "$EXAMPLES_JSON"
test -f "$EXPERIMENT_CONFIG"
test -f "$COUNTERFACTUAL_SUITE_MANIFEST"
test -f "$ADAPTER_PATH/adapter_model.safetensors"
[[ "$TRAIN_GPU_ID" =~ ^[0-9]+$ ]] || { set_status failed "invalid TRAIN_GPU_ID"; exit 1; }
[[ "$VLLM_GPU_ID" =~ ^[0-9]+$ ]] || { set_status failed "invalid VLLM_GPU_ID"; exit 1; }
if [[ "$TRAIN_GPU_ID" == "$VLLM_GPU_ID" ]]; then
  set_status failed "trainer and vLLM GPUs must be distinct"
  exit 1
fi

if [[ -f "$OUTPUT_DIR/final/adapter_model.safetensors" ]]; then
  set_status complete "already complete output=$OUTPUT_DIR"
  exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  set_status failed "nonempty incomplete output requires manual resume audit: $OUTPUT_DIR"
  exit 1
fi

duplicate_pattern="run_transition_grpo.py.*${OUTPUT_DIR}"
if pgrep -f "$duplicate_pattern" >/dev/null 2>&1; then
  set_status failed "duplicate trainer already exists for output=$OUTPUT_DIR"
  exit 1
fi

gpu_ready=0
for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if [[ "$TRAIN_GPU_ID" -lt "${#gpu_memory[@]}" ]] \
    && [[ "$VLLM_GPU_ID" -lt "${#gpu_memory[@]}" ]] \
    && [[ "${gpu_memory[$TRAIN_GPU_ID]}" -le "$GPU_FREE_THRESHOLD_MIB" ]] \
    && [[ "${gpu_memory[$VLLM_GPU_ID]}" -le "$GPU_FREE_THRESHOLD_MIB" ]]; then
    gpu_ready=1
    break
  fi
  set_status waiting_gpu \
    "trainer_gpu=$TRAIN_GPU_ID mib=${gpu_memory[$TRAIN_GPU_ID]:-unknown} vllm_gpu=$VLLM_GPU_ID mib=${gpu_memory[$VLLM_GPU_ID]:-unknown}"
  sleep 30
done
if [[ "$gpu_ready" -ne 1 ]]; then
  set_status failed "timed out waiting for two free GPUs"
  exit 1
fi

set_status starting "experiment=$EXPERIMENT_NAME vllm_gpu=$VLLM_GPU_ID trainer_gpu=$TRAIN_GPU_ID"
CUDA_VISIBLE_DEVICES="$VLLM_GPU_ID" \
PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_GPU_MEMORY_UTILIZATION="$RTX3090_ONLINE_RL_VLLM_GPU_MEMORY_UTILIZATION" \
MAX_MODEL_LEN=8192 \
  setsid bash src/rl/frameworks/trl/start_vllm_server.sh >"$VLLM_LOG" 2>&1 &
vllm_pid=$!
vllm_pgid=$vllm_pid

ready=0
for _ in {1..90}; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    set_status failed "vllm exited before readiness"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  set_status failed "vllm readiness timeout"
  exit 1
fi

if [[ ! -f "$SMOKE_OUTPUT_DIR/final/adapter_model.safetensors" ]]; then
  if [[ -d "$SMOKE_OUTPUT_DIR" ]] && [[ -n "$(find "$SMOKE_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    set_status failed "nonempty incomplete smoke output: $SMOKE_OUTPUT_DIR"
    exit 1
  fi
  set_status smoke "experiment=$EXPERIMENT_NAME one_task_group2_one_update"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU_ID" \
  PROJECT_DIR="$RUNTIME" \
  PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python \
  MODEL_PATH="$MODEL_PATH" \
  ADAPTER_PATH="$ADAPTER_PATH" \
  EXAMPLES_JSON="$EXAMPLES_JSON" \
  OUTPUT_DIR="$SMOKE_OUTPUT_DIR" \
  EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
  COUNTERFACTUAL_SUITE_MANIFEST="$COUNTERFACTUAL_SUITE_MANIFEST" \
  VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
      --limit 1 \
      --optimizer-steps 1 \
      --group-size 2 \
      --max-agent-steps 2 \
      --max-new-tokens 256 \
      --transition-micro-batch-size "$RTX3090_ONLINE_RL_TRANSITION_MICRO_BATCH_SIZE" \
      --save-steps 1
  test -f "$SMOKE_OUTPUT_DIR/final/adapter_model.safetensors"
fi

set_status running "experiment=$EXPERIMENT_NAME questions=23 group=4 updates=23"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_ID" \
PROJECT_DIR="$RUNTIME" \
PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python \
MODEL_PATH="$MODEL_PATH" \
ADAPTER_PATH="$ADAPTER_PATH" \
EXAMPLES_JSON="$EXAMPLES_JSON" \
OUTPUT_DIR="$OUTPUT_DIR" \
EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
COUNTERFACTUAL_SUITE_MANIFEST="$COUNTERFACTUAL_SUITE_MANIFEST" \
VLLM_PORT="$VLLM_PORT" \
VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
  bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
    --transition-micro-batch-size "$RTX3090_ONLINE_RL_TRANSITION_MICRO_BATCH_SIZE" \
    --save-steps 5

test -f "$OUTPUT_DIR/final/adapter_model.safetensors"
/home/dengyan/miniconda3/envs/trl-table/bin/python - \
  "$OUTPUT_DIR/run_manifest.json" "$EXPERIMENT_NAME" "$COUNTERFACTUAL_SUITE_MANIFEST" <<'PY'
import hashlib
import json
import sys

manifest_path, experiment_name, counterfactual_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
digest = hashlib.sha256(open(counterfactual_path, "rb").read()).hexdigest()
assert manifest["experiment_config"]["experiment_name"] == experiment_name
assert manifest["experiment_config"]["admission_status"] == "allowed_process_after_gates"
assert manifest["reward_mode"] == "process"
assert manifest["process_admission_policy"] == "counterfactual-completeness"
assert manifest["counterfactual_manifest_schema"] == "process-counterfactual-suite-v2"
assert manifest["counterfactual_manifest_sha256"] == digest
assert manifest["records"] == 23
assert manifest["group_size"] == 4
assert manifest["optimizer_steps"] == 23
assert manifest["ppo_iterations"] == 1
assert manifest["optimizer_name"] == "adamw_torch"
assert manifest["learning_rate"] == 1e-6
assert manifest["lr_scheduler_type"] == "cosine"
assert manifest["warmup_ratio"] == 0.03
assert manifest["kl_beta"] == 0.0
assert manifest["temperature"] == 0.7
assert manifest["top_p"] == 0.95
PY
set_status complete "experiment=$EXPERIMENT_NAME output=$OUTPUT_DIR"
