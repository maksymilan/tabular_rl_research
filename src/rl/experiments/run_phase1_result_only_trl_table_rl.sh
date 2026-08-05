#!/usr/bin/env bash
# One exact 23-question × 4-rollout result-only control using the TRL backend.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_trl_transition_v26_20260729}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
EXAMPLES_JSON=${EXAMPLES_JSON:-"$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-"$RUNTIME/src/rl/configs/experiments/phase1_result_only.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase1-result-only-lr1e6-23x4-20260729"}
SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:-"$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase1-result-only-queue-smoke-20260729"}
STATUS=${STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.status"}
RUN_LOG=${RUN_LOG:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.run.log"}
VLLM_LOG=${VLLM_LOG:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.vllm.log"}
VLLM_PORT=${VLLM_PORT:-8030}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51230}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

vllm_pid=""
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill "$vllm_pid" 2>/dev/null || true
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
test -f "$ADAPTER_PATH/adapter_model.safetensors"

if [[ -f "$OUTPUT_DIR/final/adapter_model.safetensors" ]]; then
  set_status complete "already complete output=$OUTPUT_DIR"
  exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  set_status failed "nonempty incomplete output requires manual resume audit: $OUTPUT_DIR"
  exit 1
fi

gpu_ready=0
for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if [[ "${#gpu_memory[@]}" -ge 2 ]] \
    && [[ "${gpu_memory[0]}" -le "$GPU_FREE_THRESHOLD_MIB" ]] \
    && [[ "${gpu_memory[1]}" -le "$GPU_FREE_THRESHOLD_MIB" ]]; then
    gpu_ready=1
    break
  fi
  set_status waiting_gpu \
    "gpu0_mib=${gpu_memory[0]:-unknown} gpu1_mib=${gpu_memory[1]:-unknown}"
  sleep 30
done
if [[ "$gpu_ready" -ne 1 ]]; then
  set_status failed "timed out waiting for two free GPUs"
  exit 1
fi

set_status starting "vllm gpu=1 port=$VLLM_PORT"
CUDA_VISIBLE_DEVICES=1 \
PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_GPU_MEMORY_UTILIZATION=0.82 \
MAX_MODEL_LEN=8192 \
  bash src/rl/frameworks/trl/start_vllm_server.sh >"$VLLM_LOG" 2>&1 &
vllm_pid=$!

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
  set_status smoke "gpu=0 group=2 max_steps=2 optimizer_steps=1"
  CUDA_VISIBLE_DEVICES=0 \
  PROJECT_DIR="$RUNTIME" \
  PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python \
  MODEL_PATH="$MODEL_PATH" \
  ADAPTER_PATH="$ADAPTER_PATH" \
  EXAMPLES_JSON="$EXAMPLES_JSON" \
  OUTPUT_DIR="$SMOKE_OUTPUT_DIR" \
  OPTIMIZER_STEPS=1 \
  PPO_ITERATIONS=1 \
  PROMPTS_PER_UPDATE=1 \
  GROUP_SIZE=2 \
  LEARNING_RATE=1e-6 \
  KL_BETA=0 \
  VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
      --limit 1 \
      --max-agent-steps 2 \
      --max-new-tokens 256 \
      --transition-micro-batch-size 1 \
      --save-steps 1
  test -f "$SMOKE_OUTPUT_DIR/final/adapter_model.safetensors"
fi

set_status running "result-only questions=23 group=4 updates=23 lr=1e-6"
CUDA_VISIBLE_DEVICES=0 \
PROJECT_DIR="$RUNTIME" \
PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python \
MODEL_PATH="$MODEL_PATH" \
ADAPTER_PATH="$ADAPTER_PATH" \
EXAMPLES_JSON="$EXAMPLES_JSON" \
OUTPUT_DIR="$OUTPUT_DIR" \
EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
VLLM_PORT="$VLLM_PORT" \
VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
  bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
    --transition-micro-batch-size 1 \
    --save-steps 5

test -f "$OUTPUT_DIR/final/adapter_model.safetensors"
/home/dengyan/miniconda3/envs/trl-table/bin/python - \
  "$OUTPUT_DIR/run_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["reward_mode"] == "result-only"
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
assert manifest["trainable_part"] == "all"
assert manifest["rank_loss_coefficient"] == 0.0
PY
set_status complete "output=$OUTPUT_DIR"
