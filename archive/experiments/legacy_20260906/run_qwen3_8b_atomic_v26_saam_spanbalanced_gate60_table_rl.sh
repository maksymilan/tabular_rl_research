#!/usr/bin/env bash
# Diagnostic SAAM carrier ablation: balance reasoning/tool span means.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_saam_spanbalanced_gate60_20260830}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_gate60.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_spanbalanced_gate60_20260830/run1}
VLLM_PORT=${VLLM_PORT:-8093}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51293}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}

[[ -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  echo "missing runtime, model, or adapter" >&2
  exit 2
}
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  echo "trainer and vLLM GPUs must differ" >&2
  exit 2
}
[[ -f "$TASKS" && -f "$CONFIG" ]] || {
  echo "missing Gate60 tasks or asymmetric config" >&2
  exit 2
}
mkdir -p "$(dirname "$OUTPUT_DIR")"
[[ ! -e "$OUTPUT_DIR/run_manifest.json" ]] || {
  echo "refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 2
}
VLLM_LOG=${VLLM_LOG:-${OUTPUT_DIR}.vllm.log}
TRAIN_LOG=${TRAIN_LOG:-${OUTPUT_DIR}.train.log}

vllm_pid=""
cleanup() {
  code=$?
  if [[ -n "$vllm_pid" ]]; then
    kill -TERM -- "-$vllm_pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION=0.82 \
  MAX_MODEL_LEN=16384 bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >"$VLLM_LOG" 2>&1 &
vllm_pid=$!
ready=0
for _ in $(seq 1 300); do
  if curl -fsSL "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || {
  echo "vLLM did not become ready" >&2
  exit 1
}

setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$OUTPUT_DIR" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --optimizer-steps 4 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
    --credit-assignment saam-asymmetric-error --error-penalty 1.0 --kl-beta 0 \
    --span-balance-alpha 0.5 --protocol-runtime-root "$PROTOCOL_RUNTIME" \
    --transition-micro-batch-size 1 --save-steps 1 --save-total-limit 4 \
    --seed 20260829 \
  >"$TRAIN_LOG" 2>&1
status=$?
kill "$vllm_pid" 2>/dev/null || true
wait "$vllm_pid" 2>/dev/null || true
vllm_pid=""
exit "$status"
