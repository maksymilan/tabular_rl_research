#!/usr/bin/env bash
# Small matched SAAM experiment with four-level terminal rewards.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_saam_fourlevel_gate60_20260901}
PROJECT_DIR=${PROJECT_DIR:-$RUNTIME}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_gate60.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_fourlevel_gate60_20260901/train_two_pass_seed20260901}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8092}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51292}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-1}
RESULT_ADVANTAGE_PROFILE=${RESULT_ADVANTAGE_PROFILE:-stored}
CREDIT_ASSIGNMENT=${CREDIT_ASSIGNMENT:-saam-asymmetric-error}
CLEAN_ADVANTAGE_WEIGHT=${CLEAN_ADVANTAGE_WEIGHT:-0.25}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-60}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-30}
OPTIMIZER_STEPS=${OPTIMIZER_STEPS:-4}
TASK_LIMIT=${TASK_LIMIT:-0}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  echo "missing Python/runtime/model/adapter" >&2
  exit 2
}
[[ -f "$TASKS" && -f "$CONFIG" && -d "$PROTOCOL_RUNTIME" ]] || {
  echo "missing tasks/config/protocol runtime" >&2
  exit 2
}
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  echo "trainer and vLLM GPUs must differ" >&2
  exit 2
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 2
}
mkdir -p "$(dirname "$OUTPUT_DIR")"
LOG_ROOT=${LOG_ROOT:-$(dirname "$OUTPUT_DIR")/logs}
mkdir -p "$LOG_ROOT"

vllm_pgid=""
trainer_pgid=""
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "$trainer_pgid"
  stop_group "$vllm_pgid"
  exit "$code"
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
  VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >"$LOG_ROOT/fourlevel_vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || {
  echo "vLLM did not become ready" >&2
  exit 1
}

setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$OUTPUT_DIR" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --experiment-config "$CONFIG" --credit-assignment "$CREDIT_ASSIGNMENT" \
    --result-reward-profile four-level \
    --result-advantage-profile "$RESULT_ADVANTAGE_PROFILE" \
    --clean-advantage-weight "$CLEAN_ADVANTAGE_WEIGHT" --kl-beta 0 \
    --expected-records "$EXPECTED_RECORDS" --optimizer-steps "$OPTIMIZER_STEPS" --ppo-iterations 1 \
    --prompts-per-update "$PROMPTS_PER_UPDATE" --group-size 8 --policy-reduction trajectory_token_mean \
    --learning-rate 4e-7 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --clip-epsilon 0.2 --clip-epsilon-high 0.2 --span-balance-alpha 0.5 \
    --optimizer-name adamw_torch \
    --limit "$TASK_LIMIT" \
    --no-record-gradient-conflicts --save-steps 1 --save-total-limit 4 \
    --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" \
    --seed 20260901 \
  >"$LOG_ROOT/fourlevel_train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=""

echo "four-level SAAM training complete: $OUTPUT_DIR"
