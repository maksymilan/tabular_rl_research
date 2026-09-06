#!/usr/bin/env bash
# Gate60 matched SAAM arm: asymmetric SAAM credit with the repaired float32
# span-balanced carrier, matched to the Vanilla span-balanced control.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_optimized_20260831}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_gate60.yaml}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_spanbalanced_gate60_fixed_20260901}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train60_two_pass_seed20260829}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-6}
VLLM_PORT=${VLLM_PORT:-8114}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51314}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-1}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/SFT1 adapter\n' >&2
  exit 3
}
[[ -f "$TASKS" && -f "$CONFIG" ]] || {
  printf 'missing Gate60 tasks/config\n' >&2
  exit 3
}
[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct\n' >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || {
  printf 'refusing existing run root: %s\n' "$RUN_ROOT" >&2
  exit 3
}
mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || {
  printf 'launcher already running\n' >&2
  exit 75
}
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}

status_file="$RUN_ROOT/status"
set_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >"$status_file.next"
  mv "$status_file.next" "$status_file"
}

gpu_idle "$TRAIN_GPU" || {
  printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2
  exit 75
}
gpu_idle "$VLLM_GPU" || {
  printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2
  exit 75
}
if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
fi

vllm_pgid=''
trainer_pgid=''
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
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION=0.82 \
  MAX_MODEL_LEN=16384 bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
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
  printf 'vLLM readiness failed\n' >&2
  exit 1
}

set_status training "Gate60 records=60 k=8 prompts=30 updates=4 passes=2 trajectories=960 credit=saam-asymmetric-error span_balance_alpha=0.5 gradient_diagnostics=off transition_micro_batch=$TRANSITION_MICRO_BATCH_SIZE"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --optimizer-steps 4 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
    --credit-assignment saam-asymmetric-error --error-penalty 1.0 --kl-beta 0 \
    --span-balance-alpha 0.5 --no-record-gradient-conflicts \
    --protocol-runtime-root "$PROTOCOL_RUNTIME" \
    --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" \
    --save-steps 1 --save-total-limit 4 --seed 20260829 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=''
stop_group "$vllm_pgid"
vllm_pgid=''

[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || {
  printf 'training exited without immutable manifests\n' >&2
  exit 1
}
[[ -d "$TRAIN_OUT/checkpoint-4" && -d "$TRAIN_OUT/final" ]] || {
  printf 'training exited without final checkpoint-4 artifacts\n' >&2
  exit 1
}
set_status complete "final=$TRAIN_OUT/final checkpoint=4"
trap - EXIT INT TERM
