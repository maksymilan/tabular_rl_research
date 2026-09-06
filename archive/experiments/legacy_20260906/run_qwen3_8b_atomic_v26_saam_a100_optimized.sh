#!/usr/bin/env bash
# A100 online-RL launcher for the frozen version26 SAAM span-balanced arm.
# This changes only hardware scheduling knobs; protocol, reward, adapter and
# task identities remain bound to the 3090 Gate60 experiment.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
# The server reports P2P=OK, but a direct NCCL all-reduce on GPU4<->GPU5
# hangs in the first collective.  Shared-memory transport is healthy
# (6.08 GB/s on a 64 MiB probe), so prefer SHM until the host topology is
# repaired rather than spinning both cards in a broken P2P kernel.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# vLLM V1's engine-core allocator is not compatible with the expandable
# segments setting on this host (the same 16k eager server works without it).
# Leave it unset for both processes in this runtime; the default allocator is
# stable for the verified A100 model-load path.
unset PYTORCH_CUDA_ALLOC_CONF

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/data2/shared/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/data4/dengyan/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_gate60.yaml}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_a100_optimized_20260901}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train60_two_pass_seed20260829}
TRAIN_GPU=${TRAIN_GPU:-4}
VLLM_GPU=${VLLM_GPU:-5}
VLLM_PORT=${VLLM_PORT:-8214}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51314}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-60}
OPTIMIZER_STEPS=${OPTIMIZER_STEPS:-2}
GROUP_SIZE=${GROUP_SIZE:-8}
# SAAM has a second gradient/logit pass whose peak is much higher than the
# first pass.  Long trajectories still need a small row count, while short
# transitions can safely share a larger batch.  Dynamic packing is bounded by
# both the row cap and padded prompt+completion token budget.  Setting the
# token budget to zero restores fixed-row behavior.
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-6}
TRANSITION_MICRO_BATCH_TOKENS=${TRANSITION_MICRO_BATCH_TOKENS:-16384}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-60}
LIMIT=${LIMIT:-0}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/SFT1 adapter\n' >&2; exit 3;
}
[[ -f "$TASKS" && -f "$CONFIG" && -d "$PROTOCOL_RUNTIME" ]] || {
  printf 'missing tasks/config/protocol runtime\n' >&2; exit 3;
}
[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct\n' >&2; exit 2;
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing existing run root: %s\n' "$RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}
set_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >"$RUN_ROOT/status.next"
  mv "$RUN_ROOT/status.next" "$RUN_ROOT/status"
}
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "${trainer_pgid:-}"
  stop_group "${vllm_pgid:-}"
  [[ "$code" -eq 0 ]] || set_status failed "exit=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && {
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2; exit 75;
} || true

vllm_pgid=''; trainer_pgid=''
set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT util=$VLLM_GPU_MEMORY_UTILIZATION max_model_len=$MAX_MODEL_LEN cuda_graph=$([[ "$VLLM_ENFORCE_EAGER" == 0 ]] && echo on || echo off)"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; exit 1; }

set_status training "records=$EXPECTED_RECORDS limit=$LIMIT k=$GROUP_SIZE prompts=$PROMPTS_PER_UPDATE updates=$OPTIMIZER_STEPS credit=saam-asymmetric-error span_balance_alpha=0.5 transition_micro_batch_max_rows=$TRANSITION_MICRO_BATCH_SIZE transition_micro_batch_tokens=$TRANSITION_MICRO_BATCH_TOKENS"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONPATH= \
  "$PYTHON" "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" \
  --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" --examples-json "$TASKS" \
  --output-dir "$TRAIN_OUT" --experiment-config "$CONFIG" --limit "$LIMIT" \
  --vllm-host 127.0.0.1 --vllm-port "$VLLM_PORT" --vllm-group-port "$VLLM_GROUP_PORT" \
  --optimizer-steps "$OPTIMIZER_STEPS" --ppo-iterations 1 \
  --prompts-per-update "$PROMPTS_PER_UPDATE" --group-size "$GROUP_SIZE" \
  --credit-assignment saam-asymmetric-error --error-penalty 1.0 --kl-beta 0 \
  --span-balance-alpha 0.5 --no-record-gradient-conflicts \
  --protocol-runtime-root "$PROTOCOL_RUNTIME" \
  --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" \
  --transition-micro-batch-tokens "$TRANSITION_MICRO_BATCH_TOKENS" \
  --expected-records "$EXPECTED_RECORDS" --save-steps 1 --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --seed 20260829 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=''
stop_group "$vllm_pgid"; vllm_pgid=''

[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || {
  printf 'training exited without immutable manifests\n' >&2; exit 1;
}
set_status complete "final=$TRAIN_OUT/final"
trap - EXIT INT TERM
