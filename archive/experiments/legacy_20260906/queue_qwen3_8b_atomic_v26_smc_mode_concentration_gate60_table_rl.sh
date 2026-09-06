#!/usr/bin/env bash
# Wait for the pre-existing table_rl baseline queue, then run the isolated SMC
# mode-concentration gate on both 3090s.  Never cancels or reuses other jobs.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_smc_mode_concentration_20260906}
LAUNCHER="$RUNTIME/src/rl/experiments/run_qwen3_8b_atomic_v26_smc_mode_concentration_gate60_table_rl.sh"
LOG_ROOT=${LOG_ROOT:-$OUTPUT_ROOT/qwen3_v26_smc_mode_concentration_gate60_table_rl_20260906}
LOG_FILE=${LOG_FILE:-$LOG_ROOT/queue.log}
POLL_SECONDS=${POLL_SECONDS:-60}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8112}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51312}
NEWGNN_HOST=${NEWGNN_HOST:-NewGNN}
NEWGNN_EVAL_SCRIPT=${NEWGNN_EVAL_SCRIPT:-/home/dengyan/tabular_rl_outputs/qwen3_v26_smc_mode_concentration_gate60_table_rl_20260906/run_smc_eval_newgnn.sh}

mkdir -p "$LOG_ROOT"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"; }
die() { log "blocked: $*"; exit 3; }

[[ -x "$LAUNCHER" ]] || die "missing SMC launcher: $LAUNCHER"
gpu_idle() {
  local gpu=$1 used pids
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 1024 ]]
}

while ! gpu_idle "$TRAIN_GPU" || ! gpu_idle "$VLLM_GPU"; do
  log "waiting for GPU${TRAIN_GPU}/GPU${VLLM_GPU} to be genuinely idle"
  sleep "$POLL_SECONDS"
done

if ss -ltnH 2>/dev/null | awk -v p=":$VLLM_PORT" '$4 ~ p"$" {found=1} END {exit found ? 0 : 1}'; then
  die "vLLM port is occupied: $VLLM_PORT"
fi

log "starting isolated SMC gate: trainer_gpu=$TRAIN_GPU vllm_gpu=$VLLM_GPU"
set +e
env \
  TRAIN_GPU="$TRAIN_GPU" \
  VLLM_GPU="$VLLM_GPU" \
  VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
  LOG_ROOT="$LOG_ROOT/logs" \
  OUTPUT_DIR="$LOG_ROOT/train" \
  bash "$LAUNCHER"
training_status=$?
set -e
if [[ "$training_status" -ne 0 ]]; then
  log "SMC training failed with status=$training_status; not scheduling evaluation"
  exit "$training_status"
fi

CHECKPOINT="$LOG_ROOT/train/checkpoint-4"
[[ -f "$CHECKPOINT/trainer_state.json" && -f "$CHECKPOINT/adapter_model.safetensors" ]] || {
  die "completed training checkpoint is incomplete: $CHECKPOINT"
}
log "training complete; scheduling NewGNN matched evaluation for $CHECKPOINT"
ssh "$NEWGNN_HOST" "$NEWGNN_EVAL_SCRIPT" "$CHECKPOINT"
