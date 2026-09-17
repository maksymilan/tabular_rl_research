#!/usr/bin/env bash
# Unattended queue on table_rl:
#   1. arm A: 60-question SAAM+span recipe with 6 updates (was 4)  -> train + matched eval
#   2. arm B: same as A plus --adaptive-group-size-max 16 --carrier-repair -> train + matched eval
# Both evals use the strict carrier and no adaptive K, so they stay matched with the other five
# arms (the delivered code defaults the new mechanisms OFF).
# Every stage writes its return code into the queue log; a failed stage does not abort the queue.
set -Eeuo pipefail

BASE=/home/dengyan/tabular_rl_outputs
PROJ=${PROJ:-$BASE/qwen3_4b_saam60_updates6_20260917_r1/project}
A_RUN=$BASE/qwen3_4b_saam60_updates6_20260917_r1
B_RUN=$BASE/qwen3_4b_saam60_adaptivek_compat_20260917_r1
EVAL_A=$BASE/evaluations/qwen3_4b_saam60_updates6_checkpoint6_actionable_20260917_table_rl
EVAL_B=$BASE/evaluations/qwen3_4b_saam60_adaptivek_compat_checkpoint6_actionable_20260917_table_rl
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
QUEUE_ROOT=$BASE/saam60_updates6_queue_20260917
LOG=$QUEUE_ROOT/queue.log
mkdir -p "$QUEUE_ROOT"
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }
set_status() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >"$QUEUE_ROOT/status"; }

wait_idle() {
  local deadline=$((SECONDS + 1800)) u0 u1
  while (( SECONDS < deadline )); do
    u0=$(nvidia-smi --id="$TRAIN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    u1=$(nvidia-smi --id="$VLLM_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( u0 <= 512 && u1 <= 512 )); then return 0; fi
    sleep 30
  done
  log "ERROR: GPUs not idle (${u0}/${u1} MiB)"
  return 1
}

train() {  # $1=launcher $2=run root $3=label
  log "train $3 start -> $2"
  wait_idle || return 75
  local rc=0
  PROJECT_DIR="$PROJ" RUN_ROOT="$2" TRAIN_GPU="$TRAIN_GPU" VLLM_GPU="$VLLM_GPU" \
    bash "$1" || rc=$?
  log "train $3 finished rc=$rc"
  return "$rc"
}

eval_arm() {  # $1=adapter $2=run dir $3=label
  log "eval $3 start -> $2"
  wait_idle || return 75
  local rc=0
  ADAPTER="$1" RUN_DIR="$2" CHECKPOINT_GLOBAL_STEP=6 \
    GPU0="$TRAIN_GPU" GPU1="$VLLM_GPU" \
    bash "$BASE/run_single_matched_eval_table_rl.sh" || rc=$?
  log "eval $3 finished rc=$rc"
  return "$rc"
}

log "queue start; project=$PROJ"

set_status stage1_train_armA "60q/6upd, strict carrier, fixed K=8"
train "$PROJ/src/rl/scenarios/diagnostics/run_qwen3_4b_saam60_updates6_table_rl.sh" "$A_RUN" armA || true
set_status stage2_eval_armA "matched BIRD-dev1534"
eval_arm "$A_RUN/train/checkpoint-6" "$EVAL_A" armA || true

set_status stage3_train_armB "60q/6upd + adaptive-K(16) + carrier repair"
train "$PROJ/src/rl/scenarios/diagnostics/run_qwen3_4b_saam60_adaptivek_compat_table_rl.sh" "$B_RUN" armB || true
set_status stage4_eval_armB "matched BIRD-dev1534 (strict carrier, fixed K=8)"
eval_arm "$B_RUN/train/checkpoint-6" "$EVAL_B" armB || true

set_status queue_done "see $LOG"
log "queue done"
