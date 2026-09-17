#!/usr/bin/env bash
# Wait for the NewGNN 180-question second-epoch run to finish, then run its checkpoint-12
# matched BIRD-dev1534 eval with the same identity as every other arm
# (actionable-error-v1, enforce_eager=0, T=0/2048/30, 2-GPU even/odd shards, 24 workers).
set -Eeuo pipefail

BASE=/home/dengyan/tabular_rl_outputs
RUN=$BASE/qwen3_4b_saam180_epoch2_20260917_r1
PROJECT=$RUN/project
EVAL_RUN=$BASE/evaluations/qwen3_4b_saam180_epoch2_checkpoint12_actionable_20260917_newgnn
TRAIN_GPU=${TRAIN_GPU:-3}
VLLM_GPU=${VLLM_GPU:-7}
LOG=$BASE/saam180_epoch2_eval_after_train.log
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }

log "waiting for $RUN/train/checkpoint-12 and trainer exit"
deadline=$((SECONDS + 24 * 3600))
while (( SECONDS < deadline )); do
  if [[ -f "$RUN/train/checkpoint-12/trainer_state.json" ]]; then
    if ! pgrep -f "$RUN/project/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" >/dev/null; then
      break
    fi
  fi
  sleep 120
done
if [[ ! -f "$RUN/train/checkpoint-12/trainer_state.json" ]]; then
  log "ERROR: checkpoint-12 never appeared within 24h"
  exit 1
fi
log "checkpoint-12 present; waiting for GPUs to free"

deadline=$((SECONDS + 3600))
while (( SECONDS < deadline )); do
  u0=$(nvidia-smi --id="$TRAIN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  u1=$(nvidia-smi --id="$VLLM_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( u0 <= 512 && u1 <= 512 )); then break; fi
  sleep 30
done

log "starting matched eval -> $EVAL_RUN"
rc=0
ADAPTER="$RUN/train/checkpoint-12" RUN_DIR="$EVAL_RUN" CHECKPOINT_GLOBAL_STEP=12 \
  SOURCE_CHECKPOINT="newgnn:$RUN/train/checkpoint-12" \
  GPU0="$TRAIN_GPU" GPU1="$VLLM_GPU" PORT0=18410 PORT1=18411 \
  PROJECT="$PROJECT" \
  bash "$BASE/run_single_matched_eval_table_rl.sh" || rc=$?
log "matched eval finished rc=$rc"
exit "$rc"
