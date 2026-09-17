#!/usr/bin/env bash
# Follow-up queue: wait for the running saam60 queue to finish, then continue the 180-question
# arm for a second epoch (updates 6 -> 12) from its own checkpoint-6 and evaluate it.
#
# Ordering matters: table_rl has two GPUs and the running queue owns them, so this script waits
# (up to 30h) for its status to reach queue_done before claiming the cards.
set -Eeuo pipefail

BASE=/home/dengyan/tabular_rl_outputs
A_QUEUE=$BASE/saam60_updates6_queue_20260917
A_STATUS=$A_QUEUE/status
PROJ=${PROJ:-$BASE/qwen3_4b_saam60_updates6_20260917_r1/project}
RUN_180=$BASE/qwen3_4b_correctness_only_saam180_table_rl_20260916_r1
E2_RUN=$BASE/qwen3_4b_saam180_epoch2_20260917_r1
# The resume contract requires the checkpoint to be an immediate child of the output directory
# and the directory to already hold the prior run's manifest/lock, so the checkpoint was staged
# there up front; the original 1-epoch run root stays untouched.
RESUME_CKPT=$E2_RUN/train/checkpoint-6
E2_EVAL=$BASE/evaluations/qwen3_4b_saam180_epoch2_checkpoint12_actionable_20260917_table_rl
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
FOLLOW_ROOT=$BASE/saam180_epoch2_queue_20260917
LOG=$FOLLOW_ROOT/queue.log
mkdir -p "$FOLLOW_ROOT"
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }
set_status() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >"$FOLLOW_ROOT/status"; }

log "follow-up queue start; waiting for $A_STATUS to reach queue_done"
deadline=$((SECONDS + 30 * 3600))
while (( SECONDS < deadline )); do
  if [[ -f "$A_STATUS" ]] && grep -q "queue_done" "$A_STATUS"; then break; fi
  sleep 120
done
if ! grep -q "queue_done" "$A_STATUS" 2>/dev/null; then
  log "ERROR: saam60 queue did not finish within 30h; aborting follow-up"
  set_status followup_aborted "saam60 queue unfinished"
  exit 1
fi
log "saam60 queue done; starting 180-epoch2 resume"

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

set_status stage1_train_180_epoch2 "resume from checkpoint-6, updates 6 -> 12"
wait_idle || true
train_rc=0
PROJECT_DIR="$PROJ" RUN_ROOT="$E2_RUN" RESUME_CHECKPOINT="$RESUME_CKPT" \
  TRAIN_GPU="$TRAIN_GPU" VLLM_GPU="$VLLM_GPU" \
  bash "$PROJ/src/rl/scenarios/diagnostics/run_qwen3_4b_saam180_epoch2_resume_table_rl.sh" || train_rc=$?
log "180-epoch2 training rc=$train_rc"

if (( train_rc == 0 )); then
  set_status stage2_eval_180_epoch2 "matched BIRD-dev1534 at checkpoint-12"
  wait_idle || true
  eval_rc=0
  ADAPTER="$E2_RUN/train/checkpoint-12" RUN_DIR="$E2_EVAL" CHECKPOINT_GLOBAL_STEP=12 \
    GPU0="$TRAIN_GPU" GPU1="$VLLM_GPU" \
    bash "$BASE/run_single_matched_eval_table_rl.sh" || eval_rc=$?
  log "180-epoch2 eval rc=$eval_rc"
else
  log "skipping eval: training failed"
fi

set_status followup_done "train_rc=$train_rc"
log "follow-up queue done"
