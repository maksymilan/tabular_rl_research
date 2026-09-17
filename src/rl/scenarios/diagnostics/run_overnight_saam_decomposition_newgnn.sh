#!/usr/bin/env bash
# Overnight SAAM-attribution decomposition pipeline (NewGNN, unattended).
#
# Stage 1: train the plain-GRPO (credit_assignment=trajectory, no SAAM) + span-balanced arm
#          with the same cohort/seed/reward/budget as the correctness-only SAAM arm.
# Stage 2: matched BIRD-dev1534 greedy eval of that arm's checkpoint-4.
# Stage 3: matched BIRD-dev1534 greedy eval of the already-trained SAAM-without-span arm.
#
# Stage 3 runs even if stage 1 fails, so the completed run still gets measured. Every stage
# writes its return code into the pipeline log and the status file; GPUs are re-checked for
# idleness before each eval and the trainers/eval controllers use their own cleanup traps.
set -Eeuo pipefail

BASE=${BASE:-/home/dengyan/tabular_rl_outputs}
PIPELINE_ROOT=${PIPELINE_ROOT:-$BASE/overnight_saam_decomposition_20260916}
PROJECT=${PROJECT:-$BASE/qwen3_4b_saam_nospan_newgnn_20260916_r1/project}
TRAIN_RUN=$BASE/qwen3_4b_grpo_span_newgnn_20260916_r1
NOSPAN_RUN=$BASE/qwen3_4b_saam_nospan_newgnn_20260916_r1
EVAL_GRPO=$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260916_newgnn
EVAL_NOSPAN=$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260916_newgnn

PYTHON_BIN=/home/dengyan/miniconda3/envs/trl-table/bin/python
RUNTIME=$BASE/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
MODEL=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
SOURCE_INPUT=/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl
DATABASE_ROOT=/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases
EVAL_LAUNCHER="$PROJECT/src/rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh"

TRAIN_GPU=${TRAIN_GPU:-6}
VLLM_GPU=${VLLM_GPU:-7}
EVAL_PORT0=${EVAL_PORT0:-18382}
EVAL_PORT1=${EVAL_PORT1:-18383}

mkdir -p "$PIPELINE_ROOT"
LOG="$PIPELINE_ROOT/pipeline.log"
STATUS="$PIPELINE_ROOT/status"
exec >>"$LOG" 2>&1

log() { echo "[$(date -Is)] $*"; }
set_status() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >"$STATUS"; }

wait_gpus_idle() {
  local deadline=$((SECONDS + 1800)) used0 used1
  while (( SECONDS < deadline )); do
    used0=$(nvidia-smi --id="$TRAIN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    used1=$(nvidia-smi --id="$VLLM_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used0 <= 512 && used1 <= 512 )); then
      log "GPUs $TRAIN_GPU/$VLLM_GPU idle (${used0}/${used1} MiB)"
      return 0
    fi
    sleep 30
  done
  log "ERROR: GPUs $TRAIN_GPU/$VLLM_GPU not idle within 30 minutes"
  return 1
}

run_eval() {
  local adapter=$1 rundir=$2 label=$3
  wait_gpus_idle || return 1
  # The launcher is fail-closed on existing output dirs; use a fresh directory per attempt.
  if [[ -e "$rundir" ]]; then
    log "eval $label: output dir already exists, refusing to reuse: $rundir"
    return 1
  fi
  mkdir -p "$rundir/controller"
  cp "$PROJECT/src/rl/evaluation/runners/formal_v26_rollout_passk.py" \
     "$PROJECT/src/rl/evaluation/runners/make_eval_shards.py" \
     "$PROJECT/src/rl/evaluation/runners/merge_eval_shards.py" "$rundir/controller/"
  # The copied controllers import rl.* and vLLM's CUDA-graph capture needs a linker-visible
  # libcuda.so; both were missing in the first version of this pipeline. See
  # docs/current/rl_pipeline.md ("流水线与评测启动检查清单").
  export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
  ADAPTER="$adapter" RUN_DIR="$rundir" SOURCE_CHECKPOINT="$adapter" CHECKPOINT_GLOBAL_STEP=4 \
    GPU0="$TRAIN_GPU" GPU1="$VLLM_GPU" PORT0="$EVAL_PORT0" PORT1="$EVAL_PORT1" \
    ERROR_FEEDBACK_VERSION=actionable-error-v1 EVAL_WORKERS=24 MAX_TOKENS=2048 \
    TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    VLLM_START_SEQUENTIALLY=1 \
    PYTHON_BIN="$PYTHON_BIN" RUNTIME="$RUNTIME" MODEL="$MODEL" \
    SOURCE_INPUT="$SOURCE_INPUT" DATABASE_ROOT="$DATABASE_ROOT" \
    bash "$EVAL_LAUNCHER"
  local rc=$?
  log "eval $label finished rc=$rc"
  return $rc
}

log "pipeline start; project=$PROJECT train_run=$TRAIN_RUN"
set_status stage1_training_grpo_span "started"
train_rc=0
(
  cd "$PROJECT"
  PROJECT_DIR="$PROJECT" RUN_ROOT="$TRAIN_RUN" TRAIN_GPU="$TRAIN_GPU" VLLM_GPU="$VLLM_GPU" \
    VLLM_PORT=18381 VLLM_GROUP_PORT=51481 \
    bash "$PROJECT/src/rl/scenarios/diagnostics/run_qwen3_4b_grpo_span_newgnn.sh"
) || train_rc=$?
log "stage1 training rc=$train_rc"

if (( train_rc == 0 )); then
  set_status stage2_eval_grpo_span "started"
  if run_eval "$TRAIN_RUN/train/checkpoint-4" "$EVAL_GRPO" grpo_span; then
    log "stage2 ok: $EVAL_GRPO"
  else
    log "stage2 FAILED"
  fi
else
  log "skipping stage2: training failed"
fi

set_status stage3_eval_saam_nospan "started"
if run_eval "$NOSPAN_RUN/train/checkpoint-4" "$EVAL_NOSPAN" saam_nospan; then
  log "stage3 ok: $EVAL_NOSPAN"
else
  log "stage3 FAILED"
fi

set_status pipeline_done "stage1_rc=$train_rc"
log "pipeline done"
