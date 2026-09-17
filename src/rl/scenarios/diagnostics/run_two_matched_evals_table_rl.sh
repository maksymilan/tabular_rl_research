#!/usr/bin/env bash
# Sequential matched BIRD-dev1534 evals for the SAAM-decomposition arms (table_rl).
#
# Repairs the overnight pipeline's eval stage: the data-parallel eval controllers are copied
# into the run directory and import `rl.*`, so the caller MUST export PYTHONPATH pointing at
# the frozen project tree's src. Without it the controllers fail with
# "ModuleNotFoundError: No module named 'rl'" before any GPU work.
set -Eeuo pipefail

BASE=/home/dengyan/tabular_rl_outputs
PROJECT=${PROJECT:-$BASE/qwen3_4b_later_error_quarter_saam60_20260915_r1/project}
PYTHON_BIN=/home/dengyan/miniconda3/envs/trl-table/bin/python
RUNTIME=$BASE/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
MODEL=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
SOURCE_INPUT=/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl
DATABASE_ROOT=/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases
LAUNCHER=$PROJECT/src/rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

CHAIN_ROOT=$BASE/matched_evals_saam_decomposition_20260917
LOG=$CHAIN_ROOT/evals.log
mkdir -p "$CHAIN_ROOT"
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }

run_eval() {
  local label=$1 adapter=$2 rundir=$3 source_checkpoint=$4
  log "eval $label start adapter=$adapter rundir=$rundir"
  mkdir -p "$rundir/controller"
  cp "$PROJECT/src/rl/evaluation/runners/formal_v26_rollout_passk.py" \
     "$PROJECT/src/rl/evaluation/runners/make_eval_shards.py" \
     "$PROJECT/src/rl/evaluation/runners/merge_eval_shards.py" "$rundir/controller/"
  ADAPTER="$adapter" RUN_DIR="$rundir" SOURCE_CHECKPOINT="$source_checkpoint" \
    CHECKPOINT_GLOBAL_STEP=4 GPU0=0 GPU1=1 PORT0=18392 PORT1=18393 \
    ERROR_FEEDBACK_VERSION=actionable-error-v1 EVAL_WORKERS=24 MAX_TOKENS=2048 \
    PYTHON_BIN="$PYTHON_BIN" RUNTIME="$RUNTIME" MODEL="$MODEL" \
    SOURCE_INPUT="$SOURCE_INPUT" DATABASE_ROOT="$DATABASE_ROOT" \
    bash "$LAUNCHER"
  local rc=$?
  log "eval $label finished rc=$rc"
  return $rc
}

log "chain start; project=$PROJECT pythonpath=$PYTHONPATH"

run_eval grpo_span \
  "$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl/adapter" \
  "$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl" \
  "newgnn:/home/dengyan/tabular_rl_outputs/qwen3_4b_grpo_span_newgnn_20260916_r1/train/checkpoint-4" || true

run_eval saam_nospan \
  "$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl/adapter" \
  "$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl" \
  "newgnn:/home/dengyan/tabular_rl_outputs/qwen3_4b_saam_nospan_newgnn_20260916_r1/train/checkpoint-4" || true

log "chain done"
