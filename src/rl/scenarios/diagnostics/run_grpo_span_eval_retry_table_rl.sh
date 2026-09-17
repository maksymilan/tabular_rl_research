#!/usr/bin/env bash
# Retry the GRPO+span matched eval after the current chain finishes.
#
# First attempt died at vLLM startup with
#   RuntimeError: Bytes object is corrupted, checksum does not match
# raised from vllm/compilation/cuda_graph.py while both shard servers compiled into the same
# TORCHINDUCTOR_CACHE_ROOT at the same time. This retry waits for the GPUs to be free and then
# starts the two servers sequentially (VLLM_START_SEQUENTIALLY=1) with the same identity knobs
# as the 909 baseline (actionable-error-v1, 24 workers, T=0, max_tokens 2048, max_steps 30).
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

RUNDIR=$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl_r2
ADAPTER=$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl/adapter
CHAIN_LOG=$BASE/matched_evals_saam_decomposition_20260917/evals.log
log() { echo "[$(date -Is)] $*" >>"$CHAIN_LOG"; }

log "retry watcher start (grpo_span)"
deadline=$((SECONDS + 6 * 3600))
while (( SECONDS < deadline )); do
  running=$(pgrep -fc "formal_v26_rollout_passk|run_qwen3_8b_v26_checkpoint_dataparallel" || true)
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$running" == "0" ]] && (( used0 <= 512 && used1 <= 512 )); then
    break
  fi
  sleep 60
done

log "retry: GPUs free, starting grpo_span eval in $RUNDIR"
mkdir -p "$RUNDIR/controller"
cp "$PROJECT/src/rl/evaluation/runners/formal_v26_rollout_passk.py" \
   "$PROJECT/src/rl/evaluation/runners/make_eval_shards.py" \
   "$PROJECT/src/rl/evaluation/runners/merge_eval_shards.py" "$RUNDIR/controller/"

set +e
ADAPTER="$ADAPTER" RUN_DIR="$RUNDIR" \
  SOURCE_CHECKPOINT="newgnn:/home/dengyan/tabular_rl_outputs/qwen3_4b_grpo_span_newgnn_20260916_r1/train/checkpoint-4" \
  CHECKPOINT_GLOBAL_STEP=4 GPU0=0 GPU1=1 PORT0=18394 PORT1=18395 \
  ERROR_FEEDBACK_VERSION=actionable-error-v1 EVAL_WORKERS=24 MAX_TOKENS=2048 \
  VLLM_START_SEQUENTIALLY=1 \
  PYTHON_BIN="$PYTHON_BIN" RUNTIME="$RUNTIME" MODEL="$MODEL" \
  SOURCE_INPUT="$SOURCE_INPUT" DATABASE_ROOT="$DATABASE_ROOT" \
  bash "$LAUNCHER"
rc=$?
set -e
log "retry grpo_span finished rc=$rc"
