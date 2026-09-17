#!/usr/bin/env bash
# Three matched evals, all with VLLM_ENFORCE_EAGER=1 (table_rl).
#
# table_rl's vLLM compile path is broken: CUDA-graph capture dies with
#   RuntimeError: Bytes object is corrupted, checksum does not match
# (raised from torch inductor FxGraphCache -> triton_hash_with_backend), reproduced on every
# launch with identical expected/got bytes. In the same environment plain torch.compile and
# torch.utils._triton.triton_backend() succeed, and the same vLLM args with --enforce-eager
# start and serve normally, so the fault is confined to vLLM's compilation path. Isolated
# TORCHINDUCTOR_CACHE_ROOT / VLLM_CACHE_ROOT / TRITON_CACHE_DIR and sequential startup did not
# help. All arms therefore run with --enforce-eager, and the 909 baseline arm is re-run under
# the same setting so the comparison set stays internally matched. Greedy T=0 decoding is not
# affected by cuda-graph vs eager execution; the deviation is recorded in evaluation_config.
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

CACHE_BASE=$BASE/eval_caches_saam_decomposition_20260917
CHAIN_LOG=$BASE/matched_evals_saam_decomposition_20260917/evals.log
log() { echo "[$(date -Is)] $*" >>"$CHAIN_LOG"; }

run_eval() {
  local label=$1 adapter=$2 rundir=$3 source_checkpoint=$4 port0=$5 port1=$6
  mkdir -p "$rundir/controller" "$CACHE_BASE/$label/torchinductor" \
           "$CACHE_BASE/$label/vllm" "$CACHE_BASE/$label/triton"
  cp "$PROJECT/src/rl/evaluation/runners/formal_v26_rollout_passk.py" \
     "$PROJECT/src/rl/evaluation/runners/make_eval_shards.py" \
     "$PROJECT/src/rl/evaluation/runners/merge_eval_shards.py" "$rundir/controller/"
  log "eval $label start (triton libcuda shim set, isolated cache) adapter=$adapter rundir=$rundir"
  TORCHINDUCTOR_CACHE_ROOT="$CACHE_BASE/$label/torchinductor" \
  VLLM_CACHE_ROOT="$CACHE_BASE/$label/vllm" \
  TRITON_CACHE_DIR="$CACHE_BASE/$label/triton" \
  VLLM_START_SEQUENTIALLY=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
  ADAPTER="$adapter" RUN_DIR="$rundir" SOURCE_CHECKPOINT="$source_checkpoint" \
    CHECKPOINT_GLOBAL_STEP=4 GPU0=0 GPU1=1 PORT0="$port0" PORT1="$port1" \
    ERROR_FEEDBACK_VERSION=actionable-error-v1 EVAL_WORKERS=24 MAX_TOKENS=2048 \
    PYTHON_BIN="$PYTHON_BIN" RUNTIME="$RUNTIME" MODEL="$MODEL" \
    SOURCE_INPUT="$SOURCE_INPUT" DATABASE_ROOT="$DATABASE_ROOT" \
    bash "$LAUNCHER"
  local rc=$?
  log "eval $label finished rc=$rc"
  return $rc
}

run_eval grpo_span \
  "$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl/adapter" \
  "$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl_r4" \
  "newgnn:/home/dengyan/tabular_rl_outputs/qwen3_4b_grpo_span_newgnn_20260916_r1/train/checkpoint-4" \
  18406 18407 || true

run_eval saam_nospan \
  "$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl/adapter" \
  "$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl_r3" \
  "newgnn:/home/dengyan/tabular_rl_outputs/qwen3_4b_saam_nospan_newgnn_20260916_r1/train/checkpoint-4" \
  18408 18409 || true

log "isolated-cache chain done"
