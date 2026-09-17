#!/usr/bin/env bash
# One matched BIRD-dev1534 checkpoint eval on table_rl, with the environment traps handled:
#   * PYTHONPATH=<project>/src  (controllers import rl.*)
#   * TRITON_LIBCUDA_PATH=<python env>/var/triton-libcuda  (vLLM CUDA-graph capture needs it)
#   * a fresh RUN_DIR (the launcher refuses to reuse an existing one)
#   * fail-closed GPU idleness check with a bounded wait
# See docs/current/rl_pipeline.md ("流水线与评测启动检查清单").
#
# Required env: ADAPTER, RUN_DIR, CHECKPOINT_GLOBAL_STEP
# Optional env: PORT0 (18410) PORT1 (18411) GPU0 (0) GPU1 (1) SOURCE_CHECKPOINT PROJECT
set -Eeuo pipefail

ADAPTER=${ADAPTER:?ADAPTER must point to the adapter directory to evaluate}
RUN_DIR=${RUN_DIR:?RUN_DIR must point to a fresh evaluation directory}
CHECKPOINT_GLOBAL_STEP=${CHECKPOINT_GLOBAL_STEP:?CHECKPOINT_GLOBAL_STEP}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
PORT0=${PORT0:-18410}
PORT1=${PORT1:-18411}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$ADAPTER}

BASE=/home/dengyan/tabular_rl_outputs
PROJECT=${PROJECT:-$BASE/qwen3_4b_later_error_quarter_saam60_20260915_r1/project}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
PYTHON_ENV=$(dirname "$(dirname "$PYTHON_BIN")")
TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-"$PYTHON_ENV/var/triton-libcuda"}
RUNTIME=${RUNTIME:-$BASE/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
MODEL=${MODEL:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}
SOURCE_INPUT=${SOURCE_INPUT:-/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl}
DATABASE_ROOT=${DATABASE_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases}
LAUNCHER=$PROJECT/src/rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_LIBCUDA_PATH

LOG_ROOT=$BASE/single_matched_evals_20260917
mkdir -p "$LOG_ROOT"
LOG=$LOG_ROOT/$(basename "$RUN_DIR").log
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }

if [[ -e "$RUN_DIR" ]]; then
  log "ERROR: RUN_DIR already exists, refusing to reuse: $RUN_DIR"
  exit 2
fi
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { log "ERROR: adapter incomplete: $ADAPTER"; exit 2; }
[[ -e "$TRITON_LIBCUDA_PATH/libcuda.so" ]] || { log "ERROR: triton libcuda shim missing"; exit 2; }

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  used0=$(nvidia-smi --id="$GPU0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  used1=$(nvidia-smi --id="$GPU1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used0 <= 512 && used1 <= 512 )); then break; fi
  sleep 30
done
(( used0 <= 512 && used1 <= 512 )) || { log "ERROR: GPUs not idle (${used0}/${used1} MiB)"; exit 75; }

mkdir -p "$RUN_DIR/controller"
cp "$PROJECT/src/rl/evaluation/runners/formal_v26_rollout_passk.py" \
   "$PROJECT/src/rl/evaluation/runners/make_eval_shards.py" \
   "$PROJECT/src/rl/evaluation/runners/merge_eval_shards.py" "$RUN_DIR/controller/"

log "eval start adapter=$ADAPTER run_dir=$RUN_DIR step=$CHECKPOINT_GLOBAL_STEP"
set +e
ADAPTER="$ADAPTER" RUN_DIR="$RUN_DIR" SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" \
  CHECKPOINT_GLOBAL_STEP="$CHECKPOINT_GLOBAL_STEP" GPU0="$GPU0" GPU1="$GPU1" \
  PORT0="$PORT0" PORT1="$PORT1" \
  ERROR_FEEDBACK_VERSION=actionable-error-v1 EVAL_WORKERS=24 MAX_TOKENS=2048 \
  VLLM_START_SEQUENTIALLY=1 \
  PYTHON_BIN="$PYTHON_BIN" RUNTIME="$RUNTIME" MODEL="$MODEL" \
  SOURCE_INPUT="$SOURCE_INPUT" DATABASE_ROOT="$DATABASE_ROOT" \
  bash "$LAUNCHER"
rc=$?
set -e
log "eval finished rc=$rc"
exit $rc
