#!/usr/bin/env bash
# Persistently generate one authorized NewGNN shard for the deterministic 480-task extension.
set -euo pipefail

GPU_ID=${1:?GPU id 6 or 7 is required}
case "$GPU_ID" in
  6) SHARD=02 ;;
  7) SHARD=03 ;;
  *) printf 'NewGNN append worker only permits GPU 6 or 7\n' >&2; exit 2 ;;
esac

O=/home/dengyan/tabular_rl_outputs
export HOST_ROLE=newgnn GPU_ID
export RUNTIME=$O/rl_runtime_phase8_parallel_20260801
export TASKS=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/tasks.jsonl
export TASK_ID_FILE=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/shards/shard-$SHARD.ids
export OUTPUT_DIR=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/worker$GPU_ID
export STATUS=$O/logs/mixed_pool_append480_newgnn_gpu${GPU_ID}_20260801.status
export RUN_LOG=$O/logs/mixed_pool_append480_newgnn_gpu${GPU_ID}_20260801.log
export TASK_BATCH_SIZE=8
export FREE_SAMPLES=10
export LAUNCHER=$RUNTIME/src/rl/experiments/run_mixed_pool_search_shard.sh
exec bash "$RUNTIME/src/rl/experiments/watchdog_mixed_pool_search_shard.sh"
