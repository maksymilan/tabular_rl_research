#!/usr/bin/env bash
# Persistently generate one of the two table_rl shards for the deterministic 480-task extension.
set -euo pipefail

GPU_ID=${1:?GPU id 0 or 1 is required}
case "$GPU_ID" in
  0) SHARD=00 ;;
  1) SHARD=01 ;;
  *) printf 'table_rl append worker only permits GPU 0 or 1\n' >&2; exit 2 ;;
esac

O=/home/dengyan/tabular_rl_outputs
export HOST_ROLE=table_rl GPU_ID
export RUNTIME=$O/rl_runtime_rank_score_v3_20260731
export TASKS=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/tasks.jsonl
export TASK_ID_FILE=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/shards/shard-$SHARD.ids
export OUTPUT_DIR=$O/phase8_mixed_pool_search_20260801/candidate_480_seed101/worker$GPU_ID
export STATUS=$O/logs/mixed_pool_append480_table_gpu${GPU_ID}_20260801.status
export RUN_LOG=$O/logs/mixed_pool_append480_table_gpu${GPU_ID}_20260801.log
export TASK_BATCH_SIZE=8
export FREE_SAMPLES=1
export LAUNCHER=$RUNTIME/src/rl/experiments/run_mixed_pool_search_shard.sh
exec bash "$RUNTIME/src/rl/experiments/watchdog_mixed_pool_search_shard.sh"
