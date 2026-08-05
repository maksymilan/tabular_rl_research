#!/usr/bin/env bash
set -euo pipefail

O=/home/dengyan/tabular_rl_outputs
export HOST_ROLE=table_rl GPU_ID=1
export RUNTIME=$O/rl_runtime_rank_score_v3_20260731
export TASKS=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101/tasks.jsonl
export TASK_ID_FILE=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101/shards/shard-00.ids
export OUTPUT_DIR=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101
export STATUS=$O/logs/mixed_pool_search_table_gpu1_20260801.status
export RUN_LOG=$O/logs/mixed_pool_search_table_gpu1_20260801.log
export TASK_BATCH_SIZE=4
export LAUNCHER=$RUNTIME/src/rl/experiments/run_mixed_pool_search_shard.sh
exec bash "$RUNTIME/src/rl/experiments/watchdog_mixed_pool_search_shard.sh"
