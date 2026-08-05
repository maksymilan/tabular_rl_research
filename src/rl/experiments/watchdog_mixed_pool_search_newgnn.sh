#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?GPU id 6 or 7 is required}
case "$GPU_ID" in
  6) SHARD=01 ;;
  7) SHARD=02 ;;
  *) printf 'NewGNN mixed-pool worker only permits GPU 6 or 7\n' >&2; exit 2 ;;
esac
O=/home/dengyan/tabular_rl_outputs
RUNTIME=$O/rl_runtime_phase8_parallel_20260801
HARDWARE_PROFILE=${HARDWARE_PROFILE:-$RUNTIME/src/rl/configs/hardware/rtx3090_24gb.sh}
test -f "$HARDWARE_PROFILE"
# shellcheck source=../configs/hardware/rtx3090_24gb.sh
source "$HARDWARE_PROFILE"
export HOST_ROLE=newgnn GPU_ID
export RUNTIME
export TASKS=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101/tasks.jsonl
export TASK_ID_FILE=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101/shards/shard-$SHARD.ids
export OUTPUT_DIR=$O/phase8_mixed_pool_search_20260801/candidate_360_seed101/worker$GPU_ID
export STATUS=$O/logs/mixed_pool_search_newgnn_gpu${GPU_ID}_20260801.status
export RUN_LOG=$O/logs/mixed_pool_search_newgnn_gpu${GPU_ID}_20260801.log
export TASK_BATCH_SIZE=${TASK_BATCH_SIZE:-$RTX3090_ROLLOUT_TASK_BATCH_SIZE}
export FREE_SAMPLES=${FREE_SAMPLES:-$RTX3090_SHARED_HOST_FREE_SAMPLES}
export LAUNCHER=$RUNTIME/src/rl/experiments/run_mixed_pool_search_shard.sh
exec bash "$RUNTIME/src/rl/experiments/watchdog_mixed_pool_search_shard.sh"
