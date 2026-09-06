#!/usr/bin/env bash
# Balanced mixed60 controlled queue using independent Rank and Process views.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
RANK_POOL_DIR=$O/phase8_controlled_20260801/balanced_mixed60_rank_seed101 \
PROCESS_POOL_DIR=$O/phase8_controlled_20260801/balanced_mixed60_process_seed101 \
POOL_VALIDATION_STATUS=$O/logs/balanced_mixed60_views_table_rl_20260801.status \
STATUS=$O/logs/phase8_controlled_queue_v3_table_rl_20260801.status \
RUN_LOG=$O/logs/phase8_controlled_queue_v3_table_rl_20260801.log \
LOCK=$O/logs/phase8_controlled_queue_v3_table_rl_20260801.lock \
PHASE_ROOT=$O/phase8_controlled_20260801/stage1_balanced_mixed60_v3 \
  bash "$R/src/rl/experiments/run_phase8_controlled_queue_table_rl.sh"
