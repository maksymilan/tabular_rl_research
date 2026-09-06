#!/usr/bin/env bash
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
POOL_DIR=$O/phase8_controlled_20260801/fixed_pool_60_seed101_causal_repair_v2 \
POOL_VALIDATION_STATUS=$O/logs/fixed_pool_causal_repair_table_rl_20260801.status \
STATUS=$O/logs/phase8_controlled_queue_v2_table_rl_20260801.status \
RUN_LOG=$O/logs/phase8_controlled_queue_v2_table_rl_20260801.log \
LOCK=$O/logs/phase8_controlled_queue_v2_table_rl_20260801.lock \
PHASE_ROOT=$O/phase8_controlled_20260801/stage1_causal_repair_v2 \
  bash "$R/src/rl/experiments/run_phase8_controlled_queue_table_rl.sh"
