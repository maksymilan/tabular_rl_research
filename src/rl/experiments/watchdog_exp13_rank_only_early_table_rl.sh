#!/usr/bin/env bash
# Use GPU1 for Rank-only training as soon as the independent rank view is frozen.
set -euo pipefail
O=/home/dengyan/tabular_rl_outputs
R=$O/rl_runtime_rank_score_v3_20260731
POOL=$O/phase8_controlled_20260801/balanced_mixed60_rank_seed101
ARTIFACT=trl-transition-v26-exp13-fixed-rank-only-action-mean-sft2-60xk4-seed101-20260801
STATUS=$O/logs/stage1_exp13_fixed_rank_only_action_mean.train.status
RUN_LOG=$O/logs/stage1_exp13_fixed_rank_only_action_mean.train.log
WATCH=$O/logs/stage1_exp13_early_watchdog_20260801.status
if [[ ! -f "$POOL/manifest.json" ]] || ! grep -q '"status": "frozen_rank_ready"' "$POOL/manifest.json"; then
  printf '%s\twaiting_rank_view\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$POOL" >"$WATCH"
  exit 0
fi
if grep -q $'\tcomplete\t' "$STATUS" 2>/dev/null; then
  printf '%s\tcomplete\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ARTIFACT" >"$WATCH"
  exit 0
fi
printf '%s\tstarting_or_observing\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ARTIFACT" >"$WATCH"
cd "$R"
nohup env \
  EXPERIMENT_NAME=exp13_fixed_rank_only_action_mean \
  EXPERIMENT_CONFIG="$R/src/rl/configs/experiments/exp13_fixed_rank_only_action_mean.yaml" \
  ARTIFACT="$ARTIFACT" STATUS="$STATUS" RUN_LOG="$RUN_LOG" \
  TRAIN_RUNTIME="$R" OUTPUT_ROOT="$O" POOL_DIR="$POOL" GPU_ID=1 \
  bash src/rl/experiments/run_fixed_pool_controlled_train_table_rl.sh \
  >>"$O/logs/stage1_exp13_early_watchdog_20260801.log" 2>&1 &
