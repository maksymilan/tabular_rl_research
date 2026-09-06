#!/usr/bin/env bash
# Cron-safe, fail-closed launcher for the conditional dense Stage2 queue.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_dense_stage2_20260802}
STATUS=${STATUS:-$O/logs/dense_stage2_scale120_queue_table_rl_20260802.status}
WATCHDOG_LOG=${WATCHDOG_LOG:-$O/logs/dense_stage2_scale120_watchdog_table_rl_20260802.log}
state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS" 2>/dev/null || printf missing)
case "$state" in
  complete|complete_no_scaleup|blocked|failed) exit 0 ;;
esac
printf '%s\tlaunch_or_poll\tprevious=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$state" >>"$WATCHDOG_LOG"
OUTPUT_ROOT="$O" TRAIN_RUNTIME="$R" \
  bash "$R/src/rl/experiments/run_dense_stage2_scale120_queue_table_rl.sh"
