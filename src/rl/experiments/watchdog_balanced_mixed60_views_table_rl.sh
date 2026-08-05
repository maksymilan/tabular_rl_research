#!/usr/bin/env bash
set -euo pipefail
O=/home/dengyan/tabular_rl_outputs
R=$O/rl_runtime_rank_score_v3_20260731
status=$O/logs/balanced_mixed60_views_table_rl_20260801.status
if grep -q $'\tcomplete\t' "$status" 2>/dev/null; then exit 0; fi
if pgrep -u dengyan -f 'prepare_balanced_mixed60_views_table_rl.sh' >/dev/null 2>&1; then exit 0; fi
cd "$R"
nohup bash src/rl/experiments/prepare_balanced_mixed60_views_table_rl.sh \
  >>"$O/logs/balanced_mixed60_views_watchdog_table_rl_20260801.log" 2>&1 &
