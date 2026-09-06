#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
STATUS="$OUTPUT_ROOT/logs/fixed_prefix_rescore5_queue_table_rl_20260801.status"
LATEST="$OUTPUT_ROOT/logs/fixed_prefix_rescore5_watchdog_table_rl_20260801.latest"
HISTORY="$OUTPUT_ROOT/logs/fixed_prefix_rescore5_watchdog_table_rl_20260801.history"
LOCK="$OUTPUT_ROOT/logs/fixed_prefix_rescore5_watchdog_table_rl_20260801.lock"
exec 9>"$LOCK"
flock -n 9 || exit 0
state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS" 2>/dev/null || printf 'missing')
pids=$(pgrep -f '[r]un_fixed_prefix_rescore5_queue_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
if [[ "$state" == complete ]]; then
  action=done
elif [[ "$state" == failed ]]; then
  action=fail_closed
elif [[ -z "$pids" ]]; then
  cd "$RUNTIME"
  nohup bash src/rl/experiments/run_fixed_prefix_rescore5_queue_table_rl.sh \
    >>"$OUTPUT_ROOT/logs/fixed_prefix_rescore5_queue_table_rl_20260801.nohup.log" \
    2>&1 </dev/null &
  action=restarted_queue
  pids=$!
fi
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tstate=$state pids=${pids:-none}"
printf '%b\n' "$line" >"$LATEST"
printf '%b\n' "$line" >>"$HISTORY"
