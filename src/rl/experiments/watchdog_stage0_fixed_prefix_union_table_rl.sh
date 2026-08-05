#!/usr/bin/env bash
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/eval_runtime_version36_20260728}
S="$O/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.status"
L="$O/logs/stage0_fixed_prefix_union_watchdog_20260801.latest"
H="$O/logs/stage0_fixed_prefix_union_watchdog_20260801.history"
K="$O/logs/stage0_fixed_prefix_union_watchdog_20260801.lock"
mkdir -p "$O/logs"
exec 9>"$K"; flock -n 9 || exit 0
state=$(awk -F '\t' 'NR==1 {print $2}' "$S" 2>/dev/null || printf missing)
pids=$(pgrep -f '[r]un_stage0_fixed_prefix_union_queue_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
if [[ "$state" == complete ]]; then action=done
elif [[ "$state" == failed || "$state" == blocked ]]; then action=fail_closed
elif [[ -z "$pids" ]]; then
  cd "$R"; nohup bash src/rl/experiments/run_stage0_fixed_prefix_union_queue_table_rl.sh \
    >>"$O/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.nohup.log" 2>&1 </dev/null &
  pids=$!; action=restarted_queue
fi
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tstate=$state pids=${pids:-none}"
printf '%b\n' "$line" >"$L"; printf '%b\n' "$line" >>"$H"
