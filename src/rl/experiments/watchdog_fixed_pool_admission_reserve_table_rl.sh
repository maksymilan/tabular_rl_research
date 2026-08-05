#!/usr/bin/env bash
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
S="$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.status"
L="$O/logs/fixed_pool_admission_reserve_watchdog_20260801.latest"
H="$O/logs/fixed_pool_admission_reserve_watchdog_20260801.history"
K="$O/logs/fixed_pool_admission_reserve_watchdog_20260801.lock"
mkdir -p "$O/logs"
exec 9>"$K"
flock -n 9 || exit 0
state=$(awk -F '\t' 'NR==1 {print $2}' "$S" 2>/dev/null || printf missing)
pids=$(pgrep -f '[r]un_fixed_pool_admission_reserve_generation_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
if [[ "$state" == complete ]]; then
  action=done
elif [[ "$state" == failed ]]; then
  action=fail_closed
elif [[ -z "$pids" ]]; then
  cd "$R"
  nohup bash src/rl/experiments/run_fixed_pool_admission_reserve_generation_table_rl.sh \
    >>"$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.nohup.log" \
    2>&1 </dev/null &
  pids=$!
  action=restarted_queue
fi
groups=$(find "$O/phase8_controlled_20260801/fixed_pool_60_seed101_admission_reserve_v1/groups" \
  -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tstate=$state pids=${pids:-none} groups=$groups/12"
printf '%b\n' "$line" >"$L"
printf '%b\n' "$line" >>"$H"
