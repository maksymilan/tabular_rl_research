#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_rank_score_v3_20260731}
STATUS="$OUTPUT_ROOT/logs/fixed_pool60_validation_table_rl_20260801.status"
LATEST="$OUTPUT_ROOT/logs/fixed_pool60_validation_watchdog_20260801.latest"
HISTORY="$OUTPUT_ROOT/logs/fixed_pool60_validation_watchdog_20260801.history"
LOCK="$OUTPUT_ROOT/logs/fixed_pool60_validation_watchdog_20260801.lock"
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$LOCK"
flock -n 9 || exit 0
state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS" 2>/dev/null || printf 'missing')
pids=$(pgrep -f '[r]un_fixed_pool60_validation_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
if [[ "$state" == complete ]]; then
  action=done
elif [[ "$state" == failed || "$state" == blocked ]]; then
  action=fail_closed
elif [[ -z "$pids" ]]; then
  cd "$RUNTIME"
  nohup bash src/rl/experiments/run_fixed_pool60_validation_table_rl.sh \
    >>"$OUTPUT_ROOT/logs/fixed_pool60_validation_table_rl_20260801.nohup.log" \
    2>&1 </dev/null &
  action=restarted_queue
  pids=$!
fi
pool="$OUTPUT_ROOT/phase8_controlled_20260801/fixed_pool_60_seed101"
groups=$(find "$pool/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
frozen=$(test -f "$pool/manifest.json" && printf yes || printf no)
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tstate=$state pids=${pids:-none} groups=$groups/60 frozen=$frozen"
printf '%b\n' "$line" >"$LATEST"
printf '%b\n' "$line" >>"$HISTORY"
