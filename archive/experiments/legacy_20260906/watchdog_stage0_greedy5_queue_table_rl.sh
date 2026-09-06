#!/usr/bin/env bash
# Cron-safe watchdog for the diagnostic greedy queue. Failed jobs remain fail-closed.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/stage0_greedy5_queue_table_rl_20260801.status}
LATEST=${LATEST:-$OUTPUT_ROOT/logs/stage0_greedy5_watchdog_table_rl_20260801.latest}
HISTORY=${HISTORY:-$OUTPUT_ROOT/logs/stage0_greedy5_watchdog_table_rl_20260801.history}
LOCK=${LOCK:-$OUTPUT_ROOT/logs/stage0_greedy5_watchdog_table_rl_20260801.lock}
QUEUE_SCRIPT=${QUEUE_SCRIPT:-$RUNTIME/src/rl/experiments/run_stage0_greedy5_queue_table_rl.sh}

mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$LOCK"
flock -n 9 || exit 0

timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS" 2>/dev/null || printf 'missing')
queue_pids=$(pgrep -f '[r]un_stage0_greedy5_queue_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
heavy_pids=$(ps -eo pid=,args= | awk '
  $0 ~ /(rollout_passk.py|vllm.entrypoints.openai.api_server|run_equal300_greedy_table_rl.sh)/ { print $1 }
' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
detail="state=$state queue_pids=${queue_pids:-none} heavy_pids=${heavy_pids:-none}"

case "$state" in
  complete)
    action=done
    ;;
  failed | blocked)
    action=fail_closed
    ;;
  *)
    if [[ -z "$queue_pids" ]]; then
      cd "$RUNTIME"
      nohup bash "$QUEUE_SCRIPT" \
        >>"$OUTPUT_ROOT/logs/stage0_greedy5_queue_table_rl_20260801.nohup.log" \
        2>&1 </dev/null &
      action=restarted_queue
      detail="$detail new_pid=$!"
    fi
    ;;
esac

line="$timestamp\t$action\t$detail"
printf '%b\n' "$line" >"$LATEST"
printf '%b\n' "$line" >>"$HISTORY"
