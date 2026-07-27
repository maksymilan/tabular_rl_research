#!/usr/bin/env bash
set -euo pipefail

# Server-side watchdog for the unattended checkpoint-560 evaluation chain.
# It never starts a duplicate and only restarts after all owned service ports
# are free. Persistent failures stop after a small bounded retry count.

PROJECT_DIR="${PROJECT_DIR:-/home/dengyan/tabular_rl_outputs/eval_runtime_20260726}"
CHAIN_SESSION="${CHAIN_SESSION:-bird_cp560_overnight_eval}"
CONCURRENCY="${CONCURRENCY:-24}"
STATUS_FILE="${STATUS_FILE:-/home/dengyan/tabular_rl_outputs/logs/bird_cp560_overnight_eval_chain.status}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-/home/dengyan/tabular_rl_outputs/logs/bird_cp560_eval_supervisor.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
PORTS=(18035 18036 18037 18038)

mkdir -p "$(dirname "$SUPERVISOR_LOG")"
exec >>"$SUPERVISOR_LOG" 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

chain_complete() {
  test -f "$STATUS_FILE" && grep -q $'\tcomplete\t' "$STATUS_FILE"
}

owned_port_in_use() {
  local port
  for port in "${PORTS[@]}"; do
    if ss -ltnH "sport = :$port" | grep -q .; then
      return 0
    fi
  done
  return 1
}

restart_count=0
printf '%s supervisor started concurrency=%s\n' "$(timestamp)" "$CONCURRENCY"
while true; do
  if chain_complete; then
    printf '%s chain reports complete; supervisor exiting\n' "$(timestamp)"
    exit 0
  fi
  if tmux has-session -t "$CHAIN_SESSION" 2>/dev/null; then
    sleep "$POLL_SECONDS"
    continue
  fi
  if owned_port_in_use; then
    printf '%s chain missing but an owned port remains live; refusing duplicate restart\n' \
      "$(timestamp)"
    sleep "$POLL_SECONDS"
    continue
  fi
  restart_count=$((restart_count + 1))
  if test "$restart_count" -gt "$MAX_RESTARTS"; then
    printf '%s restart limit exceeded (%s); supervisor exiting for manual diagnosis\n' \
      "$(timestamp)" "$MAX_RESTARTS"
    exit 75
  fi
  printf '%s restarting chain attempt=%s\n' "$(timestamp)" "$restart_count"
  tmux new-session -d -s "$CHAIN_SESSION" -c "$PROJECT_DIR" \
    env PROJECT_DIR="$PROJECT_DIR" CONCURRENCY="$CONCURRENCY" \
    bash src/eval/run_bird_cp560_overnight_eval_chain_local_gpu.sh
  sleep "$POLL_SECONDS"
done
