#!/usr/bin/env bash
set -uo pipefail

REPRO_ROOT="${REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/text2sql_reproduction}"
SCRIPT_DIR="${SCRIPT_DIR:-$REPRO_ROOT/scripts}"
LOG_DIR="$REPRO_ROOT/logs"
QUEUE_LOG="$LOG_DIR/full_bird_queue.log"
STATUS_FILE="$LOG_DIR/full_bird_queue.status"

mkdir -p "$LOG_DIR"

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'stage=arctic_full_greedy\n'
} >"$STATUS_FILE"

EVAL_SCOPE=full N=1 TEMPERATURE=0.0 \
  bash "$SCRIPT_DIR/run_arctic_bird.sh" >>"$QUEUE_LOG" 2>&1
arctic_rc=$?
printf 'arctic_exit_code=%s\n' "$arctic_rc" >>"$STATUS_FILE"

printf 'stage=sqlr1_full_majority_vote\n' >>"$STATUS_FILE"
EVAL_SCOPE=full N=8 TEMPERATURE=0.8 \
  bash "$SCRIPT_DIR/run_sqlr1_bird.sh" >>"$QUEUE_LOG" 2>&1
sqlr1_rc=$?
{
  printf 'sqlr1_exit_code=%s\n' "$sqlr1_rc"
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'stage=finished\n'
} >>"$STATUS_FILE"

if [[ "$arctic_rc" -ne 0 || "$sqlr1_rc" -ne 0 ]]; then
  exit 1
fi
