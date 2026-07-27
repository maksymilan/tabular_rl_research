#!/usr/bin/env bash
set -euo pipefail

# Queue the checkpoint-560 K=4 evaluation behind the already-running greedy
# full-dev job. The queue never starts while the greedy tmux session is live,
# and it requires exactly 1,534 durable greedy records before proceeding.

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
GREEDY_SESSION=${GREEDY_SESSION:-bird_cp560_dev1534}
EXPECTED_TASKS=${EXPECTED_TASKS:-1534}
POLL_SECONDS=${POLL_SECONDS:-5}
GREEDY_RESULT_DIR="$PROJECT_DIR/data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_greedy1_bird_set"
PASSK_RESULT_DIR="$PROJECT_DIR/data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk4_bird_set"
QUEUE_LOG=${QUEUE_LOG:-/tmp/qwen25_coder7b_cp560_passk4_after_greedy.queue.log}

exec >>"$QUEUE_LOG" 2>&1

completed_rows() {
  local path="$1"
  if test -f "$path"; then
    wc -l < "$path" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

printf '%s waiting for greedy full-dev completion\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
while true; do
  greedy_rows=$(completed_rows "$GREEDY_RESULT_DIR/all.jsonl")
  if test "$greedy_rows" -eq "$EXPECTED_TASKS" &&
     ! tmux has-session -t "$GREEDY_SESSION" 2>/dev/null; then
    break
  fi
  sleep "$POLL_SECONDS"
done

passk_rows=$(completed_rows "$PASSK_RESULT_DIR/all.jsonl")
if test "$passk_rows" -eq "$EXPECTED_TASKS"; then
  printf '%s pass@K result is already complete; nothing to start\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  exit 0
fi

printf '%s greedy complete; starting checkpoint-560 K=4 evaluation on GPU 1\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cd "$PROJECT_DIR"
exec /usr/bin/caffeinate -dims /bin/bash \
  src/eval/run_bird_coder_cp560_tool_passk4_dev1534_table_rl.sh
