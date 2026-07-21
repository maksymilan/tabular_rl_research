#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
REMOTE_PORT=${REMOTE_PORT:-8015}
LOCAL_PORT=${LOCAL_PORT:-18015}
SERVED_MODEL=${SERVED_MODEL:-qwen25_bird_sft1_grounded_all651}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/data/results/qwen2.5_7b_bird_sft1_grounded_v4d_all651_bird_dev_stratified30}
LOG=${LOG:-/tmp/bird_sft1_grounded_v4d_all651_dev30_pipeline.log}

exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(timestamp) waiting for the remote training/evaluation server"

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!
cleanup() {
  kill "$tunnel_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

while ! curl -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    echo "$(timestamp) SSH tunnel exited before model readiness"
    exit 1
  fi
  sleep 60
done

echo "$(timestamp) model ready; starting fixed BIRD-dev 30 evaluation"
EVAL_ENABLE_THINKING=0 .venv/bin/python src/eval/rollout.py \
  --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
  --model "$SERVED_MODEL" \
  --tasks-json data/eval_inputs/bird_dev_20240627.jsonl \
  --indices-file data/eval_inputs/bird_dev_stratified30_seed20260717.indices.json \
  --n 30 --workers 1 --max-steps 30 --max-tokens 1024 --api-retries 2 \
  --context-mode rolling-legal-history --history-turns 4 \
  --rolling-prompt-variant full --rolling-observation-style resident \
  --result-dir "$RESULT_DIR"

echo "$(timestamp) evaluation complete"
cat "$RESULT_DIR/summary.json"
ssh "$REMOTE" 'info=/home/dengyan/tabular_rl_outputs/logs/bird_sft1_grounded_v4d_all651_serve.info; if test -f "$info"; then pid=$(sed -n "s/^pid=//p" "$info"); test -n "$pid" && kill "$pid" || true; fi'
