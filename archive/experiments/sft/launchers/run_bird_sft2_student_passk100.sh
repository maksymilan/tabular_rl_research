#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
REMOTE_PORT=${REMOTE_PORT:-8015}
LOCAL_PORT=${LOCAL_PORT:-18016}
SERVED_MODEL=${SERVED_MODEL:-qwen25_bird_sft1_grounded_all651_epoch1}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/data/results/bird_sft2_student_epoch1_passk100_k4}
LOG=${LOG:-/tmp/bird_sft2_student_epoch1_passk100_k4.log}

exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(timestamp) starting SFT-2 student pass@k pilot"

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!
cleanup() {
  kill "$tunnel_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    echo "$(timestamp) SSH tunnel exited before model readiness"
    exit 1
  fi
  sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost EVAL_ENABLE_THINKING=0 \
  .venv/bin/python -u src/eval/rollout_passk.py \
  --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
  --model "$SERVED_MODEL" \
  --examples-json data/eval_inputs/bird_train_sft2_student_pilot1000.jsonl \
  --n 100 --n-samples 4 --pass-k 1,2,4 \
  --workers 2 --sample-workers 4 --max-inflight-requests 8 \
  --max-steps 30 --max-tokens 1024 --temperature 0.7 --top-p 0.95 \
  --sample-detail full --summary-every 5 \
  --context-mode rolling-legal-history --history-turns 4 \
  --rolling-prompt-variant full --rolling-observation-style resident \
  --result-dir "$RESULT_DIR"

echo "$(timestamp) SFT-2 student pass@k pilot complete"

