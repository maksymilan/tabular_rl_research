#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
REMOTE_PORT=${REMOTE_PORT:-8017}
LOCAL_PORT=${LOCAL_PORT:-18020}
SERVED_MODEL=qwen25_bird_sft1_grounded_651_epoch1_sft2_scale
REMOTE_OUTPUT=/home/dengyan/tabular_rl_outputs
BASE_MODEL=/home/dengyan/models/Qwen2.5-7B-Instruct
ADAPTER=$REMOTE_OUTPUT/checkpoints/qwen2.5-7b-bird-sft1-grounded-v4d-all651-6400-qlora/checkpoint-270
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
REMOTE_PID_FILE=$REMOTE_OUTPUT/logs/bird_sft2_remaining840.vllm.pid
LOG=${LOG:-/tmp/bird_sft2_student_remaining478_retry1.log}

exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
tunnel_pid=""

stop_remote_model() {
  ssh "$REMOTE" "if test -f '$REMOTE_PID_FILE'; then pid=\$(cat '$REMOTE_PID_FILE'); test -n \"\$pid\" && kill \"\$pid\" 2>/dev/null || true; fi"
}
cleanup() {
  stop_remote_model >/dev/null 2>&1 || true
  test -n "$tunnel_pid" && kill "$tunnel_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "$(timestamp) starting remaining-478 transport retry"
stop_remote_model
sleep 3
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!

ssh "$REMOTE" "CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 nohup '$VLLM_PY' \
  -m vllm.entrypoints.openai.api_server \
  --model '$BASE_MODEL' --served-model-name '$SERVED_MODEL' \
  --host 127.0.0.1 --port '$REMOTE_PORT' \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules '$SERVED_MODEL=$ADAPTER' \
  >'$REMOTE_OUTPUT/logs/$SERVED_MODEL.retry1.vllm.log' 2>&1 </dev/null & echo \$! >'$REMOTE_PID_FILE'"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
    echo "$(timestamp) model ready"
    break
  fi
  sleep 3
done
curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"

run_bucket_guarded() {
  local bucket="$1"
  local input="$2"
  local count="$3"
  local result_dir="$4"
  echo "$(timestamp) starting $bucket retry count=$count K=4"
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost EVAL_ENABLE_THINKING=0 \
    .venv/bin/python -u src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" --model "$SERVED_MODEL" \
    --examples-json "$input" --n "$count" --n-samples 4 --pass-k 1,2,4 \
    --workers 2 --sample-workers 4 --max-inflight-requests 8 \
    --max-steps 30 --max-tokens 1024 --temperature 0.7 --top-p 0.95 \
    --sample-detail full --summary-every 10 \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --result-dir "$result_dir" &
  local rollout_pid=$!
  local failures=0
  while kill -0 "$rollout_pid" 2>/dev/null; do
    sleep 15
    if ! kill -0 "$tunnel_pid" 2>/dev/null; then
      echo "$(timestamp) tunnel exited; terminating rollout"
      kill "$rollout_pid" 2>/dev/null || true
      wait "$rollout_pid" 2>/dev/null || true
      return 75
    fi
    if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
      failures=0
    else
      failures=$((failures + 1))
      if (( failures >= 3 )); then
        echo "$(timestamp) endpoint failed three health checks; terminating rollout"
        kill "$rollout_pid" 2>/dev/null || true
        wait "$rollout_pid" 2>/dev/null || true
        return 75
      fi
    fi
  done
  wait "$rollout_pid"
}

run_bucket_guarded medium \
  data/eval_inputs/bird_train_sft2_student_remaining_medium_transport_retry1.jsonl 208 \
  data/results/bird_sft2_student_epoch1_passk_medium_remaining270_k4_transport_retry1
run_bucket_guarded hard \
  data/eval_inputs/bird_train_sft2_student_remaining_hard.jsonl 270 \
  data/results/bird_sft2_student_epoch1_passk_hard_remaining270_k4
echo "$(timestamp) remaining-478 transport retry complete"
