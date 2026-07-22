#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
REMOTE_PORT=${REMOTE_PORT:-8016}
LOCAL_PORT=${LOCAL_PORT:-18019}
SERVED_MODEL=qwen25_bird_sft1_grounded_651_epoch1_sft2
REMOTE_OUTPUT=/home/dengyan/tabular_rl_outputs
BASE_MODEL=/home/dengyan/models/Qwen2.5-7B-Instruct
ADAPTER=$REMOTE_OUTPUT/checkpoints/qwen2.5-7b-bird-sft1-grounded-v4d-all651-6400-qlora/checkpoint-270
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
REMOTE_PID_FILE=$REMOTE_OUTPUT/logs/bird_sft2_medium_hard60.vllm.pid
LOG=${LOG:-/tmp/bird_sft2_student_medium_hard60.log}

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

echo "$(timestamp) starting medium/hard SFT-2 student rollout supplement"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!

ssh "$REMOTE" "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 nohup '$VLLM_PY' \
  -m vllm.entrypoints.openai.api_server \
  --model '$BASE_MODEL' --served-model-name '$SERVED_MODEL' \
  --host 127.0.0.1 --port '$REMOTE_PORT' \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules '$SERVED_MODEL=$ADAPTER' \
  >'$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.log' 2>&1 </dev/null & echo \$! >'$REMOTE_PID_FILE'"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
    echo "$(timestamp) model ready"
    break
  fi
  sleep 3
done
curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"

run_bucket() {
  local bucket="$1"
  echo "$(timestamp) starting $bucket-30 K=4"
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost EVAL_ENABLE_THINKING=0 \
    .venv/bin/python -u src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" --model "$SERVED_MODEL" \
    --examples-json "data/eval_inputs/bird_train_sft2_student_pilot1000_buckets/$bucket.jsonl" \
    --n 30 --n-samples 4 --pass-k 1,2,4 \
    --workers 2 --sample-workers 4 --max-inflight-requests 8 \
    --max-steps 30 --max-tokens 1024 --temperature 0.7 --top-p 0.95 \
    --sample-detail full --summary-every 5 \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --result-dir "data/results/bird_sft2_student_epoch1_passk_${bucket}30_k4"
}

run_bucket medium
run_bucket hard
echo "$(timestamp) medium/hard SFT-2 student rollout supplement complete"

