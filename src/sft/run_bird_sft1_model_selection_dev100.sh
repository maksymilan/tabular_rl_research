#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
REMOTE_PORT=${REMOTE_PORT:-8015}
LOCAL_PORT=${LOCAL_PORT:-18017}
REMOTE_OUTPUT=/home/dengyan/tabular_rl_outputs
BASE_MODEL=/home/dengyan/models/Qwen2.5-7B-Instruct
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
INDICES=data/eval_inputs/bird_dev_stratified100_disjoint_seed20260719.indices.json
LOG=${LOG:-/tmp/bird_sft1_model_selection_dev100.log}

exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
remote_pid_file="$REMOTE_OUTPUT/logs/bird_sft1_model_selection_dev100.vllm.pid"
tunnel_pid=""

stop_remote_model() {
  ssh "$REMOTE" "if test -f '$remote_pid_file'; then pid=\$(cat '$remote_pid_file'); if test -n \"\$pid\"; then kill \"\$pid\" 2>/dev/null || true; fi; fi"
  ssh "$REMOTE" "for i in \$(seq 1 60); do mem=\$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits); test \"\$mem\" -lt 1024 && exit 0; sleep 2; done; exit 1"
}

cleanup() {
  stop_remote_model >/dev/null 2>&1 || true
  if test -n "$tunnel_pid"; then
    kill "$tunnel_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

start_remote_model() {
  local served_model="$1"
  local adapter="$2"
  local remote_log="$REMOTE_OUTPUT/logs/${served_model}.vllm.log"
  ssh "$REMOTE" "CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 nohup '$VLLM_PY' \
    -m vllm.entrypoints.openai.api_server \
    --model '$BASE_MODEL' --served-model-name '$served_model' \
    --host 127.0.0.1 --port '$REMOTE_PORT' \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --enable-lora --max-lora-rank 16 \
    --lora-modules '$served_model=$adapter' \
    >'$remote_log' 2>&1 </dev/null & echo \$! >'$remote_pid_file'"
  for _ in $(seq 1 120); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" \
      | grep -q "$served_model"; then
      echo "$(timestamp) model ready: $served_model"
      return 0
    fi
    sleep 3
  done
  echo "$(timestamp) model readiness timeout: $served_model"
  return 1
}

run_eval() {
  local served_model="$1"
  local result_dir="$2"
  echo "$(timestamp) evaluating $served_model"
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost EVAL_ENABLE_THINKING=0 \
    .venv/bin/python -u src/eval/rollout.py \
    --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" --model "$served_model" \
    --tasks-json data/eval_inputs/bird_dev_20240627.jsonl \
    --indices-file "$INDICES" --n 100 --workers 2 \
    --max-steps 30 --max-tokens 1024 --api-retries 2 \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --result-dir "$result_dir"
}

echo "$(timestamp) starting disjoint dev-100 paired model selection"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!
sleep 2

old_name=qwen25_bird_grounded_r2_206_epoch1
old_adapter=$REMOTE_OUTPUT/checkpoints/qwen2.5-7b-bird-grounded-r2-train1303-eval130-6400-qlora/checkpoint-82
start_remote_model "$old_name" "$old_adapter"
run_eval "$old_name" data/results/qwen2.5_7b_bird_grounded_r2_206_epoch1_bird_dev_disjoint100
stop_remote_model

new_name=qwen25_bird_sft1_grounded_651_epoch1
new_adapter=$REMOTE_OUTPUT/checkpoints/qwen2.5-7b-bird-sft1-grounded-v4d-all651-6400-qlora/checkpoint-270
start_remote_model "$new_name" "$new_adapter"
run_eval "$new_name" data/results/qwen2.5_7b_bird_sft1_grounded_651_epoch1_bird_dev_disjoint100
stop_remote_model

echo "$(timestamp) paired dev-100 model selection complete"

