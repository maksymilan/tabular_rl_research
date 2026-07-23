#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:-sft}
GPU_ID=${2:-0}
REMOTE_PORT=${3:-8023}
LOCAL_PORT=${4:-18033}
if [[ "$VARIANT" != sft ]]; then
  echo "only the post-SFT variant is run here; the frozen base artifact already exists" >&2
  exit 2
fi

EVAL_PROJECT_DIR=${EVAL_PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
ADAPTER=/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/checkpoints/qwen25_coder_7b
REMOTE_OUTPUT=/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
SERVED_MODEL=qwen25_coder_sft12_1k_direct_sql
REMOTE_PID_FILE="$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.pid"
REMOTE_LOG="$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.log"
RESULT_DIR="$EVAL_PROJECT_DIR/data/results/omnisql_sft12_1k_ablation/qwen25_coder_sft_direct_sql_greedy_bird_set"
TASKS="$EVAL_PROJECT_DIR/data/eval_inputs/bird_dev_20240627.jsonl"
LOG=${LOG:-/tmp/$SERVED_MODEL.log}
tunnel_pid=""

mkdir -p "$RESULT_DIR"
exec >>"$LOG" 2>&1
cd "$EVAL_PROJECT_DIR"

stop_remote_model() {
  ssh "$REMOTE" "if test -f '$REMOTE_PID_FILE'; then
    pid=\$(cat '$REMOTE_PID_FILE')
    if test -n \"\$pid\" && ps -p \"\$pid\" -o args= | grep -q -- '--served-model-name $SERVED_MODEL'; then
      kill \"\$pid\" 2>/dev/null || true
    fi
  fi"
}
cleanup() {
  stop_remote_model >/dev/null 2>&1 || true
  test -n "$tunnel_pid" && kill "$tunnel_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ssh "$REMOTE" "test -f '$ADAPTER/adapter_config.json'"
memory_used=$(ssh "$REMOTE" \
  "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i '$GPU_ID'")
if (( memory_used > 256 )); then
  echo "GPU $GPU_ID is not idle: ${memory_used} MiB used" >&2
  exit 5
fi

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!
ssh "$REMOTE" "mkdir -p '$REMOTE_OUTPUT/logs'; CUDA_VISIBLE_DEVICES='$GPU_ID' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup '$VLLM_PY' \
  -m vllm.entrypoints.openai.api_server \
  --model '$BASE_MODEL' --served-model-name '$SERVED_MODEL' \
  --host 127.0.0.1 --port '$REMOTE_PORT' \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules '$SERVED_MODEL=$ADAPTER' \
  >'$REMOTE_LOG' 2>&1 </dev/null & echo \$! >'$REMOTE_PID_FILE'"

ready=0
for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" \
      | grep -q "$SERVED_MODEL"; then
    ready=1
    break
  fi
  sleep 3
done
if [[ "$ready" != 1 ]]; then
  ssh "$REMOTE" "tail -n 100 '$REMOTE_LOG'" || true
  exit 6
fi

EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  .venv/bin/python -u src/eval/text2sql_passk.py \
  --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
  --model "$SERVED_MODEL" \
  --tasks-json "$TASKS" \
  --n 1534 \
  --n-samples 1 \
  --pass-k 1 \
  --workers 4 \
  --max-tokens 1024 \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty 1.05 \
  --api-retries 3 \
  --execution-timeout-seconds 20 \
  --denotation-comparison bird-set \
  --result-dir "$RESULT_DIR" \
  --resume
