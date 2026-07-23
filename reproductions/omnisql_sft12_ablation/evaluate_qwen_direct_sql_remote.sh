#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
PORT=${2:-8026}
EXPERIMENT_ROOT=/home/dengyan/tabular_rl_experiments/omnisql_sft12_ablation
EVAL_ROOT="$EXPERIMENT_ROOT/current_eval_code"
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k
BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
ADAPTER="$OUTPUT_ROOT/checkpoints/qwen25_coder_7b"
PYTHON=/home/dengyan/miniconda3/envs/sft/bin/python
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
TASKS="$EXPERIMENT_ROOT/data/eval_inputs/bird_dev_20240627_remote.jsonl"
SERVED_MODEL=qwen25_coder_sft12_1k_direct_sql
RESULT_DIR="$OUTPUT_ROOT/results/direct_sql/qwen25_coder_sft_greedy_bird_set"
PID_FILE="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.pid"
SERVER_LOG="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.log"
mkdir -p "$RESULT_DIR" "$OUTPUT_ROOT/logs"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && ps -p "$server_pid" -o args= \
      | grep -q -- "--served-model-name $SERVED_MODEL"; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

test -f "$ADAPTER/adapter_config.json"
test -f "$TASKS"
test -f "$EVAL_ROOT/src/eval/text2sql_passk.py"
memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
if (( memory_used > 256 )); then
  echo "GPU $GPU_ID is not idle: ${memory_used} MiB used" >&2
  exit 5
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "$SERVED_MODEL=$ADAPTER" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "$server_pid" > "$PID_FILE"

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
    ready=1
    break
  fi
  sleep 3
done
if [[ "$ready" != 1 ]]; then
  tail -n 100 "$SERVER_LOG" >&2
  exit 6
fi

cd "$EVAL_ROOT"
EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$PYTHON" -u src/eval/text2sql_passk.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
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
