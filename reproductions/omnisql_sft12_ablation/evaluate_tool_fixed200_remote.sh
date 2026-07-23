#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=${1:?usage: evaluate_tool_fixed200_remote.sh qwen25-coder|omnisql sft GPU PORT}
VARIANT=${2:?usage: evaluate_tool_fixed200_remote.sh qwen25-coder|omnisql sft GPU PORT}
GPU_ID=${3:?usage: evaluate_tool_fixed200_remote.sh qwen25-coder|omnisql sft GPU PORT}
PORT=${4:?usage: evaluate_tool_fixed200_remote.sh qwen25-coder|omnisql sft GPU PORT}
[[ "$VARIANT" == sft ]] || { echo "remote overnight runner expects sft" >&2; exit 2; }

PROJECT_DIR=/home/dengyan/tabular_rl_experiments/omnisql_sft12_ablation
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k
PYTHON=/home/dengyan/miniconda3/envs/sft/bin/python
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
TASKS="$PROJECT_DIR/data/eval_inputs/bird_dev_stratified200_seed20260723.jsonl"

case "$MODEL_KEY" in
  qwen25-coder)
    BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
    ADAPTER="$OUTPUT_ROOT/checkpoints/qwen25_coder_7b"
    ;;
  omnisql)
    BASE_MODEL=/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67
    ADAPTER="$OUTPUT_ROOT/checkpoints/omnisql_7b"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

SERVED_MODEL="${MODEL_KEY//-/_}_sft_tool_fixed200"
RESULT_DIR="$OUTPUT_ROOT/results/tool/${MODEL_KEY}_sft_fixed200_tool_output_only_strict_multiset"
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

test -f "$TASKS"
test -f "$ADAPTER/adapter_config.json"
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

cd "$PROJECT_DIR"
EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$PYTHON" -u src/eval/rollout_passk.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$SERVED_MODEL" \
  --examples-json "$TASKS" \
  --allow-eval-tasks \
  --n 200 \
  --n-samples 1 \
  --pass-k 1 \
  --workers 2 \
  --sample-workers 1 \
  --max-inflight-requests 2 \
  --max-steps 30 \
  --max-tokens 1024 \
  --temperature 0 \
  --top-p 1 \
  --sample-detail full \
  --summary-every 10 \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --rolling-prompt-variant full \
  --rolling-observation-style resident \
  --result-dir "$RESULT_DIR" \
  --resume
