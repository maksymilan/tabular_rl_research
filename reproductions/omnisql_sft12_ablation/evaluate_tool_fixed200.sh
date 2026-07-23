#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=${1:?usage: evaluate_tool_fixed200.sh qwen25-coder|omnisql base|sft GPU REMOTE_PORT LOCAL_PORT}
VARIANT=${2:?usage: evaluate_tool_fixed200.sh qwen25-coder|omnisql base|sft GPU REMOTE_PORT LOCAL_PORT}
GPU_ID=${3:?usage: evaluate_tool_fixed200.sh qwen25-coder|omnisql base|sft GPU REMOTE_PORT LOCAL_PORT}
REMOTE_PORT=${4:?usage: evaluate_tool_fixed200.sh qwen25-coder|omnisql base|sft GPU REMOTE_PORT LOCAL_PORT}
LOCAL_PORT=${5:?usage: evaluate_tool_fixed200.sh qwen25-coder|omnisql base|sft GPU REMOTE_PORT LOCAL_PORT}

PROJECT_DIR=${PROJECT_DIR:-/private/tmp/tabular_rl_omnisql_sft12_ablation}
PYTHON=${PYTHON:-/Users/hudou/Research/tabular_rl_research/.venv/bin/python}
TASKS=${TASKS:-/private/tmp/bird_dev_stratified200_seed20260723.jsonl}
RESULT_ROOT=${RESULT_ROOT:-/Users/hudou/Research/tabular_rl_research/data/results/omnisql_sft12_1k_ablation}
REMOTE=${REMOTE:-table_rl}
REMOTE_OUTPUT=${REMOTE_OUTPUT:-/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k}
VLLM_PY=${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}

case "$MODEL_KEY" in
  qwen25-coder)
    BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
    ADAPTER="$REMOTE_OUTPUT/checkpoints/qwen25_coder_7b"
    ;;
  omnisql)
    BASE_MODEL=/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67
    ADAPTER="$REMOTE_OUTPUT/checkpoints/omnisql_7b"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac
case "$VARIANT" in
  base) ;;
  sft) ;;
  *)
    echo "variant must be base or sft" >&2
    exit 2
    ;;
esac

SERVED_MODEL="${MODEL_KEY//-/_}_${VARIANT}_tool_fixed200"
RESULT_DIR="$RESULT_ROOT/${MODEL_KEY}_${VARIANT}_tool_fixed200_strict_multiset"
REMOTE_PID_FILE="$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.pid"
REMOTE_LOG="$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.log"
LOG=${LOG:-/tmp/$SERVED_MODEL.log}
tunnel_pid=""

mkdir -p "$(dirname "$LOG")" "$RESULT_DIR"
exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
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

test -f "$TASKS"
if [[ "$VARIANT" == sft ]]; then
  ssh "$REMOTE" "test -f '$ADAPTER/adapter_config.json'"
fi
memory_used=$(ssh "$REMOTE" \
  "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i '$GPU_ID'")
if (( memory_used > 256 )); then
  echo "GPU $GPU_ID is not idle: ${memory_used} MiB used" >&2
  exit 5
fi

echo "$(timestamp) starting $MODEL_KEY $VARIANT fixed-200 historical-tool evaluation"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!

serve_args=(
  -m vllm.entrypoints.openai.api_server
  --model "$BASE_MODEL"
  --served-model-name "$SERVED_MODEL"
  --host 127.0.0.1
  --port "$REMOTE_PORT"
  --max-model-len 8192
  --gpu-memory-utilization 0.90
  --max-num-seqs 8
  --max-num-batched-tokens 8192
)
if [[ "$VARIANT" == sft ]]; then
  serve_args+=(
    --enable-lora
    --max-lora-rank 16
    --lora-modules "$SERVED_MODEL=$ADAPTER"
  )
fi
serve_command=$(printf " %q" "$VLLM_PY" "${serve_args[@]}")
ssh "$REMOTE" "mkdir -p '$REMOTE_OUTPUT/logs'; CUDA_VISIBLE_DEVICES='$GPU_ID' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup$serve_command \
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
  echo "vLLM endpoint did not become ready" >&2
  ssh "$REMOTE" "tail -n 100 '$REMOTE_LOG'" || true
  exit 6
fi

EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$PYTHON" -u src/eval/rollout_passk.py \
  --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
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

echo "$(timestamp) completed $MODEL_KEY $VARIANT fixed-200 historical-tool evaluation"
