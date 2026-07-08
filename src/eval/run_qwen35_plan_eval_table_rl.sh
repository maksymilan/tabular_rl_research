#!/usr/bin/env bash
# Run one Qwen3.5-9B LoRA tool rollout on table_rl.
#
# Each invocation owns one GPU, one vLLM server, and one result directory. The
# vLLM process is stopped automatically after rollout.py exits.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3.5-9B}
VLLM_PY=${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
SFT_PY=${SFT_PY:-/home/dengyan/miniconda3/envs/sft/bin/python}

GPU_ID=${GPU_ID:?set GPU_ID}
VLLM_PORT=${VLLM_PORT:?set VLLM_PORT}
LORA_DIR=${LORA_DIR:?set LORA_DIR}
SERVED_MODEL=${SERVED_MODEL:?set SERVED_MODEL}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
RUN_ID=${RUN_ID:-${SERVED_MODEL}_$(date +%Y%m%d_%H%M%S)}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-3}
MAX_STEPS=${MAX_STEPS:-20}
MAX_TOKENS=${MAX_TOKENS:-768}
MAX_CONSECUTIVE_ERRORS=${MAX_CONSECUTIVE_ERRORS:-3}
TABLE_OUTPUT_ROWS=${TABLE_OUTPUT_ROWS:-0}
N_DEV=${N_DEV:-1034}

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

LOG="$LOG_DIR/${RUN_ID}.log"
VLLM_LOG="$LOG_DIR/${RUN_ID}.vllm.log"
VLLM_PID=""

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && ps -p "$VLLM_PID" >/dev/null 2>&1; then
    echo "[$(date)] stopping vLLM pid=$VLLM_PID"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "[$(date)] run_id=$RUN_ID"
  echo "[$(date)] gpu=$GPU_ID port=$VLLM_PORT"
  echo "[$(date)] lora=$LORA_DIR"
  echo "[$(date)] result_dir=$RESULT_DIR"
  echo "[$(date)] max_model_len=$MAX_MODEL_LEN max_steps=$MAX_STEPS max_consecutive_errors=$MAX_CONSECUTIVE_ERRORS table_output_rows=$TABLE_OUTPUT_ROWS"

  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export HF_HUB_OFFLINE=1
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "$SERVED_MODEL=$LORA_DIR" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "[$(date)] started vLLM pid=$VLLM_PID log=$VLLM_LOG"

  for i in $(seq 1 240); do
    if curl -sS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[$(date)] vLLM ready"
      break
    fi
    if ! ps -p "$VLLM_PID" >/dev/null 2>&1; then
      echo "[$(date)] vLLM exited early"
      tail -120 "$VLLM_LOG" || true
      exit 1
    fi
    if [ "$i" -eq 240 ]; then
      echo "[$(date)] vLLM readiness timeout"
      tail -120 "$VLLM_LOG" || true
      exit 1
    fi
    sleep 5
  done

  export EVAL_ENABLE_THINKING=0
  "$SFT_PY" src/eval/rollout.py \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --model "$SERVED_MODEL" \
    --n "$N_DEV" \
    --workers "$ROLLOUT_WORKERS" \
    --max-steps "$MAX_STEPS" \
    --max-consecutive-errors "$MAX_CONSECUTIVE_ERRORS" \
    --table-output-rows "$TABLE_OUTPUT_ROWS" \
    --max-tokens "$MAX_TOKENS" \
    --api-retries 3 \
    --result-dir "$RESULT_DIR" \
    --resume

  echo "[$(date)] rollout finished"
  test -f "$RESULT_DIR/summary.json" && cat "$RESULT_DIR/summary.json"
} >> "$LOG" 2>&1
