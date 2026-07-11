#!/usr/bin/env bash
# Run closed-loop tool-use pass@k against an arbitrary base model + LoRA on one table_rl GPU.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
LORA_DIR=${LORA_DIR:?set LORA_DIR}
GPU_ID=${GPU_ID:?set GPU_ID}
VLLM_PORT=${VLLM_PORT:?set VLLM_PORT}
SERVED_MODEL=${SERVED_MODEL:?set SERVED_MODEL}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
EXAMPLES_JSON=${EXAMPLES_JSON:?set EXAMPLES_JSON}

VLLM_PY=${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
SFT_PY=${SFT_PY:-/home/dengyan/miniconda3/envs/sft/bin/python}
RUN_ID=${RUN_ID:-${SERVED_MODEL}_passk_$(date +%Y%m%d_%H%M%S)}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
N_EXAMPLES=${N_EXAMPLES:-0}
N_SAMPLES=${N_SAMPLES:-16}
PASS_K=${PASS_K:-1,2,4,8,16}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-1}
SAMPLE_WORKERS=${SAMPLE_WORKERS:-4}
STOP_ON_SUCCESS=${STOP_ON_SUCCESS:-1}
MAX_STEPS=${MAX_STEPS:-20}
MAX_TOKENS=${MAX_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
LOG="$LOG_DIR/${RUN_ID}.log"
VLLM_LOG="$LOG_DIR/${RUN_ID}.vllm.log"
VLLM_PID=""

cleanup() {
  if [ -n "$VLLM_PID" ] && ps -p "$VLLM_PID" >/dev/null 2>&1; then
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "[$(date)] run_id=$RUN_ID gpu=$GPU_ID n_examples=$N_EXAMPLES n_samples=$N_SAMPLES pass_k=$PASS_K"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export HF_HUB_OFFLINE=1
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager --enable-lora --max-lora-rank 16 \
    --lora-modules "$SERVED_MODEL=$LORA_DIR" >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  ready=0
  for i in $(seq 1 240); do
    if curl -sS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! ps -p "$VLLM_PID" >/dev/null 2>&1; then
      tail -120 "$VLLM_LOG" || true
      exit 1
    fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "vLLM readiness timeout" >&2
    tail -120 "$VLLM_LOG" || true
    exit 1
  fi

  eval_args=(--examples-json "$EXAMPLES_JSON" --n-samples "$N_SAMPLES" --pass-k "$PASS_K")
  if [ "$STOP_ON_SUCCESS" = "1" ]; then
    eval_args+=(--stop-on-success)
  fi
  if [ "$N_EXAMPLES" -gt 0 ]; then
    eval_args+=(--n "$N_EXAMPLES")
  fi
  EVAL_ENABLE_THINKING=0 "$SFT_PY" src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" --model "$SERVED_MODEL" \
    --workers "$ROLLOUT_WORKERS" --sample-workers "$SAMPLE_WORKERS" \
    --max-steps "$MAX_STEPS" --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --api-retries 3 \
    --result-dir "$RESULT_DIR" --resume "${eval_args[@]}"

  test -f "$RESULT_DIR/summary.json" && cat "$RESULT_DIR/summary.json"
} >>"$LOG" 2>&1
