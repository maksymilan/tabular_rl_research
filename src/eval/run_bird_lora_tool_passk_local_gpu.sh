#!/usr/bin/env bash
set -euo pipefail

# Run a closed-loop BIRD tool-use evaluation on the same Linux host as vLLM.
# This is the server-native counterpart of run_bird_lora_tool_passk_table_rl.sh
# and is suitable for unattended tmux queues without an SSH tunnel.

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  test -n "${!name:-}" || die "required environment variable is unset: ${name}"
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
EXAMPLES_JSON="${EXAMPLES_JSON:-data/eval_inputs/bird_dev_20240627.jsonl}"
INDICES_FILE="${INDICES_FILE:-}"
N="${N:-1534}"
N_SAMPLES="${N_SAMPLES:-4}"
PASS_K="${PASS_K:-1,2,4}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
WORKERS="${WORKERS:-2}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$MAX_INFLIGHT}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HISTORY_TURNS="${HISTORY_TURNS:-4}"
MODEL_READY_TIMEOUT_SECONDS="${MODEL_READY_TIMEOUT_SECONDS:-600}"
MODEL_READY_POLL_SECONDS="${MODEL_READY_POLL_SECONDS:-3}"
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-0}"
SERVER_CONFIG_ID="${SERVER_CONFIG_ID:-vllm-generation-config-vllm}"
MODEL_MODE="${MODEL_MODE:-adapter}"
ALLOW_OPERATIONAL_CONCURRENCY_RESUME="${ALLOW_OPERATIONAL_CONCURRENCY_RESUME:-0}"
TOOL_EXECUTION_TIMEOUT_SECONDS="${TOOL_EXECUTION_TIMEOUT_SECONDS:-20}"
ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME="${ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME:-0}"

for name in BASE_MODEL SERVED_MODEL GPU_ID PORT RESULT_DIR VLLM_LOG VLLM_PID_FILE EVAL_LOG; do
  require_env "$name"
done
case "$MODEL_MODE" in
  adapter) require_env ADAPTER ;;
  standalone) ADAPTER="" ;;
  *) die "MODEL_MODE must be adapter or standalone" ;;
esac
for name in GPU_ID PORT N N_SAMPLES MAX_STEPS MAX_TOKENS WORKERS SAMPLE_WORKERS \
  MAX_INFLIGHT MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN HISTORY_TURNS \
  MODEL_READY_TIMEOUT_SECONDS MODEL_READY_POLL_SECONDS; do
  value="${!name}"
  is_uint "$value" || die "${name} must be a non-negative integer, got: ${value}"
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] ||
  die "GPU_MEMORY_UTILIZATION must be a number in [0,1], got: ${GPU_MEMORY_UTILIZATION}"
[[ "$TOOL_EXECUTION_TIMEOUT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "TOOL_EXECUTION_TIMEOUT_SECONDS must be non-negative, got: ${TOOL_EXECUTION_TIMEOUT_SECONDS}"
case "$ALLOW_OPERATIONAL_CONCURRENCY_RESUME" in
  0 | 1) ;;
  *) die "ALLOW_OPERATIONAL_CONCURRENCY_RESUME must be 0 or 1" ;;
esac
case "$ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME" in
  0 | 1) ;;
  *) die "ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME must be 0 or 1" ;;
esac

cd "$PROJECT_DIR"
test -x "$VLLM_PYTHON" || die "vLLM Python not executable: $VLLM_PYTHON"
test -x "$PYTHON_BIN" || die "evaluation Python not executable: $PYTHON_BIN"
test -d "$BASE_MODEL" || die "base model not found: $BASE_MODEL"
test -f "$EXAMPLES_JSON" || die "examples file not found: $EXAMPLES_JSON"
if test -n "$INDICES_FILE"; then
  test -f "$INDICES_FILE" || die "indices file not found: $INDICES_FILE"
fi
if test "$MODEL_MODE" = adapter; then
  test -f "$ADAPTER/adapter_config.json" || die "adapter config not found: $ADAPTER"
  test -f "$ADAPTER/adapter_model.safetensors" || die "adapter weights not found: $ADAPTER"
fi
mkdir -p "$(dirname "$VLLM_LOG")" "$(dirname "$VLLM_PID_FILE")" \
  "$(dirname "$EVAL_LOG")" "$RESULT_DIR"

owned_pid=""
eval_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if test -n "$eval_pid" && kill -0 "$eval_pid" 2>/dev/null; then
    kill "$eval_pid" 2>/dev/null || true
    wait "$eval_pid" 2>/dev/null || true
  fi
  if test -n "$owned_pid" && kill -0 "$owned_pid" 2>/dev/null; then
    kill "$owned_pid" 2>/dev/null || true
    wait "$owned_pid" 2>/dev/null || true
  fi
  if test -r "$VLLM_PID_FILE" &&
     test "$(tr -d '[:space:]' < "$VLLM_PID_FILE")" = "$owned_pid"; then
    rm -f "$VLLM_PID_FILE"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if test -r "$VLLM_PID_FILE"; then
  existing_pid="$(tr -d '[:space:]' < "$VLLM_PID_FILE")"
  if test -n "$existing_pid" && kill -0 "$existing_pid" 2>/dev/null; then
    die "pid file already refers to a live process: $existing_pid"
  fi
  rm -f "$VLLM_PID_FILE"
fi
if ss -ltnH "sport = :$PORT" | grep -q .; then
  die "port already in use: $PORT"
fi

lora_flags=()
if test "$MODEL_MODE" = adapter; then
  lora_flags=(
    --enable-lora
    --lora-modules "$SERVED_MODEL=$ADAPTER"
    --max-lora-rank 16
  )
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 "$VLLM_PYTHON" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  "${lora_flags[@]}" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --generation-config vllm \
  >"$VLLM_LOG" 2>&1 &
owned_pid=$!
printf '%s\n' "$owned_pid" >"$VLLM_PID_FILE"

deadline=$((SECONDS + MODEL_READY_TIMEOUT_SECONDS))
while test "$SECONDS" -lt "$deadline"; do
  if ! kill -0 "$owned_pid" 2>/dev/null; then
    die "vLLM exited before becoming ready; see $VLLM_LOG"
  fi
  if curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null |
    grep -q "$SERVED_MODEL"; then
    break
  fi
  sleep "$MODEL_READY_POLL_SECONDS"
done
curl --noproxy '*' -fsS --max-time 5 \
  "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null |
  grep -q "$SERVED_MODEL" ||
  die "timed out waiting for vLLM readiness"

operational_resume_args=()
if test "$ALLOW_OPERATIONAL_CONCURRENCY_RESUME" -eq 1; then
  operational_resume_args+=(--allow-operational-concurrency-resume)
fi
if test "$ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME" -eq 1; then
  operational_resume_args+=(--allow-operational-tool-timeout-resume)
fi
selection_args=()
if test -n "$INDICES_FILE"; then
  selection_args+=(--indices-file "$INDICES_FILE")
fi

printf 'Starting evaluation model=%s gpu=%s concurrency=%s result=%s\n' \
  "$SERVED_MODEL" "$GPU_ID" "$MAX_INFLIGHT" "$RESULT_DIR"
EVAL_ENABLE_THINKING="$EVAL_ENABLE_THINKING" \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
  "$PYTHON_BIN" -u src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_MODEL" \
    --examples-json "$EXAMPLES_JSON" \
    "${selection_args[@]}" \
    --allow-eval-tasks \
    --n "$N" \
    --n-samples "$N_SAMPLES" \
    --pass-k "$PASS_K" \
    --workers "$WORKERS" \
    --sample-workers "$SAMPLE_WORKERS" \
    --max-inflight-requests "$MAX_INFLIGHT" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
    --tool-execution-timeout-seconds "$TOOL_EXECUTION_TIMEOUT_SECONDS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --server-config-id "$SERVER_CONFIG_ID" \
    --sample-detail full \
    --summary-every 10 \
    --context-mode rolling-legal-history \
    --history-turns "$HISTORY_TURNS" \
    --rolling-prompt-variant full \
    --rolling-observation-style resident \
    --denotation-comparison bird-set \
    --result-dir "$RESULT_DIR" \
    --resume \
    "${operational_resume_args[@]}" \
    >"$EVAL_LOG" 2>&1 &
eval_pid=$!
set +e
wait "$eval_pid"
eval_status=$?
set -e
eval_pid=""
if test "$eval_status" -ne 0; then
  printf 'Evaluation failed with status %s; see %s\n' "$eval_status" "$EVAL_LOG" >&2
  exit "$eval_status"
fi

printf 'Evaluation complete: model=%s result=%s\n' "$SERVED_MODEL" "$RESULT_DIR"
