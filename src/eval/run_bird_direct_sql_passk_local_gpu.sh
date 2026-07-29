#!/usr/bin/env bash
set -euo pipefail

# Run direct-SQL pass@K on the same Linux host as vLLM.

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  test -n "${!name:-}" || die "required environment variable is unset: ${name}"
}

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
TASKS_JSON="${TASKS_JSON:-data/eval_inputs/bird_dev_20240627.jsonl}"
N="${N:-1534}"
N_SAMPLES="${N_SAMPLES:-8}"
PASS_K="${PASS_K:-1,2,4,8}"
WORKERS="${WORKERS:-3}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
PROMPT_PROFILE="${PROMPT_PROFILE:-canonical-json-v1}"
SCHEMA_VALUE_COUNT="${SCHEMA_VALUE_COUNT:-2}"
SCHEMA_METADATA_JSON="${SCHEMA_METADATA_JSON:-}"
EXECUTION_TIMEOUT_SECONDS="${EXECUTION_TIMEOUT_SECONDS:-20}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

for name in BASE_MODEL SERVED_MODEL GPU_ID PORT RESULT_DIR VLLM_LOG VLLM_PID_FILE EVAL_LOG; do
  require_env "$name"
done
cd "$PROJECT_DIR"
test -x "$VLLM_PYTHON" || die "vLLM Python not executable: $VLLM_PYTHON"
test -x "$PYTHON_BIN" || die "evaluation Python not executable: $PYTHON_BIN"
test -d "$BASE_MODEL" || die "base model not found: $BASE_MODEL"
test -f "$TASKS_JSON" || die "tasks file not found: $TASKS_JSON"
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

CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 "$VLLM_PYTHON" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL" \
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

deadline=$((SECONDS + 600))
while test "$SECONDS" -lt "$deadline"; do
  if ! kill -0 "$owned_pid" 2>/dev/null; then
    die "vLLM exited before becoming ready; see $VLLM_LOG"
  fi
  if curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null |
    grep -q "$SERVED_MODEL"; then
    break
  fi
  sleep 3
done
curl --noproxy '*' -fsS --max-time 5 \
  "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null |
  grep -q "$SERVED_MODEL" ||
  die "timed out waiting for vLLM readiness"

PROMPT_ARGS=(
  --prompt-profile "$PROMPT_PROFILE"
  --schema-value-count "$SCHEMA_VALUE_COUNT"
)
if test -n "$SCHEMA_METADATA_JSON"; then
  PROMPT_ARGS+=(--schema-metadata-json "$SCHEMA_METADATA_JSON")
fi

EVAL_ENABLE_THINKING=0 \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
  "$PYTHON_BIN" -u src/eval/text2sql_passk.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_MODEL" \
    --tasks-json "$TASKS_JSON" \
    --n "$N" \
    --n-samples "$N_SAMPLES" \
    --pass-k "$PASS_K" \
    --workers "$WORKERS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --api-retries 3 \
    --execution-timeout-seconds "$EXECUTION_TIMEOUT_SECONDS" \
    --denotation-comparison bird-set \
    "${PROMPT_ARGS[@]}" \
    --result-dir "$RESULT_DIR" \
    --resume \
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

printf 'Direct-SQL evaluation complete: model=%s result=%s\n' \
  "$SERVED_MODEL" "$RESULT_DIR"
