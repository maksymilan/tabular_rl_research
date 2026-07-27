#!/usr/bin/env bash
set -euo pipefail

# Run one frozen greedy feedback gate on a local GPU/vLLM pair. Each stage uses a
# distinct result directory; an interrupted stage resumes only its own exact cohort.

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  test -n "${!name:-}" || die "required environment variable is unset: ${name}"
}

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-$PYTHON_BIN}"
TASKS_JSON="${TASKS_JSON:-data/eval_inputs/bird_dev_20240627.jsonl}"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-18039}"
MAX_INFLIGHT="${MAX_INFLIGHT:-24}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
HISTORY_TURNS="${HISTORY_TURNS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"

for name in BASE_MODEL ADAPTER SERVED_MODEL INDICES_FILE RESULT_DIR; do
  require_env "$name"
done

cd "$PROJECT_DIR"
test -x "$PYTHON_BIN" || die "Python is not executable: $PYTHON_BIN"
test -x "$VLLM_PYTHON" || die "vLLM Python is not executable: $VLLM_PYTHON"
test -f "$TASKS_JSON" || die "task file does not exist: $TASKS_JSON"
test -f "$INDICES_FILE" || die "indices file does not exist: $INDICES_FILE"
test -f "$ADAPTER/adapter_config.json" || die "adapter_config.json is missing"
test -f "$ADAPTER/adapter_model.safetensors" || die "adapter weights are missing"
if ss -ltnH "sport = :$PORT" | grep -q .; then
  die "port already in use: $PORT"
fi

mkdir -p "$RESULT_DIR" "$OUTPUT_ROOT/logs"
VLLM_LOG="${VLLM_LOG:-$OUTPUT_ROOT/logs/${SERVED_MODEL}.vllm.log}"
EVAL_LOG="${EVAL_LOG:-$OUTPUT_ROOT/logs/${SERVED_MODEL}.eval.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-$OUTPUT_ROOT/logs/${SERVED_MODEL}.vllm.pid}"

vllm_pid=""
eval_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if test -n "$eval_pid" && kill -0 "$eval_pid" 2>/dev/null; then
    kill "$eval_pid" 2>/dev/null || true
    wait "$eval_pid" 2>/dev/null || true
  fi
  if test -n "$vllm_pid" && kill -0 "$vllm_pid" 2>/dev/null; then
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  if test -r "$VLLM_PID_FILE" &&
     test "$(tr -d '[:space:]' <"$VLLM_PID_FILE")" = "$vllm_pid"; then
    rm -f "$VLLM_PID_FILE"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 "$VLLM_PYTHON" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --enable-lora \
  --lora-modules "$SERVED_MODEL=$ADAPTER" \
  --max-lora-rank 16 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_INFLIGHT" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --generation-config vllm \
  >"$VLLM_LOG" 2>&1 &
vllm_pid=$!
printf '%s\n' "$vllm_pid" >"$VLLM_PID_FILE"

for _ in $(seq 1 200); do
  kill -0 "$vllm_pid" 2>/dev/null ||
    die "vLLM exited before readiness; see $VLLM_LOG"
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
  die "vLLM readiness timed out"

EVAL_ENABLE_THINKING=0 \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
  "$PYTHON_BIN" -u src/eval/rollout.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_MODEL" \
    --tasks-json "$TASKS_JSON" \
    --indices-file "$INDICES_FILE" \
    --n 1534 \
    --workers "$MAX_INFLIGHT" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
    --context-mode rolling-legal-history \
    --history-turns "$HISTORY_TURNS" \
    --rolling-prompt-variant full \
    --rolling-observation-style resident \
    --denotation-comparison bird-set \
    --result-dir "$RESULT_DIR" \
    --resume \
    >"$EVAL_LOG" 2>&1 &
eval_pid=$!
set +e
wait "$eval_pid"
status=$?
set -e
eval_pid=""
test "$status" -eq 0 ||
  die "evaluation exited with status $status; see $EVAL_LOG"
