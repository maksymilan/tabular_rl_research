#!/usr/bin/env bash
set -euo pipefail

# Run one adapter-backed or standalone tool-use model on table_rl with a local
# closed-loop BIRD pass@K client. Experiment-specific values are supplied by
# the caller.

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

REMOTE="${REMOTE:-table_rl}"
PROJECT_DIR="${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
REMOTE_VLLM_PYTHON="${REMOTE_VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
EXAMPLES_JSON="${EXAMPLES_JSON:-data/eval_inputs/bird_dev_20240627.jsonl}"
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
EXISTING_SERVICE_POLICY="${EXISTING_SERVICE_POLICY:-reject}"
SERVER_CONFIG_ID="${SERVER_CONFIG_ID:-vllm-generation-config-vllm}"
MODEL_MODE="${MODEL_MODE:-adapter}"
ALLOW_OPERATIONAL_CONCURRENCY_RESUME="${ALLOW_OPERATIONAL_CONCURRENCY_RESUME:-0}"

for name in BASE_MODEL SERVED_MODEL GPU_ID REMOTE_PORT LOCAL_PORT RESULT_DIR; do
  require_env "$name"
done
case "$MODEL_MODE" in
  adapter) require_env ADAPTER ;;
  standalone) ADAPTER="" ;;
  *) die "MODEL_MODE must be adapter or standalone" ;;
esac

for name in GPU_ID REMOTE_PORT LOCAL_PORT N N_SAMPLES MAX_STEPS MAX_TOKENS \
  WORKERS SAMPLE_WORKERS MAX_INFLIGHT HISTORY_TURNS MODEL_READY_TIMEOUT_SECONDS \
  MODEL_READY_POLL_SECONDS MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN; do
  value="${!name}"
  is_uint "$value" || die "${name} must be a non-negative integer, got: ${value}"
done

[[ "$SERVED_MODEL" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "SERVED_MODEL may contain only letters, digits, dot, underscore, and hyphen"
case "$EXISTING_SERVICE_POLICY" in
  reject | reuse | adopt) ;;
  *) die "EXISTING_SERVICE_POLICY must be reject, reuse, or adopt" ;;
esac
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] ||
  die "GPU_MEMORY_UTILIZATION must be a number in [0,1], got: ${GPU_MEMORY_UTILIZATION}"
case "$ALLOW_OPERATIONAL_CONCURRENCY_RESUME" in
  0 | 1) ;;
  *) die "ALLOW_OPERATIONAL_CONCURRENCY_RESUME must be 0 or 1" ;;
esac

for value in "$BASE_MODEL" "$ADAPTER" "$REMOTE_OUTPUT_ROOT" "$REMOTE_VLLM_PYTHON"; do
  [[ "$value" != *"'"* ]] || die "remote paths may not contain a single quote: ${value}"
done

cd "$PROJECT_DIR"
test -f "$EXAMPLES_JSON" || die "examples file not found: ${EXAMPLES_JSON}"

VLLM_LOG="${VLLM_LOG:-${REMOTE_OUTPUT_ROOT}/logs/${SERVED_MODEL}.vllm.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${REMOTE_OUTPUT_ROOT}/logs/${SERVED_MODEL}.vllm.pid}"
LOCAL_EVAL_LOG="${LOCAL_EVAL_LOG:-/tmp/${SERVED_MODEL}.eval.log}"
LOCAL_TUNNEL_LOG="${LOCAL_TUNNEL_LOG:-/tmp/${SERVED_MODEL}.tunnel.log}"

for value in "$VLLM_LOG" "$VLLM_PID_FILE"; do
  [[ "$value" != *"'"* ]] || die "remote paths may not contain a single quote: ${value}"
done

remote_pid=""
tunnel_pid=""
eval_pid=""
owns_remote_service=0

stop_owned_remote_service() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh "$REMOTE" \
      "if test -r '$VLLM_PID_FILE' && test \"\$(tr -d '[:space:]' < '$VLLM_PID_FILE')\" = '$remote_pid'; then kill '$remote_pid' 2>/dev/null || true; rm -f '$VLLM_PID_FILE'; fi" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  printf 'WARNING: could not stop owned remote vLLM pid=%s after five SSH attempts\n' \
    "$remote_pid" >&2
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if test -n "$eval_pid" && kill -0 "$eval_pid" 2>/dev/null; then
    kill "$eval_pid" 2>/dev/null || true
    wait "$eval_pid" 2>/dev/null || true
  fi
  if test -n "$tunnel_pid" && kill -0 "$tunnel_pid" 2>/dev/null; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
  if test -n "$remote_pid" && test "$owns_remote_service" -eq 1; then
    stop_owned_remote_service || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

service_state="$(
  ssh "$REMOTE" "
    set -eu
    test -x '$REMOTE_VLLM_PYTHON'
    test -d '$BASE_MODEL'
    if test '$MODEL_MODE' = adapter; then
      test -d '$ADAPTER'
      test -f '$ADAPTER/adapter_config.json'
      test -f '$ADAPTER/adapter_model.safetensors'
    fi
    mkdir -p '$REMOTE_OUTPUT_ROOT/logs'

    old_pid=''
    if test -r '$VLLM_PID_FILE'; then
      old_pid=\$(tr -d '[:space:]' < '$VLLM_PID_FILE')
      if test -n \"\$old_pid\" && kill -0 \"\$old_pid\" 2>/dev/null; then
        command_line=\$(tr '\0' ' ' < \"/proc/\$old_pid/cmdline\")
        if test '$MODEL_MODE' = adapter; then
          case \"\$command_line\" in
            *'$BASE_MODEL'*'$SERVED_MODEL'*'$ADAPTER'*'--port $REMOTE_PORT'*'--generation-config vllm'*) ;;
            *)
              printf 'live pid does not match the requested adapter service: %s\n' \"\$old_pid\" >&2
              exit 44
              ;;
          esac
        else
          case \"\$command_line\" in
            *'$BASE_MODEL'*'$SERVED_MODEL'*'--port $REMOTE_PORT'*'--generation-config vllm'*) ;;
            *)
              printf 'live pid does not match the requested standalone service: %s\n' \"\$old_pid\" >&2
              exit 44
              ;;
          esac
        fi
        if ! ss -ltnH 'sport = :$REMOTE_PORT' | grep -q .; then
          printf 'matching pid is live but requested port is not listening: %s\n' '$REMOTE_PORT' >&2
          exit 45
        fi
        if test '$EXISTING_SERVICE_POLICY' = reject; then
          printf 'vLLM pid file already refers to a matching live process: %s\n' \"\$old_pid\" >&2
          exit 42
        fi
        printf 'existing %s\n' \"\$old_pid\"
        exit 0
      fi
      rm -f '$VLLM_PID_FILE'
    fi
    if ss -ltnH 'sport = :$REMOTE_PORT' | grep -q .; then
      printf 'remote port is already in use without a matching pid file: %s\n' '$REMOTE_PORT' >&2
      exit 43
    fi
    printf 'new\n'
  "
)"

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" \
  >"$LOCAL_TUNNEL_LOG" 2>&1 &
tunnel_pid=$!
sleep 1
kill -0 "$tunnel_pid" 2>/dev/null ||
  die "SSH tunnel failed; see ${LOCAL_TUNNEL_LOG}"

if [[ "$service_state" == existing\ * ]]; then
  remote_pid="${service_state#existing }"
  if test "$EXISTING_SERVICE_POLICY" = adopt; then
    owns_remote_service=1
  fi
  printf 'Attached to matching %s vLLM pid=%s policy=%s\n' \
    "$SERVED_MODEL" "$remote_pid" "$EXISTING_SERVICE_POLICY"
else
  test "$service_state" = new || die "unexpected remote service state: ${service_state}"
  remote_lora_flags=""
  if test "$MODEL_MODE" = adapter; then
    remote_lora_flags="--enable-lora --lora-modules '$SERVED_MODEL=$ADAPTER' --max-lora-rank 16"
  fi
  remote_pid="$(
    ssh "$REMOTE" "
      set -eu
      CUDA_VISIBLE_DEVICES='$GPU_ID' HF_HUB_OFFLINE=1 nohup '$REMOTE_VLLM_PYTHON' \
        -m vllm.entrypoints.openai.api_server \
        --model '$BASE_MODEL' \
        --served-model-name '$SERVED_MODEL' \
        $remote_lora_flags \
        --port '$REMOTE_PORT' \
        --host 127.0.0.1 \
        --dtype bfloat16 \
        --max-model-len '$MAX_MODEL_LEN' \
        --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
        --max-num-seqs '$MAX_NUM_SEQS' \
        --max-num-batched-tokens '$MAX_NUM_BATCHED_TOKENS' \
        --generation-config vllm \
        > '$VLLM_LOG' 2>&1 < /dev/null &
      pid=\$!
      printf '%s\n' \"\$pid\" > '$VLLM_PID_FILE'
      printf '%s\n' \"\$pid\"
    "
  )"
  owns_remote_service=1
fi
is_uint "$remote_pid" || die "remote vLLM did not return a valid pid: ${remote_pid}"

printf 'Using %s vLLM pid=%s mode=%s gpu=%s remote_port=%s local_port=%s\n' \
  "$SERVED_MODEL" "$remote_pid" "$MODEL_MODE" "$GPU_ID" "$REMOTE_PORT" "$LOCAL_PORT"

ready=0
remote_process_failures=0
deadline=$((SECONDS + MODEL_READY_TIMEOUT_SECONDS))
while test "$SECONDS" -lt "$deadline"; do
  if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    printf 'SSH tunnel exited while waiting for vLLM readiness\n' >&2
    exit 75
  fi
  if curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:${LOCAL_PORT}/v1/models" 2>/dev/null |
    grep -q "$SERVED_MODEL"; then
    ready=1
    break
  fi
  if ssh "$REMOTE" "kill -0 '$remote_pid' 2>/dev/null" >/dev/null 2>&1; then
    remote_process_failures=0
  else
    remote_process_failures=$((remote_process_failures + 1))
    if test "$remote_process_failures" -ge 3; then
      printf 'Remote model process exited or was unreachable three consecutive times; see %s on %s\n' \
        "$VLLM_LOG" "$REMOTE" >&2
      exit 75
    fi
  fi
  sleep "$MODEL_READY_POLL_SECONDS"
done
test "$ready" -eq 1 || {
  printf 'Timed out waiting for vLLM readiness after %ss\n' "$MODEL_READY_TIMEOUT_SECONDS" >&2
  exit 75
}

printf 'vLLM ready; starting bird-set pass@K evaluation in %s\n' "$RESULT_DIR"

operational_resume_args=()
if test "$ALLOW_OPERATIONAL_CONCURRENCY_RESUME" -eq 1; then
  operational_resume_args+=(--allow-operational-concurrency-resume)
fi

EVAL_ENABLE_THINKING="$EVAL_ENABLE_THINKING" \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
  .venv/bin/python -u src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
    --model "$SERVED_MODEL" \
    --examples-json "$EXAMPLES_JSON" \
    --allow-eval-tasks \
    --n "$N" \
    --n-samples "$N_SAMPLES" \
    --pass-k "$PASS_K" \
    --workers "$WORKERS" \
    --sample-workers "$SAMPLE_WORKERS" \
    --max-inflight-requests "$MAX_INFLIGHT" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
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
    >"$LOCAL_EVAL_LOG" 2>&1 &
eval_pid=$!

health_failures=0
while kill -0 "$eval_pid" 2>/dev/null; do
  sleep 15
  if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    printf 'SSH tunnel exited during evaluation\n' >&2
    kill "$eval_pid" 2>/dev/null || true
    wait "$eval_pid" 2>/dev/null || true
    eval_pid=""
    exit 75
  fi
  if curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:${LOCAL_PORT}/v1/models" 2>/dev/null |
    grep -q "$SERVED_MODEL"; then
    health_failures=0
  else
    health_failures=$((health_failures + 1))
    if test "$health_failures" -ge 3; then
      printf 'Model endpoint failed three consecutive health checks; see %s on %s\n' \
        "$VLLM_LOG" "$REMOTE" >&2
      kill "$eval_pid" 2>/dev/null || true
      wait "$eval_pid" 2>/dev/null || true
      eval_pid=""
      exit 75
    fi
  fi
done

set +e
wait "$eval_pid"
eval_status=$?
set -e
eval_pid=""

if test "$eval_status" -ne 0; then
  printf 'Evaluation failed with status %s; see %s\n' "$eval_status" "$LOCAL_EVAL_LOG" >&2
  exit "$eval_status"
fi

printf 'Evaluation complete: model=%s result_dir=%s log=%s\n' \
  "$SERVED_MODEL" "$RESULT_DIR" "$LOCAL_EVAL_LOG"
