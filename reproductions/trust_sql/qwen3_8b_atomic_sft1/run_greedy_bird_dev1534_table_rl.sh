#!/usr/bin/env bash
set -euo pipefail

# Evaluate either the pinned Qwen3-8B base model or one QLoRA adapter with the same historical
# atomic version26 client.  This launcher never imports the current checkout's protocol modules.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
EXPECTED_COMMIT="4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
EXPECTED_PROMPT_SHA256="848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
EXPECTED_EXAMPLES_SHA256="8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0"

RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_DIR/tmp/version26-runtime-$EXPECTED_COMMIT}"
LOCAL_PYTHON="${LOCAL_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
EXAMPLES_JSON="${EXAMPLES_JSON:-$PROJECT_DIR/data/eval_inputs/bird_dev_20240627.jsonl}"
REMOTE="${REMOTE:-table_rl}"
REMOTE_VLLM_PYTHON="${REMOTE_VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
MODEL_MODE="${MODEL_MODE:-base}"
ADAPTER="${ADAPTER:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

GPU_IDS="${GPU_IDS:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
REMOTE_PORT="${REMOTE_PORT:-8016}"
LOCAL_PORT="${LOCAL_PORT:-18016}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_READY_TIMEOUT_SECONDS="${MODEL_READY_TIMEOUT_SECONDS:-900}"
MODEL_READY_POLL_SECONDS="${MODEL_READY_POLL_SECONDS:-3}"

# Locked evaluation semantics. Environment overrides are accepted only when they retain the lock.
N="${N:-1534}"
N_SAMPLES="${N_SAMPLES:-1}"
PASS_K="${PASS_K:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
HISTORY_TURNS="${HISTORY_TURNS:-4}"
WORKERS="${WORKERS:-2}"
MAX_INFLIGHT="${MAX_INFLIGHT:-2}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) die "only --dry-run is supported" ;;
esac

case "$MODEL_MODE" in
  base)
    test -z "$ADAPTER" || die "ADAPTER must be empty in MODEL_MODE=base"
    SERVED_MODEL="${SERVED_MODEL:-qwen3-8b-atomic-v26-base}"
    BACKBONE_SERVED_MODEL="${BACKBONE_SERVED_MODEL:-$SERVED_MODEL}"
    RUN_NAME="${RUN_NAME:-qwen3_8b_base_atomic_v26_bird_dev1534_greedy1_bird_set}"
    ;;
  adapter)
    test -n "$ADAPTER" || die "ADAPTER is required in MODEL_MODE=adapter"
    SERVED_MODEL="${SERVED_MODEL:-qwen3-8b-atomic-v26-sft1-qlora}"
    BACKBONE_SERVED_MODEL="${BACKBONE_SERVED_MODEL:-qwen3-8b-atomic-v26-backbone}"
    RUN_NAME="${RUN_NAME:-qwen3_8b_sft1_qlora_atomic_v26_bird_dev1534_greedy1_bird_set}"
    ;;
  *) die "MODEL_MODE must be base or adapter" ;;
esac

RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/data/results/$RUN_NAME}"
LOCAL_EVAL_LOG="${LOCAL_EVAL_LOG:-/tmp/$RUN_NAME.eval.log}"
LOCAL_TUNNEL_LOG="${LOCAL_TUNNEL_LOG:-/tmp/$RUN_NAME.tunnel.log}"
VLLM_LOG="${VLLM_LOG:-$REMOTE_OUTPUT_ROOT/logs/$RUN_NAME.vllm.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-$REMOTE_OUTPUT_ROOT/logs/$RUN_NAME.vllm.pid}"

test -x "$LOCAL_PYTHON" || die "local evaluation Python is not executable: $LOCAL_PYTHON"
test -f "$EXAMPLES_JSON" || die "BIRD evaluation input is missing: $EXAMPLES_JSON"
test -d "$RUNTIME_ROOT" ||
  die "exact runtime is missing; run prepare_version26_runtime.sh first: $RUNTIME_ROOT"

for name in N N_SAMPLES MAX_STEPS MAX_TOKENS HISTORY_TURNS WORKERS MAX_INFLIGHT \
  REMOTE_PORT LOCAL_PORT MAX_LORA_RANK TENSOR_PARALLEL_SIZE MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS MODEL_READY_TIMEOUT_SECONDS MODEL_READY_POLL_SECONDS; do
  value="${!name}"
  is_uint "$value" || die "$name must be a non-negative integer, got $value"
done
test "$N" -eq 1534 || die "N is locked to 1534"
test "$N_SAMPLES" -eq 1 || die "N_SAMPLES is locked to 1"
test "$PASS_K" = "1" || die "PASS_K is locked to 1"
test "$MAX_STEPS" -eq 30 || die "MAX_STEPS is locked to 30"
test "$MAX_TOKENS" -ge 2048 || die "MAX_TOKENS must be at least 2048"
test "$HISTORY_TURNS" -eq 4 || die "HISTORY_TURNS is locked to 4"
test "$TEMPERATURE" = "0" || die "TEMPERATURE is locked to 0"
test "$TOP_P" = "1" || die "TOP_P is locked to 1"
test "$WORKERS" -gt 0 || die "WORKERS must be positive"
test "$MAX_INFLIGHT" -gt 0 || die "MAX_INFLIGHT must be positive"
case "$DRY_RUN" in 0 | 1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac

[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU_IDS must be a comma-separated integer list"
commas="${GPU_IDS//[^,]/}"
gpu_count=$(( ${#commas} + 1 ))
test "$gpu_count" -eq "$TENSOR_PARALLEL_SIZE" ||
  die "GPU_IDS count ($gpu_count) must equal TENSOR_PARALLEL_SIZE ($TENSOR_PARALLEL_SIZE)"
[[ "$SERVED_MODEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe SERVED_MODEL: $SERVED_MODEL"
[[ "$BACKBONE_SERVED_MODEL" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "unsafe BACKBONE_SERVED_MODEL: $BACKBONE_SERVED_MODEL"
if test "$MODEL_MODE" = adapter; then
  test "$BACKBONE_SERVED_MODEL" != "$SERVED_MODEL" ||
    die "adapter alias and backbone served name must be different"
fi
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] ||
  die "GPU_MEMORY_UTILIZATION must be in [0,1]"

for value in "$BASE_MODEL" "$ADAPTER" "$REMOTE_VLLM_PYTHON" "$REMOTE_OUTPUT_ROOT" \
  "$VLLM_LOG" "$VLLM_PID_FILE"; do
  [[ "$value" != *"'"* ]] || die "remote values may not contain a single quote: $value"
done

actual_examples_sha256="$($LOCAL_PYTHON -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$EXAMPLES_JSON")"
test "$actual_examples_sha256" = "$EXPECTED_EXAMPLES_SHA256" ||
  die "BIRD input hash mismatch: $actual_examples_sha256"
example_audit="$($LOCAL_PYTHON -c '
import json, os, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
missing = [row.get("db_path") for row in rows if not os.path.isfile(row.get("db_path", ""))]
print(f"{len(rows)} {len(missing)}")
' "$EXAMPLES_JSON")"
test "$example_audit" = "1534 0" || die "BIRD input/path gate failed: $example_audit"

runtime_gate="$($LOCAL_PYTHON "$SCRIPT_DIR/verify_version26_runtime.py" \
  --runtime-root "$RUNTIME_ROOT" --json)"
actual_prompt_sha256="$($LOCAL_PYTHON -c \
  'import json,sys; print(json.load(sys.stdin)["protocol"]["rolling_system_prompt_sha256"])' \
  <<<"$runtime_gate")"
test "$actual_prompt_sha256" = "$EXPECTED_PROMPT_SHA256" ||
  die "rolling prompt hash gate failed: $actual_prompt_sha256"

if test "$DRY_RUN" -eq 1; then
  printf '%s\n' \
    "dry-run gate: OK" \
    "runtime=$RUNTIME_ROOT" \
    "source_commit=$EXPECTED_COMMIT" \
    "prompt_sha256=$EXPECTED_PROMPT_SHA256" \
    "mode=$MODEL_MODE" \
    "served_model=$SERVED_MODEL" \
    "backbone_served_model=$BACKBONE_SERVED_MODEL" \
    "dataset=BIRD-dev1534" \
    "decode=greedy temperature=0 top_p=1 max_tokens=$MAX_TOKENS" \
    "protocol=atomic version26 history=4 max_steps=30 bird-set" \
    "qwen3_enable_thinking=true reasoning_parser=none"
  exit 0
fi

mkdir -p "$RESULT_DIR"
printf '%s\n' "$runtime_gate" > "$RESULT_DIR/version26_runtime_gate.json"

remote_model_args="--model-root '$BASE_MODEL' --max-lora-rank '$MAX_LORA_RANK'"
if test "$MODEL_MODE" = adapter; then
  remote_model_args="$remote_model_args --adapter '$ADAPTER'"
fi
ssh "$REMOTE" "'$REMOTE_VLLM_PYTHON' - $remote_model_args" \
  < "$SCRIPT_DIR/verify_qwen3_model.py" \
  > "$RESULT_DIR/qwen3_model_gate.json"

ssh "$REMOTE" "
  set -eu
  test -x '$REMOTE_VLLM_PYTHON'
  test -d '$BASE_MODEL'
  mkdir -p '$REMOTE_OUTPUT_ROOT/logs'
  if test -r '$VLLM_PID_FILE'; then
    old_pid=\$(tr -d '[:space:]' < '$VLLM_PID_FILE')
    if test -n \"\$old_pid\" && kill -0 \"\$old_pid\" 2>/dev/null; then
      printf 'refusing to replace live vLLM pid %s\n' \"\$old_pid\" >&2
      exit 42
    fi
    rm -f '$VLLM_PID_FILE'
  fi
  if ss -ltnH 'sport = :$REMOTE_PORT' | grep -q .; then
    printf 'remote port is already in use: %s\n' '$REMOTE_PORT' >&2
    exit 43
  fi
"

remote_lora_flags=""
if test "$MODEL_MODE" = adapter; then
  remote_lora_flags="--enable-lora --lora-modules '$SERVED_MODEL=$ADAPTER' --max-lora-rank '$MAX_LORA_RANK'"
fi

remote_pid=""
tunnel_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if test -n "$tunnel_pid" && kill -0 "$tunnel_pid" 2>/dev/null; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
  ssh "$REMOTE" "
    if test -r '$VLLM_PID_FILE'; then
      candidate=\$(tr -d '[:space:]' < '$VLLM_PID_FILE')
      if test -n \"\$candidate\" && kill -0 \"\$candidate\" 2>/dev/null; then
        command_line=\$(tr '\0' ' ' < \"/proc/\$candidate/cmdline\")
        case \"\$command_line\" in
          *'$BASE_MODEL'*'--served-model-name $BACKBONE_SERVED_MODEL'*'--port $REMOTE_PORT'*)
            if test -z '$remote_pid' || test \"\$candidate\" = '$remote_pid'; then
              kill -TERM -- \"-\$candidate\" 2>/dev/null || \
                kill -TERM \"\$candidate\" 2>/dev/null || true
              attempt=0
              while kill -0 \"\$candidate\" 2>/dev/null && test \"\$attempt\" -lt 15; do
                sleep 1
                attempt=\$((attempt + 1))
              done
              if kill -0 \"\$candidate\" 2>/dev/null; then
                kill -KILL -- \"-\$candidate\" 2>/dev/null || \
                  kill -KILL \"\$candidate\" 2>/dev/null || true
              fi
              rm -f '$VLLM_PID_FILE'
            fi
            ;;
        esac
      else
        rm -f '$VLLM_PID_FILE'
      fi
    fi
  " >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

remote_pid="$({ ssh "$REMOTE" "
  set -eu
  setsid env \
    CUDA_VISIBLE_DEVICES='$GPU_IDS' \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    '$REMOTE_VLLM_PYTHON' -m vllm.entrypoints.openai.api_server \
      --model '$BASE_MODEL' \
      --tokenizer '$BASE_MODEL' \
      --served-model-name '$BACKBONE_SERVED_MODEL' \
      $remote_lora_flags \
      --host 127.0.0.1 \
      --port '$REMOTE_PORT' \
      --dtype bfloat16 \
      --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' \
      --max-model-len '$MAX_MODEL_LEN' \
      --max-num-batched-tokens '$MAX_NUM_BATCHED_TOKENS' \
      --max-num-seqs '$MAX_NUM_SEQS' \
      --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
      --generation-config vllm \
      > '$VLLM_LOG' 2>&1 < /dev/null &
  pid=\$!
  printf '%s\n' \"\$pid\" > '$VLLM_PID_FILE'
  printf '%s\n' \"\$pid\"
"; } )"
is_uint "$remote_pid" || die "vLLM did not return a valid remote pid: $remote_pid"

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" \
  > "$LOCAL_TUNNEL_LOG" 2>&1 &
tunnel_pid=$!

ready=0
deadline=$((SECONDS + MODEL_READY_TIMEOUT_SECONDS))
while test "$SECONDS" -lt "$deadline"; do
  kill -0 "$tunnel_pid" 2>/dev/null || die "SSH tunnel exited; see $LOCAL_TUNNEL_LOG"
  if curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:${LOCAL_PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL"; then
    ready=1
    break
  fi
  ssh "$REMOTE" "kill -0 '$remote_pid' 2>/dev/null" >/dev/null 2>&1 ||
    die "remote vLLM exited; see $VLLM_LOG on $REMOTE"
  sleep "$MODEL_READY_POLL_SECONDS"
done
test "$ready" -eq 1 || die "vLLM readiness timed out after ${MODEL_READY_TIMEOUT_SECONDS}s"

printf 'vLLM ready; exact version26 BIRD-dev1534 greedy evaluation starts now\n'
set +e
(
  unset PYTHONPATH EVAL_SYSTEM_PROMPT_VARIANT
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export EVAL_ENABLE_THINKING=1
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  cd "$PROJECT_DIR"
  "$LOCAL_PYTHON" -u "$RUNTIME_ROOT/src/eval/rollout_passk.py" \
    --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" \
    --model "$SERVED_MODEL" \
    --examples-json "$EXAMPLES_JSON" \
    --allow-eval-tasks \
    --n 1534 \
    --n-samples 1 \
    --pass-k 1 \
    --workers "$WORKERS" \
    --sample-workers 1 \
    --max-inflight-requests "$MAX_INFLIGHT" \
    --max-steps 30 \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0 \
    --top-p 1 \
    --sample-detail full \
    --summary-every 10 \
    --context-mode rolling-legal-history \
    --history-turns 4 \
    --rolling-prompt-variant full \
    --rolling-observation-style resident \
    --denotation-comparison bird-set \
    --result-dir "$RESULT_DIR" \
    --resume
) 2>&1 | tee "$LOCAL_EVAL_LOG"
eval_status=${PIPESTATUS[0]}
set -e
exit "$eval_status"
