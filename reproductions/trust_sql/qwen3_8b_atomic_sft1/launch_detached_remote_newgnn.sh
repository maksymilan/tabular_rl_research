#!/usr/bin/env bash
set -euo pipefail

# Stage a self-hashed controller bundle, then detach one complete evaluation arm on NewGNN.
# SSH is used only for staging/submission; model serving and the historical harness communicate
# through NewGNN localhost and continue after this launcher disconnects.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
LOCAL_PYTHON="${LOCAL_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
REMOTE="${REMOTE:-NewGNN}"
MODE="${MODE:-base}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_${MODEL_SIZE}_atomic_v26}"
REMOTE_SOURCE="${REMOTE_SOURCE:-/home/dengyan/tabular_rl_outputs/eval_inputs/qwen3_atomic_v26/bird_dev_20240627.jsonl}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}"
if test "$MODEL_SIZE" = 4b; then
  MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}"
  ADAPTER="${ADAPTER:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}"
else
  MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
  ADAPTER="${ADAPTER:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}"
fi
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-512}"
MODEL_READY_TIMEOUT="${MODEL_READY_TIMEOUT:-900}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

case "$MODE" in
  base)
    GPU_IDS="${GPU_IDS:-5}"
    PORT="${PORT:-8020}"
    ;;
  adapter)
    GPU_IDS="${GPU_IDS:-6}"
    PORT="${PORT:-8021}"
    ;;
  *) die "MODE must be base or adapter" ;;
esac
case "$MODEL_SIZE" in 4b | 8b) ;; *) die "MODEL_SIZE must be 4b or 8b" ;; esac

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) die "only --dry-run is supported" ;;
esac
case "$DRY_RUN" in 0 | 1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac

[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU_IDS must be comma-separated integers"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer"
(( PORT >= 1024 && PORT <= 65535 )) || die "PORT must be in [1024,65535]"
[[ "$MAX_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || die "MAX_GPU_MEMORY_MIB must be an integer"
[[ "$MODEL_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "MODEL_READY_TIMEOUT must be positive"

RUN_ID="${RUN_ID:-${MODE}_$(date -u +%Y%m%d_%H%M%S)_${RANDOM}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || die "unsafe RUN_ID: $RUN_ID"
for remote_value in \
  "$REMOTE_PYTHON" "$REMOTE_RUN_ROOT" "$REMOTE_SOURCE" "$RUNTIME_ROOT" \
  "$MODEL_ROOT" "$ADAPTER"; do
  [[ "$remote_value" != *"'"* ]] || die "remote path contains a single quote: $remote_value"
done

bundle_files=(
  prepare_remote_eval_inputs.py
  remote_eval_lock.json
  remote_eval_supervisor.py
  runtime_lock.json
  verify_qwen3_model.py
  verify_remote_eval_assets.py
  verify_version26_runtime.py
)
shared_model_files=(
  "$PROJECT_DIR/reproductions/trust_sql/qwen3_8b_sql_controls/verify_pinned_qwen3_model.py"
  "$PROJECT_DIR/reproductions/trust_sql/qwen3_8b_sql_controls/qwen3_model_specs.json"
)
if test "$MODEL_SIZE" = 4b && test "$MODE" = adapter; then
  bundle_files+=(qwen3_4b_adapter_lock.json)
fi
for name in "${bundle_files[@]}"; do
  test -f "$SCRIPT_DIR/$name" || die "missing bundle file: $SCRIPT_DIR/$name"
done
for path in "${shared_model_files[@]}"; do
  test -f "$path" || die "missing shared model gate file: $path"
done
test -x "$LOCAL_PYTHON" || die "local Python is not executable: $LOCAL_PYTHON"

SOURCE_SHA256="$($LOCAL_PYTHON -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$PROJECT_DIR/data/eval_inputs/bird_dev_20240627.jsonl")"
test "$SOURCE_SHA256" = "8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0" || \
  die "local source evaluation hash drifted: $SOURCE_SHA256"

REMOTE_RUN_DIR="$REMOTE_RUN_ROOT/$RUN_ID"
REMOTE_BUNDLE_DIR="$REMOTE_RUN_DIR/bundle"
remote_adapter_clause=""
if test "$MODE" = adapter; then
  remote_adapter_clause="--adapter '$ADAPTER'"
fi

if test "$DRY_RUN" -eq 1; then
  printf '%s\n' \
    "dry-run gate: OK" \
    "remote=$REMOTE" \
    "mode=$MODE" \
    "model_size=$MODEL_SIZE" \
    "run_id=$RUN_ID" \
    "run_dir=$REMOTE_RUN_DIR" \
    "gpu_ids=$GPU_IDS port=$PORT" \
    "runtime=$RUNTIME_ROOT" \
    "source_sha256=$SOURCE_SHA256" \
    "execution=NewGNN localhost vLLM + exact version26 harness" \
    "evaluation=1534 greedy history4 max_steps30 max_tokens2048 bird-set enable_thinking" \
    "concurrency=max_num_seqs4 workers4 max_inflight4" \
    "detachment=nohup+setsid; no SSH tunnel"
  exit 0
fi

temporary="$(mktemp -d /tmp/qwen3-atomic-v26-remote-launch.XXXXXX)"
cleanup() {
  rm -rf "$temporary"
}
trap cleanup EXIT INT TERM

checksums="$temporary/SHA256SUMS"
: > "$checksums"
for name in "${bundle_files[@]}"; do
  digest="$($LOCAL_PYTHON -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$SCRIPT_DIR/$name")"
  printf '%s  %s\n' "$digest" "$name" >> "$checksums"
done
for path in "${shared_model_files[@]}"; do
  name="$(basename -- "$path")"
  digest="$($LOCAL_PYTHON -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$path")"
  printf '%s  %s\n' "$digest" "$name" >> "$checksums"
done

ssh "$REMOTE" "
  set -eu
  test -x '$REMOTE_PYTHON'
  test -f '$REMOTE_SOURCE'
  test -d '$RUNTIME_ROOT'
  test -d '$MODEL_ROOT'
  mkdir -p '$REMOTE_RUN_ROOT'
  mkdir '$REMOTE_RUN_DIR'
  mkdir '$REMOTE_BUNDLE_DIR'
"
scp "${bundle_files[@]/#/$SCRIPT_DIR/}" "${shared_model_files[@]}" "$checksums" \
  "$REMOTE:$REMOTE_BUNDLE_DIR/"

remote_pid="$(ssh "$REMOTE" "
  set -eu
  cd '$REMOTE_BUNDLE_DIR'
  sha256sum -c SHA256SUMS >/dev/null
  actual_source=\$(sha256sum '$REMOTE_SOURCE' | awk '{print \$1}')
  test \"\$actual_source\" = '$SOURCE_SHA256'
  nohup setsid '$REMOTE_PYTHON' '$REMOTE_BUNDLE_DIR/remote_eval_supervisor.py' \\
    --mode '$MODE' \\
    --model-size '$MODEL_SIZE' \\
    --run-dir '$REMOTE_RUN_DIR' \\
    --gpu-ids '$GPU_IDS' \\
    --port '$PORT' \\
    --source '$REMOTE_SOURCE' \\
    --runtime-root '$RUNTIME_ROOT' \\
    --model-root '$MODEL_ROOT' \\
    $remote_adapter_clause \\
    --max-gpu-memory-mib '$MAX_GPU_MEMORY_MIB' \\
    --model-ready-timeout '$MODEL_READY_TIMEOUT' \\
    > '$REMOTE_RUN_DIR/supervisor.log' 2>&1 < /dev/null &
  pid=\$!
  printf '%s\n' \"\$pid\" > '$REMOTE_RUN_DIR/submitted.pid'
  printf '%s\n' \"\$pid\"
")"
[[ "$remote_pid" =~ ^[0-9]+$ ]] || die "remote supervisor did not return a PID: $remote_pid"

printf '%s\n' \
  "submitted detached all-NewGNN evaluation" \
  "mode=$MODE" \
  "model_size=$MODEL_SIZE" \
  "pid=$remote_pid" \
  "run_dir=$REMOTE_RUN_DIR" \
  "status=$REMOTE_RUN_DIR/status.json" \
  "supervisor_log=$REMOTE_RUN_DIR/supervisor.log"
