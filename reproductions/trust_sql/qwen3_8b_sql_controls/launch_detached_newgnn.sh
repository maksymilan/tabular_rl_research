#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

PROJECT_DIR="${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}"
REMOTE="${REMOTE:-NewGNN}"
MODE="${MODE:-}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
RUN_ID="${RUN_ID:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sql_controls}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
REMOTE_SOURCE_INPUT="${REMOTE_SOURCE_INPUT:-/home/dengyan/tabular_rl_outputs/eval_inputs/qwen3_atomic_v26/bird_dev_20240627.jsonl}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
WORKERS="${WORKERS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
N="${N:-1534}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  die "usage: MODE=direct|iterative RUN_ID=name $0 [--dry-run]"
fi

[[ "$MODE" == "direct" || "$MODE" == "iterative" ]] ||
  die "MODE must be direct or iterative"
[[ "$MODEL_SIZE" == "4b" || "$MODEL_SIZE" == "8b" ]] ||
  die "MODEL_SIZE must be 4b or 8b"
[[ "$RUN_ID" =~ ^[a-z0-9][a-z0-9_]{5,95}$ ]] ||
  die "RUN_ID must contain only lowercase letters, digits, and underscores"
[[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || die "MAX_MODEL_LEN must be an integer"
[[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]] ||
  die "MAX_NUM_BATCHED_TOKENS must be an integer"
[[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] || die "MAX_NUM_SEQS must be an integer"
[[ "$WORKERS" =~ ^[0-9]+$ ]] || die "WORKERS must be an integer"
(( MAX_NUM_BATCHED_TOKENS >= 1024 )) ||
  die "MAX_NUM_BATCHED_TOKENS must be at least 1024"
(( MAX_NUM_SEQS >= 1 && MAX_NUM_SEQS <= 32 )) ||
  die "MAX_NUM_SEQS must be in [1,32]"
(( WORKERS >= 1 && WORKERS <= 32 )) || die "WORKERS must be in [1,32]"
(( WORKERS <= MAX_NUM_SEQS )) || die "WORKERS cannot exceed MAX_NUM_SEQS"
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] ||
  die "GPU_MEMORY_UTILIZATION must be a decimal in [0.50,0.95]"
[[ "$N" =~ ^[0-9]+$ ]] || die "N must be an integer"
(( N >= 1 && N <= 1534 )) || die "N must be in [1,1534]"

if [[ "$MODEL_SIZE" == "4b" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}"
else
  MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
fi
[[ "$MODEL_ROOT" == /home/dengyan/* ]] || die "MODEL_ROOT must be below /home/dengyan"
[[ "$REMOTE_SOURCE_INPUT" == /home/dengyan/* ]] ||
  die "REMOTE_SOURCE_INPUT must be below /home/dengyan"

if [[ "$MODE" == "direct" ]]; then
  GPU="${GPU:-5}"
  PORT="${PORT:-8030}"
else
  GPU="${GPU:-6}"
  PORT="${PORT:-8031}"
fi
[[ "$GPU" =~ ^[0-9]+$ ]] || die "GPU must be a non-negative integer"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer"

cd "$PROJECT_DIR"
SUPERVISOR="reproductions/trust_sql/qwen3_8b_sql_controls/remote_supervisor.py"
PREPARER="reproductions/trust_sql/qwen3_8b_atomic_sft1/prepare_remote_eval_inputs.py"
MODEL_VERIFIER="reproductions/trust_sql/qwen3_8b_sql_controls/verify_pinned_qwen3_model.py"
MODEL_SPECS="reproductions/trust_sql/qwen3_8b_sql_controls/qwen3_model_specs.json"
LOCK="reproductions/trust_sql/qwen3_8b_atomic_sft1/remote_eval_lock.json"
for required in "$SUPERVISOR" "$PREPARER" "$MODEL_VERIFIER" "$MODEL_SPECS" "$LOCK"; do
  test -f "$required" || die "missing required file: $required"
done

REMOTE_RUN="$REMOTE_ROOT/$RUN_ID"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'host=%s\nmode=%s\nmodel_size=%s\nmodel_root=%s\nsource_input=%s\nrun_dir=%s\ngpu=%s\nport=%s\nn=%s\nmax_model_len=%s\nmax_num_batched_tokens=%s\nmax_num_seqs=%s\nworkers=%s\ngpu_memory_utilization=%s\n' \
    "$REMOTE" "$MODE" "$MODEL_SIZE" "$MODEL_ROOT" "$REMOTE_SOURCE_INPUT" \
    "$REMOTE_RUN" "$GPU" "$PORT" "$N" "$MAX_MODEL_LEN" \
    "$MAX_NUM_BATCHED_TOKENS" "$MAX_NUM_SEQS" "$WORKERS" \
    "$GPU_MEMORY_UTILIZATION"
  exit 0
fi

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-sql-controls.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT INT TERM

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -czf "$STAGE_DIR/runtime.tar.gz" src
RUNTIME_SHA="$(shasum -a 256 "$STAGE_DIR/runtime.tar.gz" | awk '{print $1}')"

ssh "$REMOTE" \
  "set -e; test -f '$REMOTE_SOURCE_INPUT'; test \"\$(sha256sum '$REMOTE_SOURCE_INPUT' | cut -d' ' -f1)\" = '8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0'; test ! -e '$REMOTE_RUN'; mkdir -p '$REMOTE_RUN/controller' '$REMOTE_RUN/input' '$REMOTE_RUN/runtime'; cp '$REMOTE_SOURCE_INPUT' '$REMOTE_RUN/input/bird_dev_20240627.jsonl'"
scp "$STAGE_DIR/runtime.tar.gz" "$REMOTE:$REMOTE_RUN/runtime.tar.gz"
scp "$SUPERVISOR" "$REMOTE:$REMOTE_RUN/controller/remote_supervisor.py"
scp "$PREPARER" "$REMOTE:$REMOTE_RUN/controller/prepare_remote_eval_inputs.py"
scp "$MODEL_VERIFIER" "$REMOTE:$REMOTE_RUN/controller/verify_pinned_qwen3_model.py"
scp "$MODEL_SPECS" "$REMOTE:$REMOTE_RUN/controller/qwen3_model_specs.json"
scp "$LOCK" "$REMOTE:$REMOTE_RUN/controller/remote_eval_lock.json"

ssh "$REMOTE" \
  "set -e; tar -xzf '$REMOTE_RUN/runtime.tar.gz' -C '$REMOTE_RUN/runtime'; nohup setsid '$REMOTE_PYTHON' -u '$REMOTE_RUN/controller/remote_supervisor.py' --mode '$MODE' --model-size '$MODEL_SIZE' --model-root '$MODEL_ROOT' --run-dir '$REMOTE_RUN' --gpu '$GPU' --port '$PORT' --runtime-sha256 '$RUNTIME_SHA' --n '$N' --max-model-len '$MAX_MODEL_LEN' --max-num-batched-tokens '$MAX_NUM_BATCHED_TOKENS' --max-num-seqs '$MAX_NUM_SEQS' --workers '$WORKERS' --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' >'$REMOTE_RUN/supervisor.log' 2>&1 </dev/null & echo \$! >'$REMOTE_RUN/supervisor.pid'"

printf 'submitted model=%s mode=%s run=%s runtime_sha256=%s\n' \
  "$MODEL_SIZE" "$MODE" "$REMOTE_RUN" "$RUNTIME_SHA"
printf 'status: ssh %s cat %s/status.json\n' "$REMOTE" "$REMOTE_RUN"
