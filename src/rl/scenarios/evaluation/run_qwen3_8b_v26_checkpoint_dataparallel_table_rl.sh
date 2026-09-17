#!/usr/bin/env bash
set -Eeuo pipefail

# Data-parallel evaluation of one intermediate checkpoint on table_rl.
# Each physical GPU owns one vLLM process and one disjoint BIRD-dev shard.
# The cleanup trap signals only the process groups created by this script.

PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
# Triton's runtime CUDA driver shim (``cuda_utils.c``) is compiled and linked with ``-lcuda``;
# the driver package ships only ``libcuda.so.1``, so without a linker-visible ``libcuda.so``
# the shim build fails (gcc CalledProcessError) and any previously cached shim is reported as
# "Bytes object is corrupted, checksum does not match". The training launcher sets this through
# start_vllm_server.sh; the evaluation path must set it too or every vLLM engine core dies at
# startup. Derive it from the python env unless the caller overrides it.
PYTHON_ENV=${PYTHON_ENV:-$(dirname "$(dirname "$PYTHON_BIN")")}
TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-"$PYTHON_ENV/var/triton-libcuda"}
export TRITON_LIBCUDA_PATH
if [[ ! -e "$TRITON_LIBCUDA_PATH/libcuda.so" ]]; then
  echo "ERROR: missing triton libcuda shim under $TRITON_LIBCUDA_PATH" >&2
  exit 2
fi
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
MODEL=${MODEL:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SOURCE_INPUT=${SOURCE_INPUT:-/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl}
DATABASE_ROOT=${DATABASE_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases}
ADAPTER=${ADAPTER:?ADAPTER must point to the staged checkpoint directory}
RUN_DIR=${RUN_DIR:?RUN_DIR must point to a new evaluation directory}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_fourlevel_batch14_700_single_gpu_a100_20260904_save10_keepall_r1/train700_single_gpu_seed20260829/checkpoint-10}
CHECKPOINT_GLOBAL_STEP=${CHECKPOINT_GLOBAL_STEP:-10}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
PORT0=${PORT0:-18270}
PORT1=${PORT1:-18271}
MAX_TOKENS=${MAX_TOKENS:-2048}
ERROR_FEEDBACK_VERSION=${ERROR_FEEDBACK_VERSION:-legacy}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
VLLM_BATCH_TOKENS=${VLLM_BATCH_TOKENS:-16384}
VLLM_MEMORY_UTILIZATION=${VLLM_MEMORY_UTILIZATION:-0.90}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-24}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$RUN_DIR/vllm_cache}
TORCHINDUCTOR_CACHE_ROOT=${TORCHINDUCTOR_CACHE_ROOT:-$RUN_DIR/torchinductor_cache}
VLLM_START_SEQUENTIALLY=${VLLM_START_SEQUENTIALLY:-0}
EVAL_WORKERS=${EVAL_WORKERS:-24}

WRAPPER=${WRAPPER:-$RUN_DIR/controller/formal_v26_rollout_passk.py}
SHARDER=${SHARDER:-$RUN_DIR/controller/make_eval_shards.py}
MERGER=${MERGER:-$RUN_DIR/controller/merge_eval_shards.py}

VLLM_PIDS=()
EVAL_PIDS=()
RUN_STATUS=running
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

check_gpu_idle() {
  local gpu=$1
  local used apps
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$used" =~ ^[0-9]+$ ]] || die "invalid memory reading for GPU $gpu: $used"
  (( used <= 512 )) || die "GPU $gpu is not idle: ${used} MiB used"
  apps=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ -z "$apps" ]] || die "GPU $gpu has compute processes: $apps"
}

check_port_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"port {port} is already in use")
PY
}

wait_ready() {
  local pid=$1 port=$2 served=$3
  local attempt response
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || die "owned vLLM pid $pid exited before readiness"
    response=$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
    if [[ "$response" == *"$served"* ]]; then
      echo "vLLM ready: gpu=$served port=$port pid=$pid"
      return 0
    fi
    sleep 3
  done
  die "vLLM readiness timeout: port=$port pid=$pid"
}

stop_group() {
  local pid=$1
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
}

force_stop_group() {
  local pid=$1
  [[ -n "$pid" ]] || return 0
  kill -KILL -- "-$pid" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  RUN_STATUS=failed
  (( exit_code == 0 )) && RUN_STATUS=completed

  local pid deadline
  for pid in "${EVAL_PIDS[@]-}"; do
    [[ -n "$pid" ]] && stop_group "$pid"
  done
  for pid in "${VLLM_PIDS[@]-}"; do
    [[ -n "$pid" ]] && stop_group "$pid"
  done
  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    local alive=0
    for pid in "${VLLM_PIDS[@]-}"; do
      if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${EVAL_PIDS[@]-}"; do
    [[ -n "$pid" ]] && force_stop_group "$pid"
  done
  for pid in "${VLLM_PIDS[@]-}"; do
    [[ -n "$pid" ]] && force_stop_group "$pid"
  done

  "$PYTHON_BIN" - "$RUN_DIR/owned_process_cleanup.json" "$RUN_STATUS" "$STARTED_AT" "${VLLM_PIDS[*]-}" "${EVAL_PIDS[*]-}" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
path, status, started, vllm, evaluators = sys.argv[1:]
payload = {
    "schema_version": "qwen3-v26-dataparallel-owned-process-cleanup-v1",
    "status": status,
    "started_at_utc": started,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "vllm_process_group_leaders": [int(x) for x in vllm.split() if x.isdigit()],
    "evaluator_process_group_leaders": [int(x) for x in evaluators.split() if x.isdigit()],
    "policy": "signal-only-owned-process-groups; no process discovery or preemption",
}
with open(path, "w", encoding="utf-8") as target:
    json.dump(payload, target, ensure_ascii=False, indent=2)
    target.write("\n")
PY
  exit "$exit_code"
}

trap 'exit 143' INT TERM
trap cleanup EXIT

[[ -x "$PYTHON_BIN" ]] || die "Python is missing or not executable: $PYTHON_BIN"
[[ -d "$RUNTIME" ]] || die "runtime is missing: $RUNTIME"
[[ -d "$MODEL" ]] || die "base model is missing: $MODEL"
[[ -f "$SOURCE_INPUT" ]] || die "source input is missing: $SOURCE_INPUT"
[[ -d "$DATABASE_ROOT" ]] || die "database root is missing: $DATABASE_ROOT"
[[ -f "$ADAPTER/adapter_model.safetensors" && -f "$ADAPTER/adapter_config.json" ]] || die "adapter is incomplete: $ADAPTER"
[[ -f "$WRAPPER" && -f "$SHARDER" && -f "$MERGER" ]] || die "controller files are missing under $RUN_DIR/controller"
[[ ! -e "$RUN_DIR/results" ]] || die "results directory already exists; resume is forbidden: $RUN_DIR/results"

check_gpu_idle "$GPU0"
check_gpu_idle "$GPU1"
check_port_free "$PORT0"
check_port_free "$PORT1"
[[ "$GPU0" != "$GPU1" ]] || die "GPU0 and GPU1 must be distinct"
[[ "$PORT0" != "$PORT1" ]] || die "PORT0 and PORT1 must be distinct"

mkdir -p "$RUN_DIR/input" "$RUN_DIR/shards" "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/controller"
mkdir -p "$VLLM_CACHE_ROOT/gpu0" "$VLLM_CACHE_ROOT/gpu1" \
  "$TORCHINDUCTOR_CACHE_ROOT/gpu0" "$TORCHINDUCTOR_CACHE_ROOT/gpu1"

"$PYTHON_BIN" - "$SOURCE_INPUT" "$RUN_DIR/input/bird_dev_20240627.table_rl.jsonl" "$DATABASE_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

source, output, db_root = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1534 or [int(row.get("example_index", i)) for i, row in enumerate(rows)] != list(range(1534)):
    raise SystemExit("source input must contain exactly example_index 0..1533")
remapped = []
for row in rows:
    item = dict(row)
    db_id = str(item["db_id"])
    item["db_path"] = str(db_root / db_id / f"{db_id}.sqlite")
    if not pathlib.Path(item["db_path"]).is_file():
        raise SystemExit(f"missing database for {db_id}: {item['db_path']}")
    remapped.append(item)
payload = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in remapped)
if hashlib.sha256(payload).hexdigest() != "636e096babe2db9dde096b655ed01a0e1e6aae1ea5cbf4e952770e02841721f5":
    raise SystemExit("derived input hash does not match the frozen BIRD-dev identity")
pathlib.Path(output).write_bytes(payload)
print(json.dumps({"records": len(remapped), "sha256": hashlib.sha256(payload).hexdigest()}))
PY

"$PYTHON_BIN" "$SHARDER" \
  --examples "$RUN_DIR/input/bird_dev_20240627.table_rl.jsonl" \
  --first "$RUN_DIR/shards/gpu0.jsonl" \
  --second "$RUN_DIR/shards/gpu1.jsonl" \
  --expected 1534

cat > "$RUN_DIR/evaluation_config.json" <<EOF
{
  "schema_version": "qwen3-v26-checkpoint-dataparallel-eval-v1",
  "source_checkpoint": "${SOURCE_CHECKPOINT}",
  "checkpoint": "${ADAPTER}",
  "checkpoint_global_step": ${CHECKPOINT_GLOBAL_STEP},
  "dataset": "BIRD-Dev 2024-06-27",
  "records": 1534,
  "shard_policy": "example_index_even_to_gpu0_odd_to_gpu1",
  "gpus": {"gpu0": ${GPU0}, "gpu1": ${GPU1}},
  "ports": {"gpu0": ${PORT0}, "gpu1": ${PORT1}},
  "decode": {"temperature": 0.0, "top_p": 1.0, "max_tokens": ${MAX_TOKENS}, "max_steps": 30, "n_samples": 1},
  "protocol_version": "version26",
  "protocol_hash": "4da19387399bd3a5",
  "runtime": "${RUNTIME}",
  "error_feedback_version": "${ERROR_FEEDBACK_VERSION}",
  "vllm": {"enforce_eager": ${VLLM_ENFORCE_EAGER}, "cache_root": "${VLLM_CACHE_ROOT}"},
  "started_at_utc": "${STARTED_AT}"
}
EOF

SERVING0=qwen3-v26-checkpoint${CHECKPOINT_GLOBAL_STEP}-gpu0
SERVING1=qwen3-v26-checkpoint${CHECKPOINT_GLOBAL_STEP}-gpu1

VLLM_COMMON_ARGS=(
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-batched-tokens "$VLLM_BATCH_TOKENS"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --gpu-memory-utilization "$VLLM_MEMORY_UTILIZATION"
  --generation-config vllm
  --enable-lora
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  VLLM_COMMON_ARGS+=(--enforce-eager)
fi

start_vllm() {
  local gpu=$1 port=$2 serving=$3 log_file=$4 cache_dir=$5 inductor_dir=$6
  setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_CACHE_ROOT="$cache_dir" \
    TORCHINDUCTOR_CACHE_DIR="$inductor_dir" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --tokenizer "$MODEL" --served-model-name "${serving}-base" \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 \
    "${VLLM_COMMON_ARGS[@]}" \
    --lora-modules "$serving=$ADAPTER" --max-lora-rank 64 \
    > "$log_file" 2>&1 < /dev/null &
  VLLM_PIDS+=("$!")
}

start_vllm "$GPU0" "$PORT0" "$SERVING0" "$RUN_DIR/logs/vllm_gpu0.log" \
  "$VLLM_CACHE_ROOT/gpu0" "$TORCHINDUCTOR_CACHE_ROOT/gpu0"
if [[ "$VLLM_START_SEQUENTIALLY" == "1" ]]; then
  wait_ready "${VLLM_PIDS[0]}" "$PORT0" "${SERVING0}-base"
fi
start_vllm "$GPU1" "$PORT1" "$SERVING1" "$RUN_DIR/logs/vllm_gpu1.log" \
  "$VLLM_CACHE_ROOT/gpu1" "$TORCHINDUCTOR_CACHE_ROOT/gpu1"

printf '%s\n' "${VLLM_PIDS[@]}" > "$RUN_DIR/vllm_pids.txt"
wait_ready "${VLLM_PIDS[0]}" "$PORT0" "${SERVING0}-base"
wait_ready "${VLLM_PIDS[1]}" "$PORT1" "${SERVING1}-base"

setsid env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 EVAL_ENABLE_THINKING=1 \
  ERROR_FEEDBACK_VERSION="$ERROR_FEEDBACK_VERSION" \
  TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$RUNTIME" FORMAL_TOOL_EXECUTION_TIMEOUT_SECONDS=10 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$PYTHON_BIN" -u "$WRAPPER" \
  --base-url "http://127.0.0.1:${PORT0}/v1" --model "$SERVING0" \
  --examples-json "$RUN_DIR/shards/gpu0.jsonl" --allow-eval-tasks --n 767 --n-samples 1 --pass-k 1 \
  --workers "$EVAL_WORKERS" --sample-workers 1 --first-sample-workers 0 --max-inflight-requests "$EVAL_WORKERS" \
  --max-steps 30 --max-tokens "$MAX_TOKENS" --temperature 0 --top-p 1 --api-retries 3 \
  --few-shot 0 --sample-detail full --summary-every 10 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full \
  --rolling-observation-style resident --denotation-comparison bird-set \
  --result-dir "$RUN_DIR/results/gpu0" \
  > "$RUN_DIR/logs/eval_gpu0.log" 2>&1 &
EVAL_PIDS+=("$!")
setsid env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 EVAL_ENABLE_THINKING=1 \
  ERROR_FEEDBACK_VERSION="$ERROR_FEEDBACK_VERSION" \
  TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$RUNTIME" FORMAL_TOOL_EXECUTION_TIMEOUT_SECONDS=10 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$PYTHON_BIN" -u "$WRAPPER" \
  --base-url "http://127.0.0.1:${PORT1}/v1" --model "$SERVING1" \
  --examples-json "$RUN_DIR/shards/gpu1.jsonl" --allow-eval-tasks --n 767 --n-samples 1 --pass-k 1 \
  --workers "$EVAL_WORKERS" --sample-workers 1 --first-sample-workers 0 --max-inflight-requests "$EVAL_WORKERS" \
  --max-steps 30 --max-tokens "$MAX_TOKENS" --temperature 0 --top-p 1 --api-retries 3 \
  --few-shot 0 --sample-detail full --summary-every 10 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full \
  --rolling-observation-style resident --denotation-comparison bird-set \
  --result-dir "$RUN_DIR/results/gpu1" \
  > "$RUN_DIR/logs/eval_gpu1.log" 2>&1 &
EVAL_PIDS+=("$!")
printf '%s\n' "${EVAL_PIDS[@]}" > "$RUN_DIR/eval_pids.txt"

set +e
wait "${EVAL_PIDS[0]}"
RC0=$?
wait "${EVAL_PIDS[1]}"
RC1=$?
set -e
if (( RC0 != 0 || RC1 != 0 )); then
  echo "evaluation shard failure: gpu0=$RC0 gpu1=$RC1" >&2
  exit 1
fi

"$PYTHON_BIN" "$MERGER" \
  --examples "$RUN_DIR/input/bird_dev_20240627.table_rl.jsonl" \
  --shard "$RUN_DIR/results/gpu0/all.jsonl" \
  --shard "$RUN_DIR/results/gpu1/all.jsonl" \
  --output "$RUN_DIR/results/merged" \
  --adapter "$ADAPTER" \
  --expected 1534

echo "checkpoint-${CHECKPOINT_GLOBAL_STEP} data-parallel evaluation completed: $RUN_DIR"
