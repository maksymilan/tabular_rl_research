#!/usr/bin/env bash
# One-shot Gate60 SAAM four-level pipeline.  RL uses one GPU for the trainer
# and one for online vLLM rollout; after RL exits, both GPUs run disjoint
# BIRD-dev evaluation shards and the results are merged with coverage checks.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
MODE=${1:---run}
[[ "$MODE" == "--run" || "$MODE" == "--plan" ]] || {
  printf 'usage: %s [--plan|--run]\n' "$0" >&2
  exit 2
}

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_saam_fourlevel_gate60_20260901}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
EVAL_RUNTIME=${EVAL_RUNTIME:-$OUTPUT_ROOT/eval_runtime_qwen3_8b_v26_saam_gate60_20260829}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_gate60.yaml}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_fourlevel_gate60_train_eval_20260902}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train_two_pass_seed20260902}
EVAL_ROOT=${EVAL_ROOT:-$RUN_ROOT/eval_dev_dataparallel}
# The evaluator's canonical source input lives in the project data tree, not
# in the stripped evaluation runtime bundle.
EVAL_INPUT=${EVAL_INPUT:-/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl}
EVAL_ENTRY=${EVAL_ENTRY:-$EVAL_RUNTIME/src/rl/evaluation/formal_v26_rollout_passk.py}

# Same physical pair in both stages.  The roles change after training.
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
EVAL_GPU_A=${EVAL_GPU_A:-$TRAIN_GPU}
EVAL_GPU_B=${EVAL_GPU_B:-$VLLM_GPU}
TRAIN_VLLM_PORT=${TRAIN_VLLM_PORT:-8092}
TRAIN_VLLM_GROUP_PORT=${TRAIN_VLLM_GROUP_PORT:-51292}
EVAL_PORT_A=${EVAL_PORT_A:-8087}
EVAL_PORT_B=${EVAL_PORT_B:-8088}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-1}
MAXIMUM_USED_MIB=${MAXIMUM_USED_MIB:-512}
MODEL_READY_TIMEOUT=${MODEL_READY_TIMEOUT:-900}
EXPECTED_EVAL_RECORDS=${EXPECTED_EVAL_RECORDS:-1534}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
for path in "$PYTHON" "$RUNTIME" "$MODEL_PATH" "$ADAPTER_PATH" "$PROTOCOL_RUNTIME" "$EVAL_RUNTIME" "$TASKS" "$CONFIG" "$EVAL_INPUT" "$EVAL_ENTRY"; do
  [[ -e "$path" ]] || die "missing path: $path"
done
MAKE_SHARDS=${MAKE_SHARDS:-$EVAL_RUNTIME/src/rl/evaluation/make_eval_shards.py}
MERGE_SHARDS=${MERGE_SHARDS:-$EVAL_RUNTIME/src/rl/evaluation/merge_eval_shards.py}
[[ -f "$MAKE_SHARDS" ]] || MAKE_SHARDS="$SCRIPT_DIR/../evaluation/make_eval_shards.py"
[[ -f "$MERGE_SHARDS" ]] || MERGE_SHARDS="$SCRIPT_DIR/../evaluation/merge_eval_shards.py"
[[ -f "$MAKE_SHARDS" && -f "$MERGE_SHARDS" ]] || die "missing shard helper scripts"
[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || die "training GPU and rollout GPU must differ"
[[ "$EVAL_GPU_A" != "$EVAL_GPU_B" ]] || die "evaluation GPUs must differ"
for value in "$TRAIN_GPU" "$VLLM_GPU" "$EVAL_GPU_A" "$EVAL_GPU_B" "$TRAIN_VLLM_PORT" "$TRAIN_VLLM_GROUP_PORT" "$EVAL_PORT_A" "$EVAL_PORT_B" "$MAXIMUM_USED_MIB" "$EXPECTED_EVAL_RECORDS"; do
  is_uint "$value" || die "non-negative integer required: $value"
done
is_posint "$MODEL_READY_TIMEOUT" || die "MODEL_READY_TIMEOUT must be positive"
if [[ "$MODE" == "--plan" ]]; then
  "$PYTHON" - <<PY
import json
print(json.dumps({
  "status": "plan_only_no_mutation",
  "training": {"trainer_gpu": int("$TRAIN_GPU"), "rollout_gpu": int("$VLLM_GPU"), "output": "$TRAIN_OUT"},
  "evaluation": {"gpu_a": int("$EVAL_GPU_A"), "gpu_b": int("$EVAL_GPU_B"), "records": int("$EXPECTED_EVAL_RECORDS"), "output": "$EVAL_ROOT/final"},
  "stages": ["online RL", "stop rollout vLLM", "two disjoint greedy eval shards", "coverage-checked merge"],
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite existing run root: $RUN_ROOT"
mkdir -p "$RUN_ROOT/logs"
exec > >(tee "$RUN_ROOT/logs/pipeline.log") 2>&1

STATUS="$RUN_ROOT/status.json"
set_status() {
  local state=$1 detail=$2
  STATE="$state" DETAIL="$detail" STATUS_PATH="$STATUS" RUN_ROOT_VALUE="$RUN_ROOT" \
    "$PYTHON" - <<'PY'
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["STATUS_PATH"])
record = {"schema_version": "qwen3-v26-saam-fourlevel-train-eval-dp-v1", "state": os.environ["STATE"], "detail": os.environ["DETAIL"], "updated_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": os.environ["RUN_ROOT_VALUE"]}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        json.dump(record, target, ensure_ascii=False, indent=2, sort_keys=True); target.write("\n"); target.flush(); os.fsync(target.fileno())
    os.replace(temporary, path)
finally:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
PY
}

gpu_idle() {
  local gpu=$1 used pids
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le "$MAXIMUM_USED_MIB" ]]
}
stop_group() {
  local pgid=${1:-}; [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
vllm_pgid=""; trainer_pgid=""; eval_a_vllm_pgid=""; eval_b_vllm_pgid=""; eval_a_pgid=""; eval_b_pgid=""
cleanup() {
  local code=$?; trap - EXIT INT TERM
  stop_group "$eval_a_pgid"; stop_group "$eval_b_pgid"; stop_group "$eval_a_vllm_pgid"; stop_group "$eval_b_vllm_pgid"; stop_group "$trainer_pgid"; stop_group "$vllm_pgid"
  [[ "$code" -eq 0 ]] || set_status failed "exit=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM
wait_health() {
  local pid=$1 port=$2 label=$3
  for _ in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || die "$label vLLM exited before readiness"
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  die "$label vLLM readiness timed out"
}
wait_idle_pair() {
  for _ in $(seq 1 180); do gpu_idle "$EVAL_GPU_A" && gpu_idle "$EVAL_GPU_B" && return 0; sleep 2; done
  die "evaluation GPU pair did not become idle"
}

set_status starting_training "trainer_gpu=$TRAIN_GPU rollout_gpu=$VLLM_GPU"
gpu_idle "$TRAIN_GPU" || die "training GPU$TRAIN_GPU is busy"
gpu_idle "$VLLM_GPU" || die "rollout GPU$VLLM_GPU is busy"
curl -fsS "http://127.0.0.1:$TRAIN_VLLM_PORT/health" >/dev/null 2>&1 && die "training vLLM port is occupied: $TRAIN_VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table MODEL_PATH="$MODEL_PATH" VLLM_PORT="$TRAIN_VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" >"$RUN_ROOT/logs/train_vllm.log" 2>&1 &
vllm_pgid=$!
wait_health "$vllm_pgid" "$TRAIN_VLLM_PORT" training

set_status training "four-level SAAM Gate60: 60 records, 30x8 per update, 4 updates, 2 passes"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$TRAIN_VLLM_PORT" VLLM_GROUP_PORT="$TRAIN_VLLM_GROUP_PORT" PYTHONPATH= bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" --experiment-config "$CONFIG" --credit-assignment saam-asymmetric-error --result-reward-profile four-level --kl-beta 0 --no-record-gradient-conflicts --save-steps 1 --save-total-limit 4 --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" --seed 20260902 >"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
if ! wait "$trainer_pgid"; then die "RL trainer failed; see $RUN_ROOT/logs/train.log"; fi
trainer_pgid=""; stop_group "$vllm_pgid"; vllm_pgid=""
[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || die "training completed without immutable manifests"
[[ -f "$TRAIN_OUT/checkpoint-4/adapter_model.safetensors" && -f "$TRAIN_OUT/final/adapter_model.safetensors" && -f "$TRAIN_OUT/final/adapter_config.json" ]] || die "training completed without final adapter artifacts"

set_status preparing_evaluation "creating two disjoint example-index shards"
SHARD_DIR="$EVAL_ROOT/shards"; RESULT_A="$EVAL_ROOT/gpu${EVAL_GPU_A}"; RESULT_B="$EVAL_ROOT/gpu${EVAL_GPU_B}"
mkdir -p "$SHARD_DIR" "$RESULT_A" "$RESULT_B"
"$PYTHON" "$MAKE_SHARDS" --examples "$EVAL_INPUT" --first "$SHARD_DIR/gpu${EVAL_GPU_A}.jsonl" --second "$SHARD_DIR/gpu${EVAL_GPU_B}.jsonl" --expected "$EXPECTED_EVAL_RECORDS"

start_eval_vllm() {
  local gpu=$1 port=$2 log=$3
  setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn "$PYTHON" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --served-model-name saam-fourlevel-final --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 24 --gpu-memory-utilization 0.90 --generation-config vllm --enable-lora --lora-modules "saam-fourlevel-final=$TRAIN_OUT/final" --max-lora-rank 64 >"$log" 2>&1 &
}
start_eval_worker() {
  local gpu=$1 port=$2 shard=$3 shard_size=$4 result=$5 log=$6
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$EVAL_RUNTIME" TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" EVAL_ENABLE_THINKING=1 FORMAL_TOOL_EXECUTION_TIMEOUT_SECONDS=10 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u "$EVAL_ENTRY" --base-url "http://127.0.0.1:$port/v1" --model saam-fourlevel-final --examples-json "$shard" --allow-eval-tasks --n "$shard_size" --result-dir "$result" --workers 24 --sample-workers 1 --first-sample-workers 0 --max-inflight-requests 24 --max-steps 30 --max-tokens 2048 --n-samples 1 --pass-k 1 --sample-detail full --summary-every 10 --temperature 0 --top-p 1 --api-retries 3 --few-shot 0 --context-mode rolling-legal-history --history-turns 4 --tool-execution-timeout-seconds 10 --server-config-id qwen3-v26-saam-fourlevel-dp-v1 --denotation-comparison bird-set >"$log" 2>&1 &
}

wait_idle_pair
set_status evaluating "two GPU data-parallel BIRD-dev pass@1"
start_eval_vllm "$EVAL_GPU_A" "$EVAL_PORT_A" "$RUN_ROOT/logs/eval_gpu${EVAL_GPU_A}_vllm.log"
eval_a_vllm_pgid=$!
start_eval_vllm "$EVAL_GPU_B" "$EVAL_PORT_B" "$RUN_ROOT/logs/eval_gpu${EVAL_GPU_B}_vllm.log"
eval_b_vllm_pgid=$!
wait_health "$eval_a_vllm_pgid" "$EVAL_PORT_A" "evaluation GPU$EVAL_GPU_A"; wait_health "$eval_b_vllm_pgid" "$EVAL_PORT_B" "evaluation GPU$EVAL_GPU_B"
SHARD_A_SIZE=$(( (EXPECTED_EVAL_RECORDS + 1) / 2 )); SHARD_B_SIZE=$(( EXPECTED_EVAL_RECORDS / 2 ))
start_eval_worker "$EVAL_GPU_A" "$EVAL_PORT_A" "$SHARD_DIR/gpu${EVAL_GPU_A}.jsonl" "$SHARD_A_SIZE" "$RESULT_A" "$RUN_ROOT/logs/eval_gpu${EVAL_GPU_A}.log"
eval_a_pgid=$!
start_eval_worker "$EVAL_GPU_B" "$EVAL_PORT_B" "$SHARD_DIR/gpu${EVAL_GPU_B}.jsonl" "$SHARD_B_SIZE" "$RESULT_B" "$RUN_ROOT/logs/eval_gpu${EVAL_GPU_B}.log"
eval_b_pgid=$!
if ! wait "$eval_a_pgid"; then die "evaluation shard GPU$EVAL_GPU_A failed"; fi
eval_a_pgid=""
if ! wait "$eval_b_pgid"; then die "evaluation shard GPU$EVAL_GPU_B failed"; fi
eval_b_pgid=""; stop_group "$eval_a_vllm_pgid"; eval_a_vllm_pgid=""; stop_group "$eval_b_vllm_pgid"; eval_b_vllm_pgid=""

set_status merging "validating shard coverage and writing one ordered result"
"$PYTHON" "$MERGE_SHARDS" --examples "$EVAL_INPUT" --shard "$RESULT_A/all.jsonl" --shard "$RESULT_B/all.jsonl" --output "$EVAL_ROOT/final" --adapter "$TRAIN_OUT/final" --expected "$EXPECTED_EVAL_RECORDS"
set_status complete "training=$TRAIN_OUT evaluation=$EVAL_ROOT/final"
printf 'PIPELINE_COMPLETE training=%s evaluation=%s\n' "$TRAIN_OUT" "$EVAL_ROOT/final"
