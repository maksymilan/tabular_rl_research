#!/usr/bin/env bash
# One-group performance audit on table_rl.  This is intentionally isolated
# from the completed training/evaluation roots and never stops foreign PIDs.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the instrumented source copy}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the audit root}
TRAIN_GPU=${1:?usage: run_perf_audit_pair.sh TRAIN_GPU VLLM_GPU [EXAMPLE_INDEX]}
VLLM_GPU=${2:?usage: run_perf_audit_pair.sh TRAIN_GPU VLLM_GPU [EXAMPLE_INDEX]}
EXAMPLE_INDEX=${3:-7365}
PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
PYTHON="$PYTHON_ENV/bin/python"
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380}
EXAMPLES_JSON=${EXAMPLES_JSON:-$PROJECT_DIR/data_new60.jsonl}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$PROJECT_DIR/src/rl/configs/experiments/qwen3_4b_atomic_v26_signed_vanilla_grpo_c2to6_balanced60_table_rl.yaml}
VLLM_PORT=${VLLM_PORT:-18340}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51440}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MICRO_BATCH_TOKENS=${MICRO_BATCH_TOKENS:-4096}
BASE_STORAGE=${BASE_STORAGE:-4bit}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-default}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
OLD_POLICY_LOGPROB_SOURCE=${OLD_POLICY_LOGPROB_SOURCE:-actor}
OUTPUT_DIR="$RUN_ROOT/train"

export PROJECT_DIR RUN_ROOT PYTHON PYTHON_ENV MODEL_PATH ADAPTER_PATH EXAMPLES_JSON
export EXPERIMENT_CONFIG OUTPUT_DIR VLLM_PORT VLLM_GROUP_PORT
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-$PYTHON_ENV/var/triton-libcuda}
export RL_PERF_TRACE_PATH=${RL_PERF_TRACE_PATH:-$RUN_ROOT/perf_trace.jsonl}
export RL_PERF_TRACE_SYNC_CUDA=${RL_PERF_TRACE_SYNC_CUDA:-1}
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

source "$PROJECT_DIR/src/rl/frameworks/launcher/launch_common.sh"
assert_distinct_allowlisted_gpus "$TRAIN_GPU" "$VLLM_GPU" 1

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || exit 2
exec 8>"/tmp/table-rl-perf-audit-gpu-$TRAIN_GPU.lock"
flock -n 8 || exit 2
exec 7>"/tmp/table-rl-perf-audit-gpu-$VLLM_GPU.lock"
flock -n 7 || exit 2

trainer_pgid=
vllm_pgid=
set_status "$RUN_ROOT" preflight "isolated one-group performance audit"

COMMON_ARGS=(
  --experiment-config "$EXPERIMENT_CONFIG"
  --example-index "$EXAMPLE_INDEX"
  --expected-records 1
  --optimizer-steps 1
  --ppo-iterations 1
  --prompts-per-update 1
  --group-size 8
  --transition-micro-batch-size "$MICRO_BATCH_SIZE"
  --transition-micro-batch-tokens "$MICRO_BATCH_TOKENS"
  --replicated-base-storage "$BASE_STORAGE"
  --trainer-sharding replicated
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --old-policy-logprob-source "$OLD_POLICY_LOGPROB_SOURCE"
  --max-new-tokens 2048
  --max-context-tokens 16384
  --max-agent-steps 30
  --save-steps 1
  --save-total-limit 1
  --kl-beta 0
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_ARGS+=(--gradient-checkpointing)
elif [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
  COMMON_ARGS+=(--no-gradient-checkpointing)
fi

CUDA_VISIBLE_DEVICES="" bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
  --preflight-only "${COMMON_ARGS[@]}" >"$RUN_ROOT/logs/preflight.json"
gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU" || {
  set_status "$RUN_ROOT" waiting_resources "two idle table_rl GPUs required"
  exit 75
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "${trainer_pgid:-}"
  stop_group "${vllm_pgid:-}"
  if [[ "$code" -eq 0 ]]; then
    set_status "$RUN_ROOT" complete_audited "one group traced; owned resources released"
  else
    set_status "$RUN_ROOT" failed "audit exit=$code; owned resources released"
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$RUN_ROOT/logs/gpus_before.csv"
set_status "$RUN_ROOT" starting_server "vLLM gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/start_vllm_server.sh" \
  >"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
printf '%s\n' "$vllm_pgid" >"$RUN_ROOT/vllm.pid"
wait_for_health "$vllm_pgid" "http://127.0.0.1:$VLLM_PORT/health/"
"$PYTHON" - "$VLLM_PORT" <<'PY'
import json
import sys
from rl.frameworks.trl.serving_contract import probe_trl_server
print(json.dumps(probe_trl_server("127.0.0.1", int(sys.argv[1]))))
PY

gpu_idle "$TRAIN_GPU" || exit 75
set_status "$RUN_ROOT" starting_trainer "one group; traced rollout, old-policy, and backward"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
  "${COMMON_ARGS[@]}" \
  >"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
printf '%s\n' "$trainer_pgid" >"$RUN_ROOT/trainer.pid"
wait "$trainer_pgid"
