#!/usr/bin/env bash
# Three-A100 launcher for the version26 online RL experiment:
# two FSDP trainer ranks plus one dedicated online-vLLM card.
# Default mode is preflight; pass "run" explicitly to start 200 batch-14 updates.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
# A direct 4<->5 NCCL probe on this host hung in the first collective while
# the SHM path completed. Keep production fail-closed: topology diagnostics
# may opt into P2P in a separately named probe, but an inherited
# NCCL_P2P_DISABLE=0 must not turn a training launch into an unbounded hang.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Keep asynchronous NCCL failures fail-fast without changing the heartbeat
# threshold or hiding a stalled collective.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-1}
unset PYTORCH_CUDA_ALLOC_CONF

# Keep the normal run quiet. Enable this only for a bounded reproduction when
# a fresh NCCL flight-recorder artifact is required.
if [[ "${RL_NCCL_DIAGNOSTICS:-0}" == "1" ]]; then
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL}
  export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-4096}
  export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
  export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
fi

MODE=${1:-preflight}
[[ "$MODE" == preflight || "$MODE" == run ]] || {
  printf 'usage: %s [preflight|run]\n' "$0" >&2
  exit 2
}

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/data2/shared/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
COHORT_DIR=${COHORT_DIR:-$RUNTIME/data/rl_inputs/qwen3_v26_saam700_20260902}
TASKS=${TASKS:-$COHORT_DIR/qwen3_8b_atomic_v26_saam_fourlevel_train700_v1.jsonl}
COHORT_MANIFEST=${COHORT_MANIFEST:-$COHORT_DIR/qwen3_8b_atomic_v26_saam_fourlevel_train700_v1.manifest.json}
MANUAL_AUDIT=${MANUAL_AUDIT:-$COHORT_DIR/manual_homogeneous100_audit.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700.yaml}
PREFLIGHT=${PREFLIGHT:-$RUNTIME/src/rl/diagnostics/audit_qwen3_v26_saam700.py}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_fourlevel_batch14_700_a100_20260903}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train700_four_epochs_seed20260829}
# Leave placement empty by default: the launcher selects three currently idle
# cards (two trainer + one vLLM) and records the choice in the run status/manifest.
TRAIN_GPUS=${TRAIN_GPUS:-}
VLLM_GPU=${VLLM_GPU:-}
VLLM_PORT=${VLLM_PORT:-8214}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51314}
MASTER_PORT=${MASTER_PORT:-29770}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-14}
OPTIMIZER_STEPS=${OPTIMIZER_STEPS:-200}
GROUP_SIZE=${GROUP_SIZE:-8}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-8}
TRANSITION_MICRO_BATCH_TOKENS=${TRANSITION_MICRO_BATCH_TOKENS:-32768}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-700}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/checkpoint-6380 adapter\n' >&2
  exit 3
}
[[ -d "$PROTOCOL_RUNTIME" && -f "$TASKS" && -f "$COHORT_MANIFEST" ]] || {
  printf 'missing protocol runtime or frozen cohort files\n' >&2
  exit 3
}
[[ -f "$MANUAL_AUDIT" && -f "$CONFIG" && -f "$PREFLIGHT" ]] || {
  printf 'missing manual audit, experiment config, or preflight script\n' >&2
  exit 3
}
[[ "$PROMPTS_PER_UPDATE" -eq 14 && "$OPTIMIZER_STEPS" -eq 200 ]] || {
  printf 'frozen schedule requires 14 prompts/update and 200 updates\n' >&2
  exit 2
}
[[ "$GROUP_SIZE" -eq 8 && "$EXPECTED_RECORDS" -eq 700 ]] || {
  printf 'frozen cohort requires 700 records and K=8\n' >&2
  exit 2
}
[[ "$TRANSITION_MICRO_BATCH_SIZE" -eq 8 && "$TRANSITION_MICRO_BATCH_TOKENS" -eq 32768 ]] || {
  printf 'formal A100 packing requires max_rows=8 and token_budget=32768\n' >&2
  exit 2
}

"$PYTHON" "$PREFLIGHT" \
  --tasks "$TASKS" \
  --manifest "$COHORT_MANIFEST" \
  --manual-audit "$MANUAL_AUDIT" \
  --experiment-config "$CONFIG" \
  --adapter-path "$ADAPTER_PATH"
[[ "$MODE" == run ]] || exit 0

[[ ! -e "$RUN_ROOT" ]] || {
  printf 'refusing existing run root: %s\n' "$RUN_ROOT" >&2
  exit 3
}

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}

gpu_ids=()
while IFS= read -r gpu; do
  [[ "$gpu" =~ ^[0-9]+$ ]] && gpu_ids+=("$gpu")
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
(( ${#gpu_ids[@]} >= 3 )) || {
  printf 'A100 launcher needs at least three visible GPUs\n' >&2
  exit 75
}

if [[ -z "$TRAIN_GPUS" ]]; then
  idle_gpus=()
  for gpu in "${gpu_ids[@]}"; do
    gpu_idle "$gpu" && idle_gpus+=("$gpu")
  done
  (( ${#idle_gpus[@]} >= 3 )) || {
    printf 'fewer than three idle GPUs are available for two trainers plus vLLM\n' >&2
    exit 75
  }
  TRAIN_GPUS="${idle_gpus[0]},${idle_gpus[1]}"
  [[ -n "$VLLM_GPU" ]] || VLLM_GPU="${idle_gpus[2]}"
fi
[[ "$TRAIN_GPUS" =~ ^[0-9]+,[0-9]+$ ]] || {
  printf 'TRAIN_GPUS must contain exactly two comma-separated ids\n' >&2
  exit 2
}
IFS=, read -r TRAIN_GPU0 TRAIN_GPU1 <<<"$TRAIN_GPUS"
if [[ -z "$VLLM_GPU" ]]; then
  for gpu in "${gpu_ids[@]}"; do
    if [[ "$gpu" != "$TRAIN_GPU0" && "$gpu" != "$TRAIN_GPU1" ]] && gpu_idle "$gpu"; then
      VLLM_GPU="$gpu"
      break
    fi
  done
fi
[[ "$VLLM_GPU" =~ ^[0-9]+$ ]] || {
  printf 'no idle vLLM GPU was selected\n' >&2
  exit 75
}
[[ "$TRAIN_GPU0" != "$TRAIN_GPU1" && "$TRAIN_GPU0" != "$VLLM_GPU" && "$TRAIN_GPU1" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct\n' >&2
  exit 2
}
gpu_idle "$TRAIN_GPU0" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU0" >&2; exit 75; }
gpu_idle "$TRAIN_GPU1" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU1" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && {
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
} || true

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

set_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >"$RUN_ROOT/status.next"
  mv "$RUN_ROOT/status.next" "$RUN_ROOT/status"
}
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "${trainer_pgid:-}"
  stop_group "${vllm_pgid:-}"
  [[ "$code" -eq 0 ]] || set_status failed "exit=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

vllm_pgid=''
trainer_pgid=''
set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT util=$VLLM_GPU_MEMORY_UTILIZATION"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; exit 1; }

# A different user can claim a trainer card while vLLM is loading. Recheck
# immediately before torchrun so we never attach to a newly occupied GPU.
gpu_idle "$TRAIN_GPU0" || { printf 'trainer GPU%s became busy during vLLM startup\n' "$TRAIN_GPU0" >&2; exit 75; }
gpu_idle "$TRAIN_GPU1" || { printf 'trainer GPU%s became busy during vLLM startup\n' "$TRAIN_GPU1" >&2; exit 75; }

set_status training "records=700 epochs=4 updates=200 prompts_per_update=14 k=8 lr=4e-7 credit=saam-asymmetric-error reward=four-level span_alpha=0.5 fsdp_world_size=2 train_gpus=$TRAIN_GPUS max_rows=8 token_budget=32768"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" MASTER_ADDR=127.0.0.1 MASTER_PORT="$MASTER_PORT" \
  TABLE_RL_TRAINER_SHARDING=fsdp TABLE_RL_FSDP_BASE_STORAGE=bf16 PYTHONPATH= \
  NCCL_P2P_DISABLE=1 \
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  --master_port "$MASTER_PORT" "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" \
  --trainer-sharding fsdp --fsdp-base-storage bf16 \
  --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" \
  --examples-json "$TASKS" --output-dir "$TRAIN_OUT" --experiment-config "$CONFIG" \
  --vllm-host 127.0.0.1 --vllm-port "$VLLM_PORT" --vllm-group-port "$VLLM_GROUP_PORT" \
  --optimizer-steps "$OPTIMIZER_STEPS" --ppo-iterations 1 \
  --prompts-per-update "$PROMPTS_PER_UPDATE" --group-size "$GROUP_SIZE" \
  --reward-mode result-only --result-reward-profile four-level \
  --credit-assignment saam-asymmetric-error --error-penalty 1.0 \
  --policy-reduction trajectory_token_mean --span-balance-alpha 0.5 \
  --kl-beta 0 --no-record-gradient-conflicts \
  --protocol-runtime-root "$PROTOCOL_RUNTIME" \
  --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" \
  --transition-micro-batch-tokens "$TRANSITION_MICRO_BATCH_TOKENS" \
  --expected-records "$EXPECTED_RECORDS" --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" --seed 20260829 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=''
stop_group "$vllm_pgid"
vllm_pgid=''

[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || {
  printf 'training exited without immutable manifests\n' >&2
  exit 1
}
set_status complete "final=$TRAIN_OUT/final"
trap - EXIT INT TERM
