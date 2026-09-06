#!/usr/bin/env bash
# Isolated scheme-2 smoke: two A100s jointly train one FSDP-sharded BF16 actor
# and a third A100 runs the dedicated vLLM server. The active TRL transition
# objective and version26 protocol remain unchanged; only trainer sharding and
# GPU placement differ from the DDP A/B launcher.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TABLE_RL_TRAINER_SHARDING=fsdp
export TABLE_RL_FSDP_BASE_STORAGE=bf16
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# On this host, FSDP's large nested all-gather intermittently hangs with the
# PCIe P2P path (the two-card probe timed out at 600 s). Shared-memory NCCL is
# healthy and completes the same collective, so disable P2P by default. Set
# NCCL_P2P_DISABLE=0 explicitly only for a separately audited topology test.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-LOC}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
unset PYTORCH_CUDA_ALLOC_CONF

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/data2/shared/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$OUTPUT_ROOT/qwen3_8b_v26_speed_ab_mb8_20260902/tasks14_first_update.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700.yaml}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_v26_speed_fsdp_ab_20260902}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train}
TRAIN_GPUS=${TRAIN_GPUS:-5,6}
VLLM_GPU=${VLLM_GPU:-7}
VLLM_PORT=${VLLM_PORT:-8230}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51330}
MASTER_PORT=${MASTER_PORT:-29530}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
MAX_ROWS=${MAX_ROWS:-8}
TOKEN_BUDGET=${TOKEN_BUDGET:-24576}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/adapter\n' >&2
  exit 3
}
[[ -f "$TASKS" && -f "$CONFIG" && -d "$PROTOCOL_RUNTIME" ]] || {
  printf 'missing tasks/config/protocol runtime\n' >&2
  exit 3
}
[[ "$TRAIN_GPUS" =~ ^[0-9]+,[0-9]+$ ]] || {
  printf 'TRAIN_GPUS must contain exactly two comma-separated GPU ids\n' >&2
  exit 2
}
IFS=, read -r TRAIN_GPU0 TRAIN_GPU1 <<<"$TRAIN_GPUS"
[[ "$TRAIN_GPU0" != "$TRAIN_GPU1" && "$TRAIN_GPU0" != "$VLLM_GPU" && "$TRAIN_GPU1" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct\n' >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || {
  printf 'refusing existing run root: %s\n' "$RUN_ROOT" >&2
  exit 3
}

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}

set_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >"$RUN_ROOT/status.next"
  mv "$RUN_ROOT/status.next" "$RUN_ROOT/status"
}

stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
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

gpu_idle "$TRAIN_GPU0" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU0" >&2; exit 75; }
gpu_idle "$TRAIN_GPU1" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU1" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && {
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
} || true

vllm_pgid=''; trainer_pgid=''
set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT util=$VLLM_GPU_MEMORY_UTILIZATION eager=$VLLM_ENFORCE_EAGER train_gpus=$TRAIN_GPUS sharding=fsdp"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
  VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; exit 1; }

set_status training "records=14 prompts=14 k=8 updates=1 fsdp_world_size=2 max_rows=$MAX_ROWS token_budget=$TOKEN_BUDGET base_storage=bf16"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" MASTER_ADDR=127.0.0.1 MASTER_PORT="$MASTER_PORT" \
  TABLE_RL_TRAINER_SHARDING=fsdp TABLE_RL_FSDP_BASE_STORAGE=bf16 PYTHONPATH= \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port "$MASTER_PORT" \
  "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" \
  --trainer-sharding fsdp --fsdp-base-storage bf16 \
  --experiment-config "$CONFIG" --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" \
  --examples-json "$TASKS" --output-dir "$TRAIN_OUT" --expected-records 14 \
  --protocol-runtime-root "$PROTOCOL_RUNTIME" --vllm-host 127.0.0.1 \
  --vllm-port "$VLLM_PORT" --vllm-group-port "$VLLM_GROUP_PORT" \
  --optimizer-steps 1 --ppo-iterations 1 --prompts-per-update 14 --group-size 8 \
  --credit-assignment saam-asymmetric-error --result-reward-profile four-level \
  --policy-reduction trajectory_token_mean --error-penalty 1.0 --kl-beta 0 \
  --optimizer-name adamw_torch --learning-rate 4e-7 --weight-decay 0.1 \
  --transition-micro-batch-size "$MAX_ROWS" --transition-micro-batch-tokens "$TOKEN_BUDGET" \
  --max-agent-steps 30 --max-batch-calls 5 --max-new-tokens 2048 \
  --max-context-tokens 16384 --history-turns 4 --temperature 0.8 --top-p 1.0 \
  --top-k 0 --enable-thinking --save-steps 1 --save-total-limit 1 --seed 20260829 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=''
stop_group "$vllm_pgid"; vllm_pgid=''

[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || {
  printf 'training exited without immutable manifests\n' >&2
  exit 1
}
set_status complete "final=$TRAIN_OUT/final"
trap - EXIT INT TERM
