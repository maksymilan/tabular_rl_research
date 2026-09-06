#!/usr/bin/env bash
# 4k SFT -> SAAM four-level RL gate: 60 BIRD-train prompts, 15 prompts/update,
# 8 rollouts/prompt, four optimizer updates.  This is a small RL validation
# run; BIRD-dev evaluation is performed only after training by a separate
# handoff script.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_diagnostic_current_20260904}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/checkpoint-6380}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_gate60_15.yaml}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_v26_4k_rl_saam_fourlevel_gate60_15_table_rl_20260905}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8235}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51335}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-16384}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
for path in "$PYTHON" "$RUNTIME" "$MODEL_PATH" "$ADAPTER_PATH" "$PROTOCOL_RUNTIME" "$TASKS" "$CONFIG"; do
  [[ -e "$path" ]] || die "missing path: $path"
done
[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" != "$VLLM_GPU" ]] || die "trainer/vLLM GPU ids must differ"
[[ ! -e "$RUN_ROOT" ]] || die "refusing existing run root: $RUN_ROOT"
mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || die "launcher already running"
exec > >(tee "$RUN_ROOT/logs/launcher.log") 2>&1

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}
stop_group() {
  local pgid=${1:-}; [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
vllm_pgid=""; trainer_pgid=""
cleanup() {
  local code=$?; trap - EXIT INT TERM
  stop_group "$trainer_pgid"; stop_group "$vllm_pgid"
  [[ "$code" -eq 0 ]] || printf '%s\tfailed\texit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >"$RUN_ROOT/status"
  exit "$code"
}
trap cleanup EXIT INT TERM
set_status() { printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >"$RUN_ROOT/status"; }

gpu_idle "$TRAIN_GPU" || die "trainer GPU$TRAIN_GPU is busy"
gpu_idle "$VLLM_GPU" || die "vLLM GPU$VLLM_GPU is busy"
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && die "vLLM port is occupied: $VLLM_PORT" || true

set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
  VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN="$MAX_CONTEXT_TOKENS" \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || die "vLLM readiness failed"

set_status training "records=60 prompts=15 k=8 updates=4 trainer_gpu=$TRAIN_GPU vllm_gpu=$VLLM_GPU"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --experiment-config "$CONFIG" --expected-records 60 \
    --protocol-runtime-root "$PROTOCOL_RUNTIME" \
    --expected-runtime-content-tree-sha256 5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab \
    --expected-protocol-version version26 --expected-protocol-hash 4da19387399bd3a5 \
    --expected-student-prompt-sha256 848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316 \
    --expected-initial-adapter-sha256 8900e4c482f4624ff05059a6fecc2811fa99552cddea7550d44ce8cde853abe6 \
    --expected-reference-adapter-sha256 8900e4c482f4624ff05059a6fecc2811fa99552cddea7550d44ce8cde853abe6 \
    --optimizer-steps 4 --ppo-iterations 1 --prompts-per-update 15 --group-size 8 \
    --credit-assignment saam-asymmetric-error --result-reward-profile four-level \
    --policy-reduction trajectory_token_mean --error-penalty 1.0 --kl-beta 0 \
    --span-balance-alpha 0.5 --optimizer-name adamw_torch --learning-rate 4e-7 \
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --clip-epsilon 0.2 \
    --clip-epsilon-high 0.2 --transition-micro-batch-size 1 --transition-micro-batch-tokens 24576 \
    --max-agent-steps 30 --max-batch-calls 5 --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-context-tokens "$MAX_CONTEXT_TOKENS" --history-turns 4 --temperature 0.8 \
    --top-p 1.0 --top-k 0 --enable-thinking --no-record-gradient-conflicts \
    --save-steps 1 --save-total-limit 4 --seed 20260905 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=""
stop_group "$vllm_pgid"; vllm_pgid=""

[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || die "missing immutable training manifests"
[[ -f "$TRAIN_OUT/rollouts.jsonl" && -f "$TRAIN_OUT/checkpoint-4/adapter_model.safetensors" && -f "$TRAIN_OUT/final/adapter_model.safetensors" ]] || die "missing final training artifacts"
set_status complete "final=$TRAIN_OUT/final"
printf 'GATE60_15_COMPLETE training=%s\n' "$TRAIN_OUT"
