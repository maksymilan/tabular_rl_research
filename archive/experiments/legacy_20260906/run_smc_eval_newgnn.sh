#!/usr/bin/env bash
# NewGNN-side evaluation queue for the isolated SMC checkpoint.
# It uses the existing remote_eval_supervisor bundle, waits for a genuinely
# idle NewGNN GPU, and starts a fresh adapter-serving process only after the
# current vLLM/evaluation workloads no longer occupy the selected GPU.
set -euo pipefail

ADAPTER=${1:?usage: $0 /home/dengyan/tabular_rl_outputs/.../checkpoint-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26/qwen3_v26_smc_mode_concentration_gate60_dev1534_20260906}
DATAPARALLEL=${DATAPARALLEL:-$OUTPUT_ROOT/qwen3_v26_smc_mode_concentration_gate60_table_rl_20260906/run_qwen3_v26_smc_dataparallel_newgnn.sh}
CONTROLLER_ROOT=${CONTROLLER_ROOT:-$OUTPUT_ROOT/qwen3_v26_smc_mode_concentration_gate60_table_rl_20260906/controller}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
SOURCE=${SOURCE:-$OUTPUT_ROOT/eval_inputs/qwen3_atomic_v26/bird_dev_20240627.jsonl}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
MODEL=${MODEL:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
PORT=${PORT:-8114}
POLL_SECONDS=${POLL_SECONDS:-60}
LOG_FILE=${LOG_FILE:-$RUN_DIR/queue.log}

mkdir -p "$(dirname "$RUN_DIR")"
log() { mkdir -p "$(dirname "$LOG_FILE")"; printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"; }
die() { log "blocked: $*"; exit 3; }

[[ -f "$ADAPTER/trainer_state.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || die "adapter checkpoint is incomplete: $ADAPTER"
[[ -f "$DATAPARALLEL" && -f "$CONTROLLER_ROOT/formal_v26_rollout_passk.py" && \
   -f "$CONTROLLER_ROOT/make_eval_shards.py" && -f "$CONTROLLER_ROOT/merge_eval_shards.py" && \
   -f "$SOURCE" && -d "$RUNTIME" && -d "$MODEL" ]] || die "missing NewGNN evaluation asset"
[[ ! -e "$RUN_DIR" ]] || die "refusing to reuse evaluation run: $RUN_DIR"

gpu_idle() {
  local gpu=$1 used pids
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}

selected_gpus=()
while true; do
  for gpu in $(seq 0 7); do
    if gpu_idle "$gpu"; then
      selected_gpus+=("$gpu")
      ((${#selected_gpus[@]} == 2)) && break 2
    fi
  done
  selected_gpus=()
  log "waiting for two idle NewGNN GPUs"
  sleep "$POLL_SECONDS"
done

mkdir -p "$RUN_DIR/controller"
cp "$CONTROLLER_ROOT/formal_v26_rollout_passk.py" "$RUN_DIR/controller/formal_v26_rollout_passk.py"
cp "$CONTROLLER_ROOT/make_eval_shards.py" "$RUN_DIR/controller/make_eval_shards.py"
cp "$CONTROLLER_ROOT/merge_eval_shards.py" "$RUN_DIR/controller/merge_eval_shards.py"
log "starting NewGNN data-parallel matched greedy evaluation on GPUs ${selected_gpus[*]}"
nohup env \
  PYTHON_BIN="$PYTHON" \
  RUNTIME="$RUNTIME" \
  MODEL="$MODEL" \
  SOURCE_INPUT="$SOURCE" \
  ADAPTER="$ADAPTER" \
  SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$OUTPUT_ROOT/checkpoints/checkpoint-6380}" \
  CHECKPOINT_GLOBAL_STEP="${CHECKPOINT_GLOBAL_STEP:-4}" \
  RUN_DIR="$RUN_DIR" \
  GPU0="${selected_gpus[0]}" \
  GPU1="${selected_gpus[1]}" \
  PORT0=8114 \
  PORT1=8115 \
  WRAPPER="$RUN_DIR/controller/formal_v26_rollout_passk.py" \
  SHARDER="$RUN_DIR/controller/make_eval_shards.py" \
  MERGER="$RUN_DIR/controller/merge_eval_shards.py" \
  bash "$DATAPARALLEL" \
  >"$RUN_DIR/supervisor.log" 2>&1 &
echo "$!" >"$RUN_DIR/queue_supervisor.pid"
log "data-parallel evaluation supervisor started pid=$! run_dir=$RUN_DIR gpus=${selected_gpus[*]}"
