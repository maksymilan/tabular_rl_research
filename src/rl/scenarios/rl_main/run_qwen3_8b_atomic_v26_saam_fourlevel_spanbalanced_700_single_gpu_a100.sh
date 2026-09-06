#!/usr/bin/env bash
# Single-A100 online RL experiment for the version26 screened cohort:
# one replicated trainer process plus one dedicated online-vLLM card.
# The trainer and online-vLLM process use two distinct A100 cards.
# Default mode is preflight; pass "gate" for one live update or "run" for 200 updates.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Shared resource/process behavior lives in one module so every RL launcher
# gets the same GPU checks, health wait, status writes, and cleanup semantics.
source "$SCRIPT_DIR/../../frameworks/launcher/launch_common.sh"
# TRL's server-mode vLLM still creates a small NCCL communicator between the
# trainer and serving process, even when the trainer itself has world_size=1.
# Use the same SHM-safe transport as the validated A100 FSDP path.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
unset PYTORCH_CUDA_ALLOC_CONF

MODE=${1:-preflight}
[[ "$MODE" == preflight || "$MODE" == gate || "$MODE" == run ]] || {
  printf 'usage: %s [preflight|gate|run]\n' "$0" >&2
  exit 2
}

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_a100_optimized_20260903_r2}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/data2/shared/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
COHORT_DIR=${COHORT_DIR:-$RUNTIME/data/rl_inputs/qwen3_v26_saam_screened500_20260906}
TASKS=${TASKS:-$COHORT_DIR/qwen3_8b_atomic_v26_saam_screened500_v1.jsonl}
COHORT_MANIFEST=${COHORT_MANIFEST:-$COHORT_DIR/qwen3_8b_atomic_v26_saam_screened500_v1.manifest.json}
MANUAL_AUDIT=${MANUAL_AUDIT:-$COHORT_DIR/screened500_audit.jsonl}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml}
PREFLIGHT=${PREFLIGHT:-$RUNTIME/src/rl/scenarios/diagnostics/audit_qwen3_v26_saam700.py}
DEFAULT_RUN_ROOT=$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_screened500_batch30_single_gpu_a100_20260906_r1
[[ "$MODE" == gate ]] && DEFAULT_RUN_ROOT=${DEFAULT_RUN_ROOT}_gate
RUN_ROOT=${RUN_ROOT:-$DEFAULT_RUN_ROOT}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train700_single_gpu_seed20260829}

# Leave placement empty by default. The launcher chooses exactly two idle cards
# and records the choice; it never stops or attaches to another user's process.
TRAIN_GPU=${TRAIN_GPU:-}
VLLM_GPU=${VLLM_GPU:-}
VLLM_PORT=${VLLM_PORT:-8215}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51315}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-30}
RL_MECHANISM=${RL_MECHANISM:-saam-asymmetric-error}
DEFAULT_OPTIMIZER_STEPS=200
[[ "$MODE" == gate ]] && DEFAULT_OPTIMIZER_STEPS=1
OPTIMIZER_STEPS=${OPTIMIZER_STEPS:-$DEFAULT_OPTIMIZER_STEPS}
GROUP_SIZE=${GROUP_SIZE:-8}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-500}
SAVE_STEPS=${SAVE_STEPS:-10}
# The formal 200-update run saves at most 20 checkpoints when SAVE_STEPS=10.
# Keep a generous limit so Trainer does not delete earlier checkpoints while
# preserving the positive-integer interface required by run_transition_grpo.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1000}
REPLICATED_BASE_STORAGE=${REPLICATED_BASE_STORAGE:-bf16}
OPTIMIZER_NAME=${OPTIMIZER_NAME:-adamw_torch}

# Conservative first gate for a full, unsharded BF16 base on one 40GB card.
# These only change how the same loss is accumulated; the launcher accepts the
# formal 8/32768 values after a successful local gate if more throughput is safe.
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-4}
TRANSITION_MICRO_BATCH_TOKENS=${TRANSITION_MICRO_BATCH_TOKENS:-16384}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/checkpoint-6380 adapter\n' >&2
  exit 3
}
[[ -d "$PROTOCOL_RUNTIME" && -f "$TASKS" && -f "$COHORT_MANIFEST" ]] || {
  printf 'missing protocol runtime or frozen cohort files\n' >&2
  exit 3
}
[[ -f "$PROTOCOL_RUNTIME/src/sft/protocol.py" ]] || {
  printf 'Atomic version26 runtime protocol.py is missing\n' >&2
  exit 3
}
grep -Eq 'PROTOCOL_VERSION[[:space:]]*=[[:space:]]*"version26"' \
  "$PROTOCOL_RUNTIME/src/sft/protocol.py" || {
  printf 'refusing non-version26 protocol runtime\n' >&2
  exit 2
}
[[ -f "$MANUAL_AUDIT" && -f "$CONFIG" && -f "$PREFLIGHT" ]] || {
  printf 'missing manual audit, experiment config, or preflight script\n' >&2
  exit 3
}
[[ "$PROMPTS_PER_UPDATE" -eq 30 ]] || {
  printf 'screened cohort schedule requires 30 prompts/update\n' >&2
  exit 2
}
if [[ "$MODE" == run && "$OPTIMIZER_STEPS" -ne 200 ]]; then
  printf 'formal run requires 200 updates\n' >&2
  exit 2
fi
if [[ "$MODE" == gate && "$OPTIMIZER_STEPS" -ne 1 ]]; then
  printf 'single-GPU gate requires exactly 1 update\n' >&2
  exit 2
fi
[[ "$GROUP_SIZE" -eq 8 && "$EXPECTED_RECORDS" -eq 500 ]] || {
  printf 'screened cohort requires 500 records and K=8\n' >&2
  exit 2
}
[[ "$REPLICATED_BASE_STORAGE" == bf16 || "$REPLICATED_BASE_STORAGE" == 4bit ]] || {
  printf 'REPLICATED_BASE_STORAGE must be bf16 or 4bit\n' >&2
  exit 2
}
[[ "$OPTIMIZER_NAME" == adamw_torch || "$OPTIMIZER_NAME" == paged_adamw_8bit ]] || {
  printf 'OPTIMIZER_NAME must be adamw_torch or paged_adamw_8bit\n' >&2
  exit 2
}

"$PYTHON" "$PREFLIGHT" \
  --tasks "$TASKS" \
  --manifest "$COHORT_MANIFEST" \
  --manual-audit "$MANUAL_AUDIT" \
  --experiment-config "$CONFIG" \
  --adapter-path "$ADAPTER_PATH"
if [[ "$MODE" == preflight ]]; then
  exit 0
fi

[[ ! -e "$RUN_ROOT" ]] || {
  printf 'refusing existing run root: %s\n' "$RUN_ROOT" >&2
  exit 3
}

gpu_ids=(0 1 2 3)
for gpu in "${gpu_ids[@]}"; do
  nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || {
    printf 'A100 launcher requires visible allowlisted GPU%s\n' "$gpu" >&2
    exit 75
  }
done
(( ${#gpu_ids[@]} >= 2 )) || {
  printf 'single-card fallback needs at least two visible GPUs\n' >&2
  exit 75
}

if [[ -z "$TRAIN_GPU" && -z "$VLLM_GPU" ]]; then
  idle_gpus=()
  for gpu in "${gpu_ids[@]}"; do
    gpu_idle "$gpu" && idle_gpus+=("$gpu")
  done
  (( ${#idle_gpus[@]} >= 2 )) || {
    printf 'fewer than two idle GPUs are available for trainer plus vLLM\n' >&2
    exit 75
  }
  TRAIN_GPU="${idle_gpus[0]}"
  VLLM_GPU="${idle_gpus[1]}"
elif [[ -z "$TRAIN_GPU" ]]; then
  for gpu in "${gpu_ids[@]}"; do
    if [[ "$gpu" != "$VLLM_GPU" ]] && gpu_idle "$gpu"; then
      TRAIN_GPU="$gpu"
      break
    fi
  done
elif [[ -z "$VLLM_GPU" ]]; then
  for gpu in "${gpu_ids[@]}"; do
    if [[ "$gpu" != "$TRAIN_GPU" ]] && gpu_idle "$gpu"; then
      VLLM_GPU="$gpu"
      break
    fi
  done
fi

[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ ]] || {
  printf 'could not select numeric trainer and vLLM GPUs\n' >&2
  exit 75
}
[[ "$TRAIN_GPU" -ge 0 && "$TRAIN_GPU" -le 3 && "$VLLM_GPU" -ge 0 && "$VLLM_GPU" -le 3 ]] || {
  printf 'A100 launcher only permits GPU ids 0-3\n' >&2
  exit 2
}
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct\n' >&2
  exit 2
}
gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && {
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
} || true

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

install_process_cleanup_trap "$RUN_ROOT"

vllm_pgid=''
trainer_pgid=''
set_status "$RUN_ROOT" starting_vllm "topology=single_gpu_replicated vllm_gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
wait_for_health "$vllm_pgid" "http://127.0.0.1:$VLLM_PORT/health" 300 2 || {
  printf 'vLLM readiness failed\n' >&2
  exit 1
}

# Recheck after vLLM startup: another user may have claimed the trainer card
# while the large model was loading.
gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s became busy during vLLM startup\n' "$TRAIN_GPU" >&2; exit 75; }

set_status "$RUN_ROOT" training "records=$EXPECTED_RECORDS updates=$OPTIMIZER_STEPS prompts_per_update=$PROMPTS_PER_UPDATE k=8 topology=single_gpu_replicated trainer_gpu=$TRAIN_GPU vllm_gpu=$VLLM_GPU base_storage=$REPLICATED_BASE_STORAGE optimizer=$OPTIMIZER_NAME max_rows=$TRANSITION_MICRO_BATCH_SIZE token_budget=$TRANSITION_MICRO_BATCH_TOKENS"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONPATH= \
  TABLE_RL_TRAINER_SHARDING=replicated TABLE_RL_REPLICATED_BASE_STORAGE="$REPLICATED_BASE_STORAGE" \
  "$PYTHON" "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" \
  --trainer-sharding replicated --replicated-base-storage "$REPLICATED_BASE_STORAGE" \
  --optimizer-name "$OPTIMIZER_NAME" \
  --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" \
  --examples-json "$TASKS" --output-dir "$TRAIN_OUT" --experiment-config "$CONFIG" \
  --vllm-host 127.0.0.1 --vllm-port "$VLLM_PORT" --vllm-group-port "$VLLM_GROUP_PORT" \
  --optimizer-steps "$OPTIMIZER_STEPS" --ppo-iterations 1 \
  --prompts-per-update "$PROMPTS_PER_UPDATE" --group-size "$GROUP_SIZE" \
  --reward-mode result-only --result-reward-profile four-level \
  --credit-assignment "$RL_MECHANISM" --error-penalty 1.0 \
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
set_status "$RUN_ROOT" complete "final=$TRAIN_OUT/final"
trap - EXIT INT TERM
