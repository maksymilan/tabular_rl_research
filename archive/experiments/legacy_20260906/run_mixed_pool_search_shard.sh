#!/usr/bin/env bash
# Generate one disjoint K4 candidate-search shard on a verified free GPU.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:?RUNTIME is required}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${ADAPTER_PATH:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
HOST_ROLE=${HOST_ROLE:?HOST_ROLE must be table_rl or newgnn}
HARDWARE_PROFILE=${HARDWARE_PROFILE:-$R/src/rl/configs/hardware/rtx3090_24gb.sh}
test -f "$HARDWARE_PROFILE"
# shellcheck source=../configs/hardware/rtx3090_24gb.sh
source "$HARDWARE_PROFILE"
for name in GPU_ID TASKS TASK_ID_FILE OUTPUT_DIR STATUS RUN_LOG; do
  [[ -n "${!name:-}" ]] || { printf 'missing %s\n' "$name" >&2; exit 2; }
done
if [[ "$HOST_ROLE" == table_rl ]]; then
  [[ "$GPU_ID" == 0 || "$GPU_ID" == 1 ]] || { printf 'table_rl GPU must be 0 or 1\n' >&2; exit 2; }
  FREE_SAMPLES=${FREE_SAMPLES:-1}
  TASK_BATCH_SIZE=${TASK_BATCH_SIZE:-$RTX3090_ROLLOUT_TASK_BATCH_SIZE}
elif [[ "$HOST_ROLE" == newgnn ]]; then
  [[ "$GPU_ID" == 6 || "$GPU_ID" == 7 ]] || { printf 'NewGNN GPU must be 6 or 7\n' >&2; exit 2; }
  FREE_SAMPLES=${FREE_SAMPLES:-$RTX3090_SHARED_HOST_FREE_SAMPLES}
  TASK_BATCH_SIZE=${TASK_BATCH_SIZE:-$RTX3090_ROLLOUT_TASK_BATCH_SIZE}
else
  printf 'unsupported HOST_ROLE=%s\n' "$HOST_ROLE" >&2
  exit 2
fi

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT
mkdir -p "$O/logs" "$OUTPUT_DIR/groups" "$(dirname "$STATUS")" "$(dirname "$RUN_LOG")"
exec 8>"$STATUS.lock"
if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1

[[ "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" == d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e ]]
[[ "$(grep -c '^PROTOCOL_VERSION = "version26"' "$R/src/sft/protocol.py")" -eq 1 ]]

consecutive=0
while [[ "$consecutive" -lt "$FREE_SAMPLES" ]]; do
  used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
  compute_pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
  if [[ "$used" -le "$RTX3090_FREE_MEMORY_THRESHOLD_MIB" && -z "$compute_pids" ]]; then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  set_status waiting_gpu "host=$HOST_ROLE gpu=$GPU_ID mib=$used free_samples=$consecutive/$FREE_SAMPLES compute_pids=${compute_pids:-none}"
  if [[ "$consecutive" -lt "$FREE_SAMPLES" ]]; then sleep "$RTX3090_SHARED_HOST_FREE_POLL_SECONDS"; fi
done

# Recheck at the launch boundary after the continuous-free window.
used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
compute_pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
if [[ "$used" -gt "$RTX3090_FREE_MEMORY_THRESHOLD_MIB" || -n "$compute_pids" ]]; then
  set_status waiting_gpu "launch_recheck_failed gpu=$GPU_ID mib=$used compute_pids=${compute_pids:-none}"
  exit 3
fi

assigned=$(grep -cve '^[[:space:]]*$' "$TASK_ID_FILE")
set_status generating "host=$HOST_ROLE gpu=$GPU_ID tasks=$assigned task_batch_size=$TASK_BATCH_SIZE"
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
  "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
  --model-path "$MODEL" --adapter-path "$SFT2" --tasks "$TASKS" \
  --output-dir "$OUTPUT_DIR" --group-size 4 --temperature 0.7 --top-p 0.95 \
  --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
  --history-turns 4 --seed 101 \
  --gpu-memory-utilization "$RTX3090_ROLLOUT_GPU_MEMORY_UTILIZATION" \
  --task-batch-size "$TASK_BATCH_SIZE" \
  --scheduler "$RTX3090_ROLLOUT_SCHEDULER" \
  --question-window "$RTX3090_ROLLOUT_QUESTION_WINDOW" \
  --task-id-file "$TASK_ID_FILE" --no-finalize

groups=$(find "$OUTPUT_DIR/groups" -maxdepth 1 -name 'bird_train_*.json' | wc -l)
set_status complete "host=$HOST_ROLE gpu=$GPU_ID assigned=$assigned output_groups=$groups"
