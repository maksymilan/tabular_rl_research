#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/data/dengyan/tabular_rl_outputs/logs}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/qwen3.5_9b_qlora_v9_ready_4k_epoch4.yaml}
RUN_ID=${RUN_ID:-qwen35_v9_ready_4k_epoch4_sft_$(date +%Y%m%d_%H%M%S)}
LOG=$LOG_DIR/${RUN_ID}.log
LOCK_ROOT=${LOCK_ROOT:-/data/dengyan/tabular_rl_outputs/gpu_locks}

mkdir -p "$LOG_DIR" "$LOCK_ROOT"
cd "$PROJECT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

release_lock() {
  if [[ -n "${LOCK_DIR:-}" && -d "$LOCK_DIR" ]]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}
trap release_lock EXIT INT TERM

select_idle_gpu() {
  local stable_needed=${GPU_STABLE_CHECKS:-3}
  local sleep_s=${GPU_CHECK_INTERVAL:-30}
  local max_mem=${GPU_MAX_IDLE_MEM_MB:-1000}
  local max_util=${GPU_MAX_IDLE_UTIL:-5}

  while true; do
    while IFS=, read -r idx mem util; do
      idx=$(echo "$idx" | xargs)
      mem=$(echo "$mem" | xargs)
      util=$(echo "$util" | xargs)
      [[ -z "$idx" ]] && continue
      if (( mem >= max_mem || util > max_util )); then
        continue
      fi

      local ok=1
      for _ in $(seq 1 "$stable_needed"); do
        sleep "$sleep_s"
        local line now_mem now_util
        line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v gpu="$idx" '$1 ~ gpu {print $0; exit}')
        now_mem=$(echo "$line" | awk -F, '{gsub(/ /,"",$2); print $2}')
        now_util=$(echo "$line" | awk -F, '{gsub(/ /,"",$3); print $3}')
        if [[ -z "$now_mem" || -z "$now_util" || "$now_mem" -ge "$max_mem" || "$now_util" -gt "$max_util" ]]; then
          ok=0
          break
        fi
      done

      if (( ok == 1 )); then
        LOCK_DIR=$LOCK_ROOT/gpu${idx}.lock
        if mkdir "$LOCK_DIR" 2>/dev/null; then
          echo "$idx"
          return 0
        fi
        log "GPU $idx looked idle but is locked by another local job"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

    log "No stable idle GPU yet; current GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >&2
    sleep "$sleep_s"
  done
}

{
  log "run_id=$RUN_ID"
  log "config=$CONFIG"
  log "waiting for a stable idle GPU"
  GPU=$(select_idle_gpu)
  log "selected physical GPU $GPU"

  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  log "starting LLaMA-Factory training"
  /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train "$CONFIG"
  log "training finished"
} >> "$LOG" 2>&1

echo "$LOG"
