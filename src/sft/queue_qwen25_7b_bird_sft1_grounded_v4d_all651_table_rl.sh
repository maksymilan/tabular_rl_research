#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
POLL_SECONDS=${POLL_SECONDS:-60}
RUN_ID=${RUN_ID:-bird_sft1_grounded_v4d_all651_6400_$(date +%Y%m%d_%H%M%S)}
QUEUE_LOG="$OUTPUT_ROOT/logs/$RUN_ID.queue.log"
VLLM_PY=${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
VLLM_PORT=${VLLM_PORT:-8015}
SERVED_MODEL=${SERVED_MODEL:-qwen25_bird_sft1_grounded_all651}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen2.5-7B-Instruct}
LORA_DIR=${LORA_DIR:-$OUTPUT_ROOT/checkpoints/qwen2.5-7b-bird-sft1-grounded-v4d-all651-6400-qlora}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$QUEUE_LOG" 2>&1
echo "$(date -Is) waiting for an idle GPU"

gpu_is_idle() {
  local gpu="$1"
  local memory util
  read -r memory util < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  [[ "$memory" -le 512 && "$util" -le 5 ]]
}

while true; do
  for gpu in 0 1; do
    if gpu_is_idle "$gpu"; then
      echo "$(date -Is) GPU $gpu appears idle; confirming for 15 seconds"
      sleep 15
      if gpu_is_idle "$gpu"; then
        echo "$(date -Is) starting training on GPU $gpu"
        set +e
        GPU_ID="$gpu" RUN_ID="$RUN_ID" \
          "$PROJECT_DIR/src/sft/run_qwen25_7b_bird_sft1_grounded_v4d_all651_6400_table_rl.sh"
        status=$?
        set -e
        echo "$(date -Is) training exited with status $status"
        if [[ "$status" -ne 0 ]]; then
          exit "$status"
        fi

        vllm_log="$OUTPUT_ROOT/logs/$RUN_ID.vllm.log"
        vllm_pid_file="$OUTPUT_ROOT/logs/$RUN_ID.vllm.pid"
        echo "$(date -Is) starting evaluation server on GPU $gpu port $VLLM_PORT"
        CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 nohup "$VLLM_PY" \
          -m vllm.entrypoints.openai.api_server \
          --model "$MODEL_DIR" --served-model-name "$SERVED_MODEL" \
          --host 127.0.0.1 --port "$VLLM_PORT" \
          --max-model-len 8192 --gpu-memory-utilization 0.90 \
          --max-num-seqs 8 --max-num-batched-tokens 8192 \
          --enable-lora --max-lora-rank 16 \
          --lora-modules "$SERVED_MODEL=$LORA_DIR" \
          >"$vllm_log" 2>&1 &
        vllm_pid=$!
        echo "$vllm_pid" > "$vllm_pid_file"
        printf '%s\n' \
          "run_id=$RUN_ID" "gpu=$gpu" "port=$VLLM_PORT" \
          "served_model=$SERVED_MODEL" "pid=$vllm_pid" \
          > "$OUTPUT_ROOT/logs/bird_sft1_grounded_v4d_all651_serve.info"
        echo "$(date -Is) evaluation server pid=$vllm_pid"
        exit 0
      fi
    fi
  done
  sleep "$POLL_SECONDS"
done
