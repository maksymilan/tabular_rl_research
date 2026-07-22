#!/usr/bin/env bash
# Run Qwen3.5-9B v9 SFT pass@2 on the v10 TRAIN subset, then select recovery candidates.
#
# Intended host: table_rl
# Run from: /home/dengyan/tabular_rl_project
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3.5-9B}
LORA_DIR=${LORA_DIR:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-spider-v9-ready-4k-epoch4-qlora}
VLLM_PY=${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
SFT_PY=${SFT_PY:-/home/dengyan/miniconda3/envs/sft/bin/python}
SERVED_MODEL=${SERVED_MODEL:-qwen35_v9_sft}
EXAMPLES_JSON=${EXAMPLES_JSON:-data/eval_inputs/subset_v10_clean_1600_train_examples.json}
RESULT_DIR=${RESULT_DIR:-data/results/qwen3.5_9b_sft_v9_ready_4k_epoch4/tool_pass2_train_v10_1600}
RECOVERY_OUT=${RECOVERY_OUT:-data/trajectories/recovery_candidates_v10.jsonl}
RUN_ID=${RUN_ID:-qwen35_v9_train_v10_pass2_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

LOG="$LOG_DIR/${RUN_ID}.log"
VLLM_LOG="$LOG_DIR/${RUN_ID}.vllm.log"
VLLM_PID=""

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && ps -p "$VLLM_PID" >/dev/null 2>&1; then
    echo "[$(date)] stopping vLLM pid=$VLLM_PID"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}
trap cleanup EXIT INT TERM

pick_idle_gpu() {
  while true; do
    while IFS=, read -r idx mem util; do
      idx=$(echo "$idx" | xargs)
      mem=$(echo "$mem" | xargs)
      util=$(echo "$util" | xargs)
      if [ "$mem" -le "${MAX_IDLE_MEM_MIB:-512}" ] && [ "$util" -le "${MAX_IDLE_UTIL:-5}" ]; then
        echo "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    echo "[$(date)] no idle GPU yet; current status:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    sleep 60
  done
}

find_free_port() {
  "$SFT_PY" - <<'PY'
import socket
for port in range(8000, 8020):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free local port in 8000-8019")
PY
}

{
  echo "[$(date)] run_id=$RUN_ID"
  echo "[$(date)] examples=$EXAMPLES_JSON"
  echo "[$(date)] result_dir=$RESULT_DIR"
  GPU=${GPU_ID:-}
  if [ -z "$GPU" ]; then
    GPU=$(pick_idle_gpu)
  else
    echo "[$(date)] using requested GPU=$GPU"
  fi
  PORT=${VLLM_PORT:-}
  if [ -z "$PORT" ]; then
    PORT=$(find_free_port)
  else
    echo "[$(date)] using requested port=$PORT"
  fi
  echo "[$(date)] selected GPU=$GPU port=$PORT"

  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_HUB_OFFLINE=1
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.86 \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "$SERVED_MODEL=$LORA_DIR" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "[$(date)] started vLLM pid=$VLLM_PID log=$VLLM_LOG"

  for i in $(seq 1 180); do
    if curl -sS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[$(date)] vLLM ready"
      break
    fi
    if ! ps -p "$VLLM_PID" >/dev/null 2>&1; then
      echo "[$(date)] vLLM exited early"
      tail -80 "$VLLM_LOG" || true
      exit 1
    fi
    if [ "$i" -eq 180 ]; then
      echo "[$(date)] vLLM readiness timeout"
      tail -80 "$VLLM_LOG" || true
      exit 1
    fi
    sleep 5
  done

  export EVAL_ENABLE_THINKING=0
  "$SFT_PY" src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_MODEL" \
    --examples-json "$EXAMPLES_JSON" \
    --n-samples 2 --pass-k 2 \
    --workers "${ROLLOUT_WORKERS:-4}" \
    --sample-workers "${SAMPLE_WORKERS:-2}" \
    --max-steps 20 --max-tokens 768 --api-retries 3 \
    --result-dir "$RESULT_DIR" --resume

  echo "[$(date)] rollout finished; selecting recovery candidates"
  "$SFT_PY" src/sft/select_recovery_candidates.py \
    --input "$RESULT_DIR/all.jsonl" \
    --output "$RECOVERY_OUT" \
    --limit 400 --per-db-cap 12

  echo "[$(date)] done"
  test -f "$RESULT_DIR/summary.json" && cat "$RESULT_DIR/summary.json"
} >> "$LOG" 2>&1 &

echo "started ${RUN_ID}; log=$LOG"
