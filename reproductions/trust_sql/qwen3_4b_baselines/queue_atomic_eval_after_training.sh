#!/usr/bin/env bash
set -euo pipefail

# Detached NewGNN queue: wait for the fixed Qwen3-4B training run, freeze
# checkpoint-560, then launch base/adapter atomic-v26 full evaluations on
# separate GPUs. Any failed prerequisite stops the queue without an eval run.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TRAIN_STATUS="/home/dengyan/tabular_rl_outputs/logs/qwen3_4b_atomic_v26_sft1_full_20260808.status.json"
TRAIN_PID_FILE="/home/dengyan/tabular_rl_outputs/runtime/qwen3_4b_atomic_sft1_20260808/train_full_launcher.log.pid"
RUN_ROOT="/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_atomic_v26"
SOURCE="/home/dengyan/tabular_rl_outputs/eval_inputs/qwen3_atomic_v26/bird_dev_20240627.jsonl"
RUNTIME_ROOT="/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
MODEL_ROOT="/home/dengyan/models/Qwen3-4B-TrustSQL-baseline"
ADAPTER="/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
PYTHON="/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python"
QUEUE_STATUS="$HERE/queue.status"
BASE_RUN="$RUN_ROOT/qwen3_4b_atomic_v26_base_bird_dev1534_20260808"
ADAPTER_RUN="$RUN_ROOT/qwen3_4b_atomic_v26_sft1_bird_dev1534_20260808"

printf 'waiting_for_training %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
while [[ ! -f "$TRAIN_STATUS" ]]; do
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    train_pid="$(tr -d '[:space:]' <"$TRAIN_PID_FILE")"
    if [[ "$train_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$train_pid" 2>/dev/null; then
      printf 'failed training_process_exited_without_status %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
      exit 10
    fi
  fi
  sleep 60
done

"$PYTHON" - "$TRAIN_STATUS" <<'PY'
import json, sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
if status.get("success") is not True or int(status.get("expected_global_step", -1)) != 560:
    raise SystemExit(f"training status did not pass: {status}")
PY

sha256sum -c "$HERE/STATIC_SHA256SUMS" >/dev/null
"$PYTHON" "$HERE/freeze_qwen3_4b_adapter_lock.py" \
  --adapter "$ADAPTER" --output "$HERE/qwen3_4b_adapter_lock.json" \
  >"$HERE/freeze_adapter_lock.log"

for gpu in 5 6; do
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 512 )) || {
    printf 'failed gpu_%s_not_idle_%sMiB %s\n' "$gpu" "$used" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
    exit 11
  }
done

mkdir -p "$RUN_ROOT"
for run in "$BASE_RUN" "$ADAPTER_RUN"; do
  [[ ! -e "$run" ]] || {
    printf 'failed run_directory_exists_%s %s\n' "$run" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
    exit 12
  }
  mkdir "$run"
done

nohup setsid "$PYTHON" -u "$HERE/remote_eval_supervisor.py" \
  --mode base --model-size 4b --run-dir "$BASE_RUN" --gpu-ids 5 --port 8020 \
  --source "$SOURCE" --runtime-root "$RUNTIME_ROOT" --model-root "$MODEL_ROOT" \
  >"$BASE_RUN/supervisor.log" 2>&1 </dev/null &
base_pid=$!
printf '%s\n' "$base_pid" >"$BASE_RUN/submitted.pid"

nohup setsid "$PYTHON" -u "$HERE/remote_eval_supervisor.py" \
  --mode adapter --model-size 4b --run-dir "$ADAPTER_RUN" --gpu-ids 6 --port 8021 \
  --source "$SOURCE" --runtime-root "$RUNTIME_ROOT" --model-root "$MODEL_ROOT" \
  --adapter "$ADAPTER" \
  >"$ADAPTER_RUN/supervisor.log" 2>&1 </dev/null &
adapter_pid=$!
printf '%s\n' "$adapter_pid" >"$ADAPTER_RUN/submitted.pid"

printf 'submitted base_pid=%s adapter_pid=%s %s\n' "$base_pid" "$adapter_pid" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
