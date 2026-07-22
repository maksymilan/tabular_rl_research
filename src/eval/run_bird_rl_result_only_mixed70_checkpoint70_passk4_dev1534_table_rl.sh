#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-8021}
SERVED_MODEL=qwen25_bird_sft2_result_only_mixed70_retry1_checkpoint70
BASE_MODEL=/home/dengyan/models/Qwen2.5-7B-Instruct
ADAPTER=$OUTPUT_ROOT/checkpoints/qwen25_bird_sft2_result_only_mixed70_baseline_retry1_20260721/checkpoint-70
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
EVAL_PY=/home/dengyan/miniconda3/envs/verl-table/bin/python
TASKS=data/eval_inputs/bird_dev_20240627.remote.jsonl
RUN_ID=${RUN_ID:-bird_rl_result_only_mixed70_checkpoint70_passk4_dev1534_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
LOG=$LOG_DIR/$RUN_ID.log
VLLM_LOG=$LOG_DIR/$SERVED_MODEL.vllm.log
VLLM_PID_FILE=$LOG_DIR/$SERVED_MODEL.vllm.pid
SMOKE_DIR=$OUTPUT_ROOT/results/qwen2.5_7b_bird_sft2_result_only_mixed70_retry1_checkpoint70_passk4_dev_smoke2_bird_ex
RESULT_DIR=$OUTPUT_ROOT/results/qwen2.5_7b_bird_sft2_result_only_mixed70_retry1_checkpoint70_passk4_dev1534_bird_ex

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/results"
cd "$PROJECT_DIR"
exec >>"$LOG" 2>&1

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

cleanup() {
  if test -f "$VLLM_PID_FILE"; then
    pid=$(cat "$VLLM_PID_FILE")
    test -n "$pid" && kill "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "$(timestamp) run_id=$RUN_ID"
echo "gpu=$GPU_ID port=$PORT model=$BASE_MODEL adapter=$ADAPTER"
echo "protocol=K4,temp0.7,top_p0.95,rolling4/full/resident,max_steps30,max_tokens1024"

CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 nohup "$VLLM_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "$SERVED_MODEL=$ADAPTER" \
  >"$VLLM_LOG" 2>&1 </dev/null &
echo $! >"$VLLM_PID_FILE"

for _ in $(seq 1 160); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED_MODEL"; then
    echo "$(timestamp) model ready"
    break
  fi
  sleep 3
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED_MODEL"

run_eval() {
  local n=$1
  local result_dir=$2
  EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    "$EVAL_PY" -u src/eval/rollout_passk.py \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_MODEL" \
    --examples-json "$TASKS" --allow-eval-tasks \
    --n "$n" --n-samples 4 --pass-k 1,2,4 \
    --workers 2 --sample-workers 4 --max-inflight-requests 8 \
    --max-steps 30 --max-tokens 1024 --temperature 0.7 --top-p 0.95 \
    --sample-detail full --summary-every 10 \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --denotation-comparison bird-set \
    --result-dir "$result_dir" --resume &
  local eval_pid=$!
  local failures=0
  while kill -0 "$eval_pid" 2>/dev/null; do
    sleep 15
    if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED_MODEL"; then
      failures=0
    else
      failures=$((failures + 1))
      if (( failures >= 3 )); then
        echo "$(timestamp) endpoint failed three health checks"
        kill "$eval_pid" 2>/dev/null || true
        wait "$eval_pid" 2>/dev/null || true
        return 75
      fi
    fi
  done
  wait "$eval_pid"
}

echo "$(timestamp) starting smoke2"
run_eval 2 "$SMOKE_DIR"
"$EVAL_PY" - "$SMOKE_DIR/all.jsonl" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 2, f"smoke expected 2 tasks, got {len(rows)}"
assert all(row.get("attempted_samples") == 4 for row in rows), "smoke has incomplete K=4 samples"
assert not any(
    sample.get("failure_type") == "api_error"
    for row in rows
    for sample in row.get("samples", [])
), "smoke contains API errors"
print("smoke gate passed: 2 tasks, 8 complete samples, no API errors")
PY

echo "$(timestamp) starting full BIRD-dev"
run_eval 1534 "$RESULT_DIR"
echo "$(timestamp) RL result-only BIRD-dev pass@1/2/4 complete: $RESULT_DIR"
