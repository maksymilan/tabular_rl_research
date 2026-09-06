#!/usr/bin/env bash
# Run one interleaved half of the SFT2 full BIRD-dev greedy baseline on GPU6/7.
set -euo pipefail
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
PY=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER=${ADAPTER:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
GPU_ID=${GPU_ID:?set GPU_ID to 6 or 7}
PARITY=${PARITY:?set PARITY to 0 or 1}
[[ "$GPU_ID" == 6 || "$GPU_ID" == 7 ]] || { printf 'GPU6/7 only\n' >&2; exit 2; }
[[ "$PARITY" == 0 || "$PARITY" == 1 ]] || { printf 'PARITY must be 0 or 1\n' >&2; exit 2; }
EXPECTED=767
PORT=$((18100 + GPU_ID))
TAG="sft2_version36_dev1534_greedy_t0_logprobs20_evenodd${PARITY}_20260801"
RESULT_DIR="$RUNTIME/data/results/$TAG"
STATUS="$O/logs/$TAG.status"
RUN_LOG="$O/logs/$TAG.run.log"
VLLM_LOG="$O/logs/$TAG.vllm.log"
EVAL_LOG="$O/logs/$TAG.eval.log"
PID_FILE="$O/logs/$TAG.vllm.pid"
INDICES="$O/logs/$TAG.indices.txt"
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
rows() { test -f "$RESULT_DIR/all.jsonl" && wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]' || printf '0\n'; }
mkdir -p "$O/logs" "$RESULT_DIR"
exec 9>"$O/logs/$TAG.lock"
flock -n 9 || exit 0
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
seq "$PARITY" 2 1533 >"$INDICES"
test "$(wc -l <"$INDICES" | tr -d '[:space:]')" -eq "$EXPECTED"
if [[ "$(rows)" -eq "$EXPECTED" ]]; then
  cd "$RUNTIME"
  "$PY" src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
  set_status complete "rows=$EXPECTED"
  exit 0
fi
free=0
while [[ "$free" -lt 10 ]]; do
  memory=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  compute=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')
  if [[ "$memory" -le 512 && "$compute" -eq 0 ]]; then free=$((free+1)); else free=0; fi
  set_status waiting_gpu "gpu=$GPU_ID free=$free/10 rows=$(rows)/$EXPECTED"
  [[ "$free" -eq 10 ]] || sleep 30
done
for attempt in 1 2 3; do
  set_status running "gpu=$GPU_ID parity=$PARITY attempt=$attempt rows=$(rows)/$EXPECTED concurrency=24"
  set +e
  cd "$RUNTIME"
  PROJECT_DIR="$RUNTIME" VLLM_PYTHON="$PY" PYTHON_BIN="$PY" \
  BASE_MODEL="$MODEL" ADAPTER="$ADAPTER" SERVED_MODEL="sft2-full-dev-greedy-$PARITY" \
  GPU_ID="$GPU_ID" PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON=data/eval_inputs/bird_dev_20240627.jsonl INDICES_FILE="$INDICES" \
  N=1534 N_SAMPLES=1 PASS_K=1 WORKERS=24 SAMPLE_WORKERS=1 MAX_INFLIGHT=24 \
  MAX_NUM_SEQS=24 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION=0.90 MAX_STEPS=30 MAX_TOKENS=1024 TEMPERATURE=0 TOP_P=1 \
  RECORD_LOGPROBS=1 TOP_LOGPROBS=20 HISTORY_TURNS=4 EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID="vllm-version36-sft2-full-dev-greedy-evenodd-c24-logprobs20" \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$PID_FILE" EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh >>"$RUN_LOG" 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 && "$(rows)" -eq "$EXPECTED" ]]; then
    "$PY" src/eval/build_experiment_eval_metrics.py "$RESULT_DIR"
    set_status complete "rows=$EXPECTED result=$RESULT_DIR"
    exit 0
  fi
  set_status retrying "attempt=$attempt exit=$code rows=$(rows)/$EXPECTED"
  sleep 30
done
set_status failed "rows=$(rows)/$EXPECTED"
exit 1
