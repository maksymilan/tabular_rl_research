#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
REMOTE=${REMOTE:-table_rl}
GPU_ID=${GPU_ID:-0}
REMOTE_PORT=${REMOTE_PORT:-8018}
LOCAL_PORT=${LOCAL_PORT:-18021}
SERVED_MODEL=${SERVED_MODEL:-qwen25_7b_bird_direct_sql_base_passk4}
REMOTE_OUTPUT=/home/dengyan/tabular_rl_outputs
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-7B-Instruct}
VLLM_PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
REMOTE_PID_FILE=${REMOTE_PID_FILE:-$REMOTE_OUTPUT/logs/bird_direct_sql_passk4.vllm.pid}
LOG=${LOG:-/tmp/bird_direct_sql_passk4.log}
TASKS=data/eval_inputs/bird_dev_20240627.jsonl
N_SAMPLES=${N_SAMPLES:-4}
PASS_K=${PASS_K:-1,2,4}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
REPETITION_PENALTY=${REPETITION_PENALTY:-}
RESULT_VARIANT=${RESULT_VARIANT:-passk4}
RESULT_STEM=${RESULT_STEM:-qwen2.5_7b_bird_direct_sql_base}
SMOKE_RESULT_DIR=${SMOKE_RESULT_DIR:-data/results/${RESULT_STEM}_${RESULT_VARIANT}_smoke10_bird_ex}
FULL_RESULT_DIR=${FULL_RESULT_DIR:-data/results/${RESULT_STEM}_${RESULT_VARIANT}_dev1534_bird_ex}

exec >>"$LOG" 2>&1
cd "$PROJECT_DIR"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
tunnel_pid=""

stop_remote_model() {
  ssh "$REMOTE" "if test -f '$REMOTE_PID_FILE'; then pid=\$(cat '$REMOTE_PID_FILE'); test -n \"\$pid\" && kill \"\$pid\" 2>/dev/null || true; fi"
}
cleanup() {
  stop_remote_model >/dev/null 2>&1 || true
  test -n "$tunnel_pid" && kill "$tunnel_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "$(timestamp) starting BIRD-dev direct-SQL pass@1/2/4 baseline on GPU $GPU_ID"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE" &
tunnel_pid=$!

ssh "$REMOTE" "CUDA_VISIBLE_DEVICES='$GPU_ID' HF_HUB_OFFLINE=1 nohup '$VLLM_PY' \
  -m vllm.entrypoints.openai.api_server \
  --model '$BASE_MODEL' --served-model-name '$SERVED_MODEL' \
  --host 127.0.0.1 --port '$REMOTE_PORT' \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-num-batched-tokens 16384 \
  >'$REMOTE_OUTPUT/logs/$SERVED_MODEL.vllm.log' 2>&1 </dev/null & echo \$! >'$REMOTE_PID_FILE'"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
    echo "$(timestamp) model ready"
    break
  fi
  sleep 3
done
curl --noproxy '*' -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"

run_guarded() {
  local failures=0
  EVAL_ENABLE_THINKING=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost "$@" &
  local eval_pid=$!
  while kill -0 "$eval_pid" 2>/dev/null; do
    sleep 15
    if ! kill -0 "$tunnel_pid" 2>/dev/null; then
      echo "$(timestamp) tunnel exited; terminating evaluation"
      kill "$eval_pid" 2>/dev/null || true
      wait "$eval_pid" 2>/dev/null || true
      return 75
    fi
    if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/v1/models" | grep -q "$SERVED_MODEL"; then
      failures=0
    else
      failures=$((failures + 1))
      if (( failures >= 3 )); then
        echo "$(timestamp) endpoint failed three health checks; terminating evaluation"
        kill "$eval_pid" 2>/dev/null || true
        wait "$eval_pid" 2>/dev/null || true
        return 75
      fi
    fi
  done
  wait "$eval_pid"
}

COMMON=(.venv/bin/python -u src/eval/text2sql_passk.py
  --base-url "http://127.0.0.1:${LOCAL_PORT}/v1" --model "$SERVED_MODEL"
  --tasks-json "$TASKS" --n-samples "$N_SAMPLES" --pass-k "$PASS_K"
  --workers 4 --max-tokens 1024 --temperature "$TEMPERATURE" --top-p "$TOP_P"
  --api-retries 3 --execution-timeout-seconds 20 \
  --denotation-comparison bird-set)
if [[ -n "$REPETITION_PENALTY" ]]; then
  COMMON+=(--repetition-penalty "$REPETITION_PENALTY")
fi

echo "$(timestamp) starting smoke-10"
run_guarded "${COMMON[@]}" --n 10 \
  --resume \
  --result-dir "$SMOKE_RESULT_DIR"
.venv/bin/python -c 'import json,sys; from pathlib import Path; rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]; expected=int(sys.argv[2]); assert len(rows)==10 and all(len(r.get("samples",[]))==expected for r in rows)' "$SMOKE_RESULT_DIR/all.jsonl" "$N_SAMPLES"

echo "$(timestamp) smoke passed; starting full BIRD dev-1534"
run_guarded "${COMMON[@]}" --n 1534 \
  --resume \
  --result-dir "$FULL_RESULT_DIR"
echo "$(timestamp) BIRD-dev direct-SQL pass@1/2/4 baseline complete"
