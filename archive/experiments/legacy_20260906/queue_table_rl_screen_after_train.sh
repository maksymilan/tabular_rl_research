#!/usr/bin/env bash
# Start a second, disjoint fresh K=8 screening batch as soon as the
# table_rl baseline-control60 training has released both GPUs.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUN=${TRAIN_RUN:-$OUTPUT_ROOT/qwen3_v26_reward_ab_gate12_table_rl_20260905/baseline_control60_r3/train}
export TRAIN_RUN
SCREEN_ROOT=${SCREEN_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_v26_new_screen600_table_rl_20260905}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/checkpoint-6380}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
POLL_SECONDS=${POLL_SECONDS:-60}

LOG_DIR=${LOG_DIR:-$SCREEN_ROOT/logs}
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/screen_queue.log
SELECTOR=${SELECTOR:-$SCREEN_ROOT/input/select_epoch4_pass8_boundary_cohort.py}
TASKS=${TASKS:-$SCREEN_ROOT/input/tasks600.jsonl}
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"; }
die() { log "blocked: $*"; exit 3; }

wait_for_train_exit() {
  log "waiting for table_rl training to release both GPUs"
  # Exclude this watcher itself: its command line contains TRAIN_RUN because
  # the script exports the default path above.  Without this guard the queue
  # waits forever after the trainer has exited.
  while ps -eo pid=,args= | awk -v self="$$" '$1 != self && index($0, ENVIRON["TRAIN_RUN"]) {found=1} END {exit found ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
  [[ -f "$ADAPTER/adapter_model.safetensors" ]] || die "checkpoint-6380 is missing: $ADAPTER"
  log "table_rl training released; starting second screen"
}

wait_for_free_gpu() {
  log "waiting for table_rl GPU0/1 to be free"
  while ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, 'BEGIN {ok=1; saw=0} {saw=1; g=$1+0; m=$2+0; if ((g==0 || g==1) && m>1024) ok=0} END {exit (saw && ok) ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
  sleep 15
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, 'BEGIN {ok=1; saw=0} {saw=1; g=$1+0; m=$2+0; if ((g==0 || g==1) && m>1024) ok=0} END {exit (saw && ok) ? 0 : 1}' \
    || die "table_rl GPU0/1 became occupied during preflight"
}

wait_for_free_ports() {
  log "waiting for screening ports 8110/8111 to be free"
  while ss -ltnH 2>/dev/null | awk '$4 ~ /:8110$|:8111$/ {found=1} END {exit found ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
}

cleanup() {
  status=$?
  for pid_file in "$SCREEN_ROOT"/shard-*/vllm.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(cat "$pid_file")
    kill -TERM "$pid" 2>/dev/null || true
  done
  log "screen cleanup complete status=$status"
  exit "$status"
}

start_shard() {
  local gpu=$1 port=$2 shard=$3
  local dir="$SCREEN_ROOT/$shard"
  mkdir -p "$dir"
  [[ ! -e "$dir/result" ]] || die "screen output exists: $dir/result"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --tokenizer "$MODEL" \
      --served-model-name qwen3-8b-atomic-v26-sft1-qlora \
      --host 127.0.0.1 --port "$port" --dtype bfloat16 \
      --max-model-len 16384 --max-num-batched-tokens 8192 --max-num-seqs 4 \
      --gpu-memory-utilization 0.90 --generation-config vllm --enable-lora \
      --lora-modules qwen3-8b-atomic-v26-sft1-qlora="$ADAPTER" --max-lora-rank 64
  ) >"$dir/vllm.log" 2>&1 &
  echo $! >"$dir/vllm.pid"
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 || die "vLLM not healthy for $shard"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" EVAL_ENABLE_THINKING=1
    export PYTHONPATH="$RUNTIME/src/eval:$RUNTIME/src"
    exec "$PYTHON" -u "$RUNTIME/src/eval/rollout_passk.py" \
      --base-url "http://127.0.0.1:$port/v1" \
      --model qwen3-8b-atomic-v26-sft1-qlora \
      --examples-json "$SCREEN_ROOT/input/$shard.jsonl" --n 300 \
      --workers 4 --sample-workers 1 --max-inflight-requests 4 \
      --n-samples 8 --pass-k 1,2,4,8 --max-steps 30 --max-tokens 2048 \
      --temperature 0.8 --top-p 1 --sample-detail compact --summary-every 10 \
      --context-mode rolling-legal-history --history-turns 4 \
      --rolling-prompt-variant full --rolling-observation-style resident \
      --denotation-comparison bird-set --result-dir "$dir/result"
  ) >"$dir/rollout.log" 2>&1 &
  echo $! >"$dir/rollout.pid"
  log "started $shard gpu=$gpu port=$port vllm=$(cat "$dir/vllm.pid") rollout=$(cat "$dir/rollout.pid")"
}

wait_for_train_exit
wait_for_free_gpu
wait_for_free_ports
trap cleanup EXIT INT TERM
start_shard 0 8110 shard-0
start_shard 1 8111 shard-1
wait "$(cat "$SCREEN_ROOT/shard-0/rollout.pid")"
wait "$(cat "$SCREEN_ROOT/shard-1/rollout.pid")"
[[ -f "$SELECTOR" && -f "$TASKS" ]] || die "screen selector inputs are missing"
mkdir -p "$SCREEN_ROOT/selection"
export PYTHONPATH="$RUNTIME/src:$RUNTIME/src/rl:$RUNTIME"
"$PYTHON" "$SELECTOR" \
  --tasks "$TASKS" \
  --screen "$SCREEN_ROOT/shard-0/result/all.jsonl" \
  --screen "$SCREEN_ROOT/shard-1/result/all.jsonl" \
  --output-dir "$SCREEN_ROOT/selection/boundary_cohort" \
  --count 300 --seed "qwen3-v26-screen600-table-rl-20260905"
log "second 600-task screening and 300-task boundary selection completed"
