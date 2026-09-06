#!/usr/bin/env bash
# Queue the completed table_rl baseline-control60 adapter behind the current
# NewGNN class-conditional60 full-dev evaluation.  The queue is intentionally
# fail-closed on identity/path checks, but does not cancel unrelated jobs.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUN=${TRAIN_RUN:-$OUTPUT_ROOT/qwen3_v26_reward_ab_gate12_table_rl_20260905/baseline_control60_r3/train}
export TRAIN_RUN
CURRENT_EVAL_STATUS=${CURRENT_EVAL_STATUS:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26/qwen3_v26_rl_class_conditional60_r1_dev1534_greedy_20260905e/status.json}
EVAL_ROOT=${EVAL_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_v26_reward_ab_gate12_table_rl_20260905}
EVAL_RUN=${EVAL_RUN:-$EVAL_ROOT/baseline_control60_r3_after_class_conditional60}
SCREEN_ROOT=${SCREEN_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_v26_new_screen600_table_rl_20260905}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER=${ADAPTER_PATH:-$TRAIN_RUN/checkpoint-4}
DEV_TASKS=${DEV_TASKS:-/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
GPU_ID=${GPU_ID:-1}
PORT=${PORT:-8101}
NEWGNN_HOST=${NEWGNN_HOST:-NewGNN}
POLL_SECONDS=${POLL_SECONDS:-60}

LOG_DIR=${LOG_DIR:-$(dirname "$EVAL_RUN")/queue_logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/baseline_control60_queue.log}
mkdir -p "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"; }
die() { log "blocked: $*"; exit 3; }

[[ -x "$PYTHON" ]] || die "missing training/evaluation Python: $PYTHON"
[[ -x "$VLLM_PYTHON" ]] || die "missing vLLM Python: $VLLM_PYTHON"
[[ -d "$RUNTIME" && -f "$RUNTIME/src/eval/rollout_passk.py" ]] || die "missing frozen runtime: $RUNTIME"
[[ -f "$MODEL/config.json" ]] || die "missing base model: $MODEL"
[[ -f "$DEV_TASKS" ]] || die "missing dev tasks: $DEV_TASKS"

wait_for_checkpoint() {
  log "waiting for training checkpoint-4: $ADAPTER"
  while true; do
    if [[ -f "$ADAPTER/trainer_state.json" && -f "$ADAPTER/adapter_model.safetensors" ]]; then
      if "$PYTHON" - "$ADAPTER/trainer_state.json" <<'PY' >/dev/null 2>&1
import json, sys
state = json.load(open(sys.argv[1]))
raise SystemExit(0 if int(state.get("global_step", -1)) >= 4 else 1)
PY
      then
        log "checkpoint-4 is complete"
        return
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_train_exit() {
  log "waiting for table_rl launcher and trainer cleanup"
  # Exclude this watcher itself: its command line contains TRAIN_RUN because
  # the script exports the default path above.  Without this guard the queue
  # waits forever after the trainer has exited.
  while ps -eo pid=,args= | awk -v self="$$" '$1 != self && index($0, ENVIRON["TRAIN_RUN"]) {found=1} END {exit found ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
  log "table_rl training processes are gone"
}

wait_for_current_eval() {
  log "waiting for current NewGNN 60-arm evaluation to finish"
  while true; do
    payload=$(ssh "$NEWGNN_HOST" "python3 -c 'import json; d=json.load(open(\"$CURRENT_EVAL_STATUS\")); print(str(d.get(\"state\",\"missing\"))+\"|\"+str(d.get(\"finished_at_utc\") or \"\"))'" 2>/dev/null) || {
      log "current evaluation status is temporarily unreadable; waiting"
      sleep "$POLL_SECONDS"
      continue
    }
    state=${payload%%|*}
    finished=${payload#*|}
    case "$state:$finished" in
      completed:*|failed:?*) log "current evaluation terminal state=$state"; return ;;
      *) sleep "$POLL_SECONDS" ;;
    esac
  done
}

wait_for_eval_gpu() {
  log "waiting for table_rl GPU$GPU_ID and port $PORT to be free"
  while ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v target="$GPU_ID" 'BEGIN {ok=0} {if (($1+0)==(target+0)) {ok=1; if (($2+0)>1024) exit 1}} END {exit ok ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
  while ss -ltnH 2>/dev/null | awk -v port=":$PORT" '$4 ~ port"$" {found=1} END {exit found ? 0 : 1}'; do
    sleep "$POLL_SECONDS"
  done
}

wait_for_screen() {
  log "waiting for the post-training table_rl screening batch"
  while [[ ! -f "$SCREEN_ROOT/selection/boundary_cohort/selection_manifest.json" \
    || ! -f "$SCREEN_ROOT/selection/boundary_cohort/selected_tasks.jsonl" ]]; do
    sleep "$POLL_SECONDS"
  done
  log "post-training table_rl screening completed"
}

cleanup_vllm() {
  status=$?
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  log "evaluation cleanup complete status=$status"
  exit "$status"
}

run_eval() {
  [[ ! -e "$EVAL_RUN" ]] || die "evaluation output already exists: $EVAL_RUN"
  mkdir -p "$EVAL_RUN"
  trap cleanup_vllm EXIT INT TERM

  log "starting matched greedy evaluation on table_rl GPU$GPU_ID"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --tokenizer "$MODEL" \
      --served-model-name qwen3-8b-atomic-v26-backbone \
      --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
      --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 4 \
      --gpu-memory-utilization 0.90 --generation-config vllm --enable-lora \
      --lora-modules qwen3-8b-atomic-v26-sft1-qlora="$ADAPTER" --max-lora-rank 64
  ) >"$EVAL_RUN/vllm.log" 2>&1 &
  VLLM_PID=$!
  echo "$VLLM_PID" > "$EVAL_RUN/vllm.pid"

  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "vLLM did not become healthy"

  log "vLLM healthy; evaluating 1534 dev tasks"
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONPATH="$RUNTIME/src/eval:$RUNTIME/src"
  export EVAL_ENABLE_THINKING=1
  "$PYTHON" -u "$RUNTIME/src/eval/rollout_passk.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model qwen3-8b-atomic-v26-sft1-qlora \
    --examples-json "$DEV_TASKS" --allow-eval-tasks --n 1534 \
    --n-samples 1 --pass-k 1 --workers 4 --sample-workers 1 \
    --max-inflight-requests 4 --max-steps 30 --max-tokens 2048 \
    --temperature 0 --top-p 1 --sample-detail full --summary-every 10 \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --denotation-comparison bird-set --result-dir "$EVAL_RUN/result"
  log "matched greedy evaluation completed"
}

wait_for_checkpoint
wait_for_train_exit
wait_for_screen
wait_for_current_eval
wait_for_eval_gpu
run_eval
