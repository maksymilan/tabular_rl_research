#!/usr/bin/env bash
# Launch one full pure-result-reward RL run, then evaluate its final adapter on Spider dev.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
TRAIN_GPU=${TRAIN_GPU:-0}
EVAL_GPU=${EVAL_GPU:-1}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-7B-Instruct}
INITIAL_ADAPTER=${INITIAL_ADAPTER:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-external-rollout-v3-success880-epoch4-qlora}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/sft/bin/python}
TRAIN_STEPS=${TRAIN_STEPS:-$($PYTHON -c "import json; print(len(json.load(open('$PROJECT_DIR/data/spider_data/train_spider.json'))))")}
RUN_ID=${RUN_ID:-qwen25_7b_result_only_group_reinforce_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/home/dengyan/tabular_rl_outputs/checkpoints/$RUN_ID}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/data/results/$RUN_ID/tool_zero_shot_dev1034}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TRAIN_LOG="$LOG_DIR/$RUN_ID.train.log"
EVAL_LOG="$LOG_DIR/$RUN_ID.eval.log"
TRAIN_STATUS="$OUTPUT_DIR/train.exit"
TRAIN_RUNNER="$PROJECT_DIR/src/rl/frameworks/accelerate/run_result_only_group_reinforce.sh"
EVAL_RUNNER="$PROJECT_DIR/src/eval/run_qwen35_plan_eval_table_rl.sh"

# One pass over the complete Spider train split. Each group contains four independently sampled
# episodes. The SFT adapter is initialization only; no SFT-filtered task selection is supplied.
# The explicit exit-status file lets the independent evaluation watcher distinguish completion from failure.
nohup env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$INITIAL_ADAPTER" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash -c 'bash "$1" --steps "$2" --group-size 4 --rollout-batch-size 4 --logprob-micro-batch-size 4 --train-turns all --max-steps 12 --max-new-tokens 128 --max-context-tokens 6144 --learning-rate 5e-6 --save-every 100; status=$?; printf "%s\\n" "$status" > "$3"; exit "$status"' \
  _ "$TRAIN_RUNNER" "$TRAIN_STEPS" "$TRAIN_STATUS" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

(
  while [ ! -f "$TRAIN_STATUS" ]; do sleep 60; done
  status=$(cat "$TRAIN_STATUS")
  echo "[$(date)] training exit=$status" >>"$EVAL_LOG"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
  FINAL_ADAPTER="$OUTPUT_DIR/checkpoint-$TRAIN_STEPS"
  if [ ! -d "$FINAL_ADAPTER" ]; then
    echo "missing final adapter: $FINAL_ADAPTER" >>"$EVAL_LOG"
    exit 1
  fi
  MODEL_DIR="$MODEL_PATH" GPU_ID="$EVAL_GPU" VLLM_PORT=8021 \
  LORA_DIR="$FINAL_ADAPTER" SERVED_MODEL="${RUN_ID}_final" RESULT_DIR="$RESULT_DIR" \
  RUN_ID="${RUN_ID}_eval" MAX_MODEL_LEN=8192 MAX_STEPS=30 MAX_CONSECUTIVE_ERRORS=3 \
  MAX_TOKENS=768 ROLLOUT_WORKERS=3 N_DEV=1034 TABLE_OUTPUT_ROWS=10 \
  bash "$EVAL_RUNNER" >>"$EVAL_LOG" 2>&1
) &
EVAL_PID=$!

printf 'run_id=%s\ntrain_steps=%s\ntrain_pid=%s\neval_pid=%s\ntrain_log=%s\neval_log=%s\noutput_dir=%s\nresult_dir=%s\n' \
  "$RUN_ID" "$TRAIN_STEPS" "$TRAIN_PID" "$EVAL_PID" "$TRAIN_LOG" "$EVAL_LOG" "$OUTPUT_DIR" "$RESULT_DIR"
