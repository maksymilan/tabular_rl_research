#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT="${REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/text2sql_reproduction}"
ARCTIC_CODE="${ARCTIC_CODE:-$REPRO_ROOT/third_party/ArcticTraining/projects/arctic_text2sql_r1}"
MODEL_PATH="${MODEL_PATH:-/home/dengyan/models/text2sql_reproduction/Arctic-Text2SQL-R1-7B}"
BIRD_ROOT="${BIRD_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
EVAL_SCOPE="${EVAL_SCOPE:-smoke8}"
N="${N:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ "$EVAL_SCOPE" == "full" ]]; then
  INPUT_FILE="${INPUT_FILE:-$REPRO_ROOT/data/dev_bird_arctic.json}"
  GOLD_FILE="${GOLD_FILE:-$BIRD_ROOT/dev.json}"
else
  INPUT_FILE="${INPUT_FILE:-$REPRO_ROOT/data/dev_bird_arctic_smoke8.json}"
  GOLD_FILE="${GOLD_FILE:-$REPRO_ROOT/data/bird_dev_smoke8.json}"
fi

RUN_NAME="bird_dev_${EVAL_SCOPE}_n${N}_t${TEMPERATURE}"
RESULT_DIR="$REPRO_ROOT/results/arctic/$RUN_NAME"
mkdir -p "$RESULT_DIR"

(
  cd "$ARCTIC_CODE"
  CUDA_VISIBLE_DEVICES=0,1 "$EVAL_PYTHON" bird_eval/infer.py \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$RESULT_DIR/predictions.json" \
    --tensor_parallel_size 2 \
    --n "$N" \
    --temperature "$TEMPERATURE"

  mode=greedy_search
  if [[ "$N" -gt 1 ]]; then
    mode=major_voting
  fi
  "$EVAL_PYTHON" bird_eval/evaluate_bird.py \
    --pred "$RESULT_DIR/predictions.json" \
    --gold "$GOLD_FILE" \
    --db_path "$BIRD_ROOT/dev_databases" \
    --mode "$mode" | tee "$RESULT_DIR/evaluation.log"
)
