#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT="${REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/text2sql_reproduction}"
SQLR1_CODE="${SQLR1_CODE:-$REPRO_ROOT/third_party/SQL-R1}"
MODEL_PATH="${MODEL_PATH:-/home/dengyan/models/text2sql_reproduction/SQL-R1-7B}"
BIRD_ROOT="${BIRD_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
EVAL_SCOPE="${EVAL_SCOPE:-smoke8}"
N="${N:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ "$EVAL_SCOPE" == "full" ]]; then
  INPUT_FILE="${INPUT_FILE:-$BIRD_ROOT/dev.json}"
else
  INPUT_FILE="${INPUT_FILE:-$REPRO_ROOT/data/bird_dev_smoke8.json}"
fi

RUN_NAME="bird_dev_${EVAL_SCOPE}_n${N}_t${TEMPERATURE}"
RESULT_DIR="$REPRO_ROOT/results/sqlr1/$RUN_NAME"
mkdir -p "$RESULT_DIR"

(
  cd "$SQLR1_CODE"
  CUDA_VISIBLE_DEVICES=0,1 "$EVAL_PYTHON" src/inference.py \
    --nl2sql_ckpt_path "$MODEL_PATH" \
    --dataset_name bird \
    --input_file "$INPUT_FILE" \
    --output_file "$RESULT_DIR/predictions.json" \
    --database_path "$BIRD_ROOT/dev_databases" \
    --tensor_parallel_size 2 \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --output_format json \
    --table_value_cache_path "$SQLR1_CODE/db_info/bird_db_id2sampled_db_values.json" \
    --table_info_cache_path "$SQLR1_CODE/db_info/bird_db_id2db_info.json"

  mode=greedy_search
  if [[ "$N" -gt 1 ]]; then
    mode=major_voting
  fi
  "$EVAL_PYTHON" src/evaluation_bird_post.py \
    --pred "$RESULT_DIR/predictions.json" \
    --gold "$INPUT_FILE" \
    --db_path "$BIRD_ROOT/dev_databases" \
    --mode "$mode" | tee "$RESULT_DIR/evaluation.log"
)
