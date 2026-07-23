#!/usr/bin/env bash
set -euo pipefail

# Faithful OmniSQL BIRD-dev inference wrapper. It intentionally calls the
# pinned upstream infer.py and evaluate_bird.py instead of duplicating them.

MODE="${1:-smoke}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql/source/OmniSQL}"
MODEL_DIR="${MODEL_DIR:-/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67}"
DATA_DIR="${DATA_DIR:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql/data}"
BIRD_DIR="${BIRD_DIR:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql/results}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

case "$MODE" in
  smoke)
    INPUT_FILE="$DATA_DIR/dev_bird_smoke8.json"
    # The upstream evaluator selects BIRD's `SQL` key by checking for the
    # literal substring "bird" in this path.
    GOLD_FILE="$DATA_DIR/bird_dev_smoke8.json"
    RUN_DIR="$OUTPUT_ROOT/smoke8"
    ;;
  full)
    INPUT_FILE="$DATA_DIR/dev_bird.json"
    GOLD_FILE="$BIRD_DIR/dev.json"
    RUN_DIR="$OUTPUT_ROOT/bird_dev1534"
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

INFER_SCRIPT="$SOURCE_DIR/train_and_evaluate/infer.py"
EVAL_SCRIPT="$SOURCE_DIR/train_and_evaluate/evaluate_bird.py"
DB_PATH="$BIRD_DIR/dev_databases"
GREEDY_FILE="$RUN_DIR/greedy.json"
SAMPLING_FILE="$RUN_DIR/sampling_n8_t08.json"

for required in \
  "$EVAL_PYTHON" "$INFER_SCRIPT" "$EVAL_SCRIPT" "$MODEL_DIR/config.json" \
  "$MODEL_DIR/model.safetensors" "$INPUT_FILE" "$GOLD_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -s "$GREEDY_FILE" ]]; then
  "$EVAL_PYTHON" "$INFER_SCRIPT" \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --input_file "$INPUT_FILE" \
    --output_file "$GREEDY_FILE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --n 1 \
    --temperature 0.0
else
  echo "reusing completed greedy predictions: $GREEDY_FILE"
fi

(
  cd "$(dirname "$EVAL_SCRIPT")"
  "$EVAL_PYTHON" "$EVAL_SCRIPT" \
    --pred "$GREEDY_FILE" \
    --gold "$GOLD_FILE" \
    --db_path "$DB_PATH" \
    --mode greedy_search
) | tee "$RUN_DIR/greedy_evaluation.log"

if [[ "$MODE" == "smoke" ]]; then
  exit 0
fi

if [[ ! -s "$SAMPLING_FILE" ]]; then
  "$EVAL_PYTHON" "$INFER_SCRIPT" \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --input_file "$INPUT_FILE" \
    --output_file "$SAMPLING_FILE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --n 8 \
    --temperature 0.8
else
  echo "reusing completed sampling predictions: $SAMPLING_FILE"
fi

(
  cd "$(dirname "$EVAL_SCRIPT")"
  "$EVAL_PYTHON" "$EVAL_SCRIPT" \
    --pred "$SAMPLING_FILE" \
    --gold "$GOLD_FILE" \
    --db_path "$DB_PATH" \
    --mode major_voting
) | tee "$RUN_DIR/majority_n8_evaluation.log"
