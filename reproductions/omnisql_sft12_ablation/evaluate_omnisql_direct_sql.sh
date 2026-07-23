#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
EVAL_PYTHON=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
SOURCE_DIR=${SOURCE_DIR:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql/source/OmniSQL}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/merged/omnisql_7b}
DATA_DIR=${DATA_DIR:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql/data}
BIRD_DIR=${BIRD_DIR:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/results/omnisql_direct_sql}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0,1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

case "$MODE" in
  smoke)
    INPUT_FILE="$DATA_DIR/dev_bird_smoke8.json"
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
PRED_FILE="$RUN_DIR/greedy.json"
for required in "$EVAL_PYTHON" "$INFER_SCRIPT" "$EVAL_SCRIPT" \
  "$MODEL_DIR/config.json" "$INPUT_FILE" "$GOLD_FILE"; do
  test -e "$required" || { echo "missing required artifact: $required" >&2; exit 1; }
done

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | sed '/^[[:space:]]*$/d')
if [[ -n "$gpu_pids" ]]; then
  echo "refusing to launch while GPU compute processes are active: $gpu_pids" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -s "$PRED_FILE" ]]; then
  "$EVAL_PYTHON" "$INFER_SCRIPT" \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --input_file "$INPUT_FILE" \
    --output_file "$PRED_FILE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --n 1 \
    --temperature 0.0
fi

(
  cd "$(dirname "$EVAL_SCRIPT")"
  "$EVAL_PYTHON" "$EVAL_SCRIPT" \
    --pred "$PRED_FILE" \
    --gold "$GOLD_FILE" \
    --db_path "$DB_PATH" \
    --mode greedy_search
) | tee "$RUN_DIR/greedy_evaluation.log"
