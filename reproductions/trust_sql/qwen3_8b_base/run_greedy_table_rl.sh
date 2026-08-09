#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODEL_SIZE="${MODEL_SIZE:-8b}"
case "$MODEL_SIZE" in
  4b)
    MODEL_LABEL="qwen3_4b"
    DEFAULT_MODEL_ROOT="/home/dengyan/models/Qwen3-4B-TrustSQL-baseline"
    ;;
  8b)
    MODEL_LABEL="qwen3_8b"
    DEFAULT_MODEL_ROOT="/home/dengyan/models/Qwen3-8B-TrustSQL-baseline"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b" >&2
    exit 2
    ;;
esac
REMOTE_ROOT="${REMOTE_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/trust_sql}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REMOTE_ROOT/runtime/TrustSQL-qwen3-8b-base}"
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
BIRD_ROOT="${BIRD_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
SCOPE="${EVAL_SCOPE:-smoke}"
SEED="${SEED:-20260806}"
BATCH_SIZE="${BATCH_SIZE:-16}"

case "$SCOPE" in
  smoke)
    LIMIT="${SMOKE_LIMIT:-4}"
    RUN_NAME="${MODEL_LABEL}_base_unknown_greedy_smoke${LIMIT}_seed${SEED}"
    ;;
  full)
    LIMIT=0
    RUN_NAME="${MODEL_LABEL}_base_unknown_greedy_dev1534_seed${SEED}"
    ;;
  *)
    echo "EVAL_SCOPE must be smoke or full" >&2
    exit 2
    ;;
esac

INPUT_DIR="$REMOTE_ROOT/data/${MODEL_LABEL}_base"
INPUT_FILE="$INPUT_DIR/${RUN_NAME}.jsonl"
INPUT_MANIFEST="$INPUT_DIR/${RUN_NAME}.manifest.json"
RESULT_DIR="$REMOTE_ROOT/results/$RUN_NAME"
LOG_DIR="$REMOTE_ROOT/logs/${MODEL_LABEL}_base"

test -x "$EVAL_PYTHON"
test -f "$RUNTIME_ROOT/trustsql_eval/main_batch_async.py"
test -f "$MODEL_ROOT/config.json"
test -f "$MODEL_ROOT/model.safetensors.index.json"
test -f "$BIRD_ROOT/dev.json"
test -d "$BIRD_ROOT/dev_databases"

mkdir -p "$INPUT_DIR" "$RESULT_DIR" "$LOG_DIR"

prepare_args=(
  "$SCRIPT_DIR/prepare_bird_dev.py"
  --bird-json "$BIRD_ROOT/dev.json"
  --database-root "$BIRD_ROOT/dev_databases"
  --system-prompt "$RUNTIME_ROOT/trustsql_eval/prompt_template.txt"
  --output "$INPUT_FILE"
  --manifest "$INPUT_MANIFEST"
)
if [[ "$LIMIT" -gt 0 ]]; then
  prepare_args+=(--limit "$LIMIT")
fi
"$EVAL_PYTHON" "${prepare_args[@]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$RUNTIME_ROOT/trustsql_eval"
"$EVAL_PYTHON" main_batch_async.py \
  --input_file "$INPUT_FILE" \
  --output_folder "$RESULT_DIR" \
  --system_prompt_path "$RUNTIME_ROOT/trustsql_eval/prompt_template.txt" \
  --databases_path "$BIRD_ROOT/dev_databases" \
  --documents_path "$INPUT_DIR" \
  --use_vllm \
  --model "$MODEL_ROOT" \
  --template_dir "$MODEL_ROOT" \
  --temperature 0 \
  --top_p 1 \
  --max_new_tokens 4096 \
  --max_rounds 15 \
  --rollout_number 1 \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --repetition_penalty 1.05 \
  --presence_penalty 0.1 \
  --gpu_memory_utilization 0.9 \
  2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"

"$EVAL_PYTHON" "$SCRIPT_DIR/score_bird_dev.py" \
  --results "$RESULT_DIR" \
  --bird-json "$BIRD_ROOT/dev.json" \
  --input-manifest "$INPUT_MANIFEST" \
  --database-root "$BIRD_ROOT/dev_databases" \
  --expected-rollouts 1 \
  --output "$RESULT_DIR/score.json"
