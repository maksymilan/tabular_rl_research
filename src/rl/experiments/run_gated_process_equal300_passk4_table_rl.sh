#!/usr/bin/env bash
# Matched BIRD equal-300 K=4 evaluation for one gated process-RL checkpoint.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
ADAPTER=${ADAPTER:?set ADAPTER}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
SERVED_MODEL=${SERVED_MODEL:?set SERVED_MODEL}
STATUS=${STATUS:?set STATUS}
RUN_LOG=${RUN_LOG:?set RUN_LOG}
PORT=${PORT:-18058}
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
EVAL_PYTHON=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}

cd "$RUNTIME"
RUNTIME="$RUNTIME" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
RESULT_DIR="$RESULT_DIR" \
ADAPTER="$ADAPTER" \
SERVED_MODEL="$SERVED_MODEL" \
STATUS="$STATUS" \
RUN_LOG="$RUN_LOG" \
PORT="$PORT" \
EVAL_GPU_ID="$EVAL_GPU_ID" \
EVAL_PYTHON="$EVAL_PYTHON" \
BASE_MODEL="$BASE_MODEL" \
SERVER_CONFIG_ID="vllm-generation-config-${EXPERIMENT_NAME}-version36-equal300-passk4-logprobs20" \
  bash src/rl/experiments/run_phase1_result_only_equal300_passk4_table_rl.sh
