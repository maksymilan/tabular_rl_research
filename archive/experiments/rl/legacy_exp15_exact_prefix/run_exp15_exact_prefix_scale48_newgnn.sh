#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX=${1:?usage: run_exp15_exact_prefix_scale48_newgnn.sh SHARD_INDEX GPU_ID}
GPU_ID=${2:?usage: run_exp15_exact_prefix_scale48_newgnn.sh SHARD_INDEX GPU_ID}
SHARD_COUNT=${SHARD_COUNT:-2}

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_exact_prefix_branch_20260802}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_branch_20260802}
SCALE_DIR=${SCALE_DIR:-$OUTPUT_ROOT/scale48}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}

SOURCE_POOL=$OUTPUT_ROOT/input/validated_trajectories.jsonl
SOURCE_MANIFEST=$OUTPUT_ROOT/input/manifest.json
LOG_DIR=$OUTPUT_ROOT/logs
STATUS=$LOG_DIR/scale48_shard${SHARD_INDEX}.status
mkdir -p "$LOG_DIR" "$SCALE_DIR"

printf 'state=running shard=%s/%s gpu=%s started=%s\n' \
  "$SHARD_INDEX" "$SHARD_COUNT" "$GPU_ID" "$(date -Is)" >"$STATUS"

cd "$RUNTIME"
set +e
CUDA_VISIBLE_DEVICES=$GPU_ID \
HF_HUB_OFFLINE=1 \
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
"$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
  --source-pool "$SOURCE_POOL" \
  --source-manifest "$SOURCE_MANIFEST" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$SCALE_DIR" \
  --limit-trajectories 0 \
  --anchors-per-trajectory 3 \
  --candidate-count 4 \
  --candidate-draws 8 \
  --continuation-count 2 \
  --max-pairs-per-anchor 2 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-steps 30 \
  --max-new-tokens 1024 \
  --max-context-tokens 8192 \
  --history-turns 4 \
  --seed 1515 \
  --gpu-memory-utilization 0.90 \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX" \
  --no-finalize
exit_code=$?
set -e

if [[ $exit_code -eq 0 ]]; then
  printf 'state=complete shard=%s/%s gpu=%s completed=%s\n' \
    "$SHARD_INDEX" "$SHARD_COUNT" "$GPU_ID" "$(date -Is)" >"$STATUS"
else
  printf 'state=failed shard=%s/%s gpu=%s exit_code=%s failed=%s\n' \
    "$SHARD_INDEX" "$SHARD_COUNT" "$GPU_ID" "$exit_code" "$(date -Is)" >"$STATUS"
fi
exit "$exit_code"
