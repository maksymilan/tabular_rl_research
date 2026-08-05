#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX=${1:?usage: run_exp15_exact_prefix_strict4_seed2603_newgnn.sh SHARD_INDEX GPU_ID}
GPU_ID=${2:?usage: run_exp15_exact_prefix_strict4_seed2603_newgnn.sh SHARD_INDEX GPU_ID}
SHARD_COUNT=${SHARD_COUNT:-2}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_strict4_20260803}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_branch_20260802}
STRICT_ROOT=${STRICT_ROOT:-$OUTPUT_ROOT/strict4_seed2603}
GENERATION_DIR=${GENERATION_DIR:-$STRICT_ROOT/generation_turn1plus}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
SOURCE_POOL=$OUTPUT_ROOT/input/validated_trajectories.jsonl
SOURCE_MANIFEST=$OUTPUT_ROOT/input/manifest.json
QUESTION_IDS_DATASET=$OUTPUT_ROOT/input/original_exp15/fixed_prefix_train.verified.jsonl
STATUS=$OUTPUT_ROOT/logs/strict4_seed2603_shard${SHARD_INDEX}.status
RUN_LOG=$OUTPUT_ROOT/logs/strict4_seed2603_shard${SHARD_INDEX}.run.log
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() {
  printf '%s\t%s\tshard=%s/%s gpu=%s %s\n' \
    "$(timestamp)" "$1" "$SHARD_INDEX" "$SHARD_COUNT" "$GPU_ID" "$2" >"$STATUS"
}
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

wait_for_gpu() {
  local used pids
  for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
    used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk '/^[0-9]+$/ {print}' | paste -sd, - || true)
    if [[ -n "$used" && "$used" -le "$GPU_FREE_THRESHOLD_MIB" && -z "$pids" ]]; then
      return 0
    fi
    set_status waiting_gpu "memory_mib=${used:-unknown} pids=${pids:-none}"
    sleep 30
  done
  return 1
}

mkdir -p "$OUTPUT_ROOT/logs" "$GENERATION_DIR"
exec >>"$RUN_LOG" 2>&1
wait_for_gpu
set_status running "anchors_per_trajectory=8 continuation_count=4 seed=2603"
cd "$RUNTIME"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
"$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
  --source-pool "$SOURCE_POOL" \
  --source-manifest "$SOURCE_MANIFEST" \
  --question-ids-dataset "$QUESTION_IDS_DATASET" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$GENERATION_DIR" \
  --limit-trajectories 0 \
  --anchors-per-trajectory 8 \
  --min-anchor-turn-index 1 \
  --candidate-count 4 \
  --candidate-draws 12 \
  --continuation-count 4 \
  --max-pairs-per-anchor 2 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-steps 30 \
  --max-new-tokens 1024 \
  --max-context-tokens 8192 \
  --history-turns 4 \
  --seed 2603 \
  --gpu-memory-utilization 0.90 \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX" \
  --no-finalize
set_status complete "generation_worker_complete"
trap - EXIT
