#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_exact_prefix_branch_20260802}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_branch_20260802}
PILOT_DIR=${PILOT_DIR:-$OUTPUT_ROOT/pilot8}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
STATUS=$OUTPUT_ROOT/logs/pilot8_finalize.status
SHARD_IDS=${SHARD_IDS:-"0 1"}

printf 'state=waiting_for_shards shard_ids=%s started=%s\n' \
  "$SHARD_IDS" "$(date -Is)" >"$STATUS"
while true; do
  all_complete=true
  for shard_id in $SHARD_IDS; do
    shard_state=$(sed -n 's/^state=\([^ ]*\).*/\1/p' \
      "$OUTPUT_ROOT/logs/pilot8_shard${shard_id}.status" 2>/dev/null || true)
    if [[ $shard_state == failed ]]; then
      printf 'state=blocked shard_id=%s checked=%s\n' \
        "$shard_id" "$(date -Is)" >"$STATUS"
      exit 3
    fi
    if [[ $shard_state != complete ]]; then
      all_complete=false
    fi
  done
  if [[ $all_complete == true ]]; then
    break
  fi
  sleep 30
done

printf 'state=finalizing started=%s\n' "$(date -Is)" >"$STATUS"
cd "$RUNTIME"
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
"$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
  --source-pool "$OUTPUT_ROOT/input/validated_trajectories.jsonl" \
  --source-manifest "$OUTPUT_ROOT/input/manifest.json" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$PILOT_DIR" \
  --limit-trajectories 8 \
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
  --finalize-only
printf 'state=complete completed=%s manifest=%s\n' \
  "$(date -Is)" "$PILOT_DIR/manifest.json" >"$STATUS"
