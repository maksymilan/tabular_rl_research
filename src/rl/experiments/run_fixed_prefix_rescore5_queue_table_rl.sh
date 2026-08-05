#!/usr/bin/env bash
# Score the same verifier-backed fixed-prefix pairs with five frozen checkpoints.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
DATASET=${DATASET:-$RUNTIME/data/diagnostics/fixed_prefix_dev300_v1/fixed_prefix_dev300.jsonl}
RESULT_ROOT=${RESULT_ROOT:-$RUNTIME/data/diagnostics/fixed_prefix_dev300_v1/scores}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
QUEUE_STATUS=${QUEUE_STATUS:-$OUTPUT_ROOT/logs/fixed_prefix_rescore5_queue_table_rl_20260801.status}
QUEUE_LOG=${QUEUE_LOG:-$OUTPUT_ROOT/logs/fixed_prefix_rescore5_queue_table_rl_20260801.log}
QUEUE_LOCK=${QUEUE_LOCK:-$OUTPUT_ROOT/logs/fixed_prefix_rescore5_queue_table_rl_20260801.lock}
GPU_ID=${GPU_ID:-1}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$QUEUE_STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$QUEUE_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$RESULT_ROOT"
exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then set_status failed "duplicate fixed-prefix scoring queue"; exit 1; fi
exec >>"$QUEUE_LOG" 2>&1
cd "$RUNTIME"
test -f "$DATASET"
test -d "$BASE_MODEL"
expected=$(wc -l <"$DATASET" | tr -d '[:space:]')
[[ "$expected" -gt 0 ]] || { set_status failed "empty fixed-prefix dataset"; exit 1; }

names=(sft2 exp5_rank exp8_action_only_rank exp10_score_masked exp11_action_mean)
adapters=(
  "$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase2-process-rank-lr1e6-23x4-gated-v2-20260730/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-newgnn-20260731/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp10-rank-conservative-score-masked-lr1e6-23x4-gated-v3-20260731/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp11-rank-conservative-action-mean-lr1e6-23x4-gated-v3-20260731/final"
)

for index in "${!names[@]}"; do
  name=${names[$index]}
  adapter=${adapters[$index]}
  output_jsonl="$RESULT_ROOT/${name}.jsonl"
  output_parquet="$RESULT_ROOT/${name}.parquet"
  test -f "$adapter/adapter_model.safetensors"
  if [[ -f "$output_jsonl" && -f "$output_parquet" ]] \
    && [[ "$(wc -l <"$output_jsonl" | tr -d '[:space:]')" -eq "$expected" ]]; then
    set_status scoring "ordinal=$((index + 1))/5 checkpoint=$name already_complete=1"
    continue
  fi
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
    if [[ -n "$used" && "$used" -le "$GPU_FREE_THRESHOLD_MIB" ]]; then break; fi
    set_status waiting_gpu "checkpoint=$name gpu=$GPU_ID mib=${used:-unknown}"
    sleep 30
  done
  set_status scoring "ordinal=$((index + 1))/5 checkpoint=$name pairs=$expected gpu=$GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    src/rl/diagnostics/score_action_candidates.py \
    --dataset "$DATASET" --checkpoint-name "$name" --base-model "$BASE_MODEL" \
    --adapter "$adapter" --output-jsonl "$output_jsonl" \
    --output-parquet "$output_parquet" --device cuda:0
  [[ "$(wc -l <"$output_jsonl" | tr -d '[:space:]')" -eq "$expected" ]]
done

set_status summarizing "checkpoints=5 pairs=$expected"
"$PYTHON_BIN" src/rl/diagnostics/summarize_action_margins.py \
  --reference sft2 --output-dir "$RESULT_ROOT" \
  --scores "sft2=$RESULT_ROOT/sft2.jsonl" \
  --scores "exp5_rank=$RESULT_ROOT/exp5_rank.jsonl" \
  --scores "exp8_action_only_rank=$RESULT_ROOT/exp8_action_only_rank.jsonl" \
  --scores "exp10_score_masked=$RESULT_ROOT/exp10_score_masked.jsonl" \
  --scores "exp11_action_mean=$RESULT_ROOT/exp11_action_mean.jsonl"
set_status complete "checkpoints=5 pairs=$expected result=$RESULT_ROOT"
