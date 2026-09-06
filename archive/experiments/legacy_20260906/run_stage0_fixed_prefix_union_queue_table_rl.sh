#!/usr/bin/env bash
# Rebuild the dev fixed-prefix core from all locally available K4 trajectories, then score five models.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
EXP11_STATUS=${EXP11_STATUS:-$OUTPUT_ROOT/logs/rank_score_followup_queue_table_rl_20260731.status}
OUT_DIR=${OUT_DIR:-$RUNTIME/data/diagnostics/fixed_prefix_dev300_union4_v2}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.log}
LOCK=${LOCK:-$OUTPUT_ROOT/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.lock}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf 'missing\n'; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$OUT_DIR"
exec 8>"$LOCK"
if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"
while true; do
  state=$(state_of "$EXP11_STATUS")
  case "$state" in
    complete) break ;;
    failed) set_status blocked "Exp11 failed; fixed-prefix union cannot be frozen"; exit 3 ;;
    *) set_status waiting_exp11 "state=$state"; sleep 60 ;;
  esac
done

results="$RUNTIME/data/results"
inputs=(
  "$results/qwen25_coder7b_sft2_step1682_tool_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set/all.jsonl"
  "$results/trl-transition-v26-phase2-process-rank-lr1e6-23x4-gated-v2-20260730_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set/all.jsonl"
  "$results/trl-transition-v26-exp10-rank-conservative-score-masked-lr1e6-23x4-gated-v3-20260731_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set/all.jsonl"
  "$results/trl-transition-v26-exp11-rank-conservative-action-mean-lr1e6-23x4-gated-v3-20260731_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set/all.jsonl"
)
for input in "${inputs[@]}"; do [[ "$(wc -l < "$input")" -eq 300 ]]; done
dataset="$OUT_DIR/fixed_prefix_dev300.jsonl"
manifest="$OUT_DIR/manifest.json"
if [[ ! -f "$dataset" ]]; then
  set_status building "sources=4 exact_visible_state_only=1"
  input_args=()
  for input in "${inputs[@]}"; do input_args+=(--input "$input"); done
  "$PYTHON_BIN" src/rl/diagnostics/build_fixed_prefix_dataset.py \
    "${input_args[@]}" --output "$dataset" --manifest "$manifest" \
    --max-per-question 4 --seed 101
fi

pairs=$(wc -l < "$dataset" | tr -d '[:space:]')
[[ "$pairs" -gt 0 ]]
set_status scoring "pairs=$pairs checkpoints=5 gpu=1"
DATASET="$dataset" RESULT_ROOT="$OUT_DIR/scores" GPU_ID=1 \
QUEUE_STATUS="$OUT_DIR/rescore.status" QUEUE_LOG="$OUT_DIR/rescore.log" \
QUEUE_LOCK="$OUT_DIR/rescore.lock" RUNTIME="$RUNTIME" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash src/rl/experiments/run_fixed_prefix_rescore5_queue_table_rl.sh
[[ "$(state_of "$OUT_DIR/rescore.status")" == complete ]]
set_status complete "pairs=$pairs checkpoints=5 result=$OUT_DIR/scores/checkpoint_comparison.json"
