#!/usr/bin/env bash
# Wait for Exp11 K4 to finish, then evaluate five frozen checkpoints greedily.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
PREVIOUS_STATUS=${PREVIOUS_STATUS:-$OUTPUT_ROOT/logs/rank_score_followup_queue_table_rl_20260731.status}
QUEUE_STATUS=${QUEUE_STATUS:-$OUTPUT_ROOT/logs/stage0_greedy5_queue_table_rl_20260801.status}
QUEUE_LOG=${QUEUE_LOG:-$OUTPUT_ROOT/logs/stage0_greedy5_queue_table_rl_20260801.log}
QUEUE_LOCK=${QUEUE_LOCK:-$OUTPUT_ROOT/logs/stage0_greedy5_queue_table_rl_20260801.lock}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$QUEUE_STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf 'missing\n'; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$QUEUE_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs"
exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then
  set_status failed "duplicate stage0 greedy queue"
  exit 1
fi
exec >>"$QUEUE_LOG" 2>&1

while true; do
  previous=$(state_of "$PREVIOUS_STATUS")
  case "$previous" in
    complete) break ;;
    failed) set_status blocked "Exp11 queue failed; audit before greedy"; exit 3 ;;
    *) set_status waiting_exp11 "state=$previous"; sleep 60 ;;
  esac
done

names=(sft2 exp5_rank exp8_action_only_rank exp10_score_masked exp11_action_mean)
adapters=(
  "$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase2-process-rank-lr1e6-23x4-gated-v2-20260730/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-newgnn-20260731/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp10-rank-conservative-score-masked-lr1e6-23x4-gated-v3-20260731/final"
  "$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp11-rank-conservative-action-mean-lr1e6-23x4-gated-v3-20260731/final"
)

run_one() {
  local index=$1 gpu_id=$2 port=$3
  local name=${names[$index]}
  local adapter=${adapters[$index]}
  local result_dir="$RUNTIME/data/results/stage0_${name}_version36_equal300_greedy_t0_p1_logprobs20_bird_set_20260801"
  local status="$OUTPUT_ROOT/logs/stage0_${name}_equal300_greedy_20260801.status"
  ADAPTER="$adapter" \
  SERVED_MODEL="stage0-${name}-version36-equal300-greedy" \
  SERVER_CONFIG_ID="vllm-stage0-${name}-version36-equal300-greedy" \
  RESULT_DIR="$result_dir" STATUS="$status" \
  RUN_LOG="$OUTPUT_ROOT/logs/stage0_${name}_equal300_greedy_20260801.run.log" \
  PORT="$port" EVAL_GPU_ID="$gpu_id" RUNTIME="$RUNTIME" OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$RUNTIME/src/rl/experiments/run_equal300_greedy_table_rl.sh"
  [[ "$(state_of "$status")" == complete ]] || {
    set_status failed "greedy incomplete model=$name"
    exit 1
  }
}

for ((index = 0; index < ${#names[@]}; index += 2)); do
  set_status evaluating_parallel \
    "models=${names[$index]},${names[$((index + 1))]:-none} gpu=0,1"
  run_one "$index" 0 18071 &
  left_pid=$!
  right_pid=
  if [[ $((index + 1)) -lt ${#names[@]} ]]; then
    run_one "$((index + 1))" 1 18072 &
    right_pid=$!
  fi
  set +e
  wait "$left_pid"; left_code=$?
  right_code=0
  if [[ -n "$right_pid" ]]; then wait "$right_pid"; right_code=$?; fi
  set -e
  if [[ "$left_code" -ne 0 || "$right_code" -ne 0 ]]; then
    set_status failed "parallel greedy batch index=$index left=$left_code right=$right_code"
    exit 1
  fi
done

set_status complete "five equal-300 greedy evaluations complete"
