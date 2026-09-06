#!/usr/bin/env bash
# Sequential FP32 learning-rate sweep after the corrected 8e-7 control completes.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_trustsql_fp32_lr_sweep_20260810}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
RUN_DATE=${RUN_DATE:-20260810}
VARIANT=${VARIANT:-fp32}
EVAL_PY=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASELINE_STATUS=${BASELINE_STATUS:-$O/logs/trustsql_result_only_grpo_scale60_fp32v2_queue_table_rl_20260810.status}
BASELINE_RESULT=${BASELINE_RESULT:-$ER/data/results/trustsql_result_only_grpo_scale60_fp32v2_version36_dev1534_greedy_20260810/all.jsonl}
INNER=$TR/src/rl/experiments/run_trustsql_result_only_grpo_scale60_queue_table_rl.sh
ROOT=${RUN_ROOT:-$O/trustsql_result_only_grpo_fp32_lr_sweep_20260810}
STATUS=${STATUS:-$O/logs/trustsql_result_only_grpo_fp32_lr_sweep_queue_table_rl_20260810.status}
RUN_LOG=${RUN_LOG:-$O/logs/trustsql_result_only_grpo_fp32_lr_sweep_queue_table_rl_20260810.log}
LOCK=${LOCK:-$O/logs/trustsql_result_only_grpo_fp32_lr_sweep_queue_table_rl_20260810.lock}
SUMMARY=${SUMMARY:-$ROOT/full_dev_lr_sweep_analysis.json}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }

mkdir -p "$O/logs" "$ROOT"
exec 9>"$LOCK"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

test -x "$INNER" || test -f "$INNER"
while true; do
  baseline_state=$(state_of "$BASELINE_STATUS")
  case "$baseline_state" in
    complete)
      test -f "$BASELINE_RESULT"
      test "$(wc -l <"$BASELINE_RESULT")" -eq 1534
      break
      ;;
    failed|blocked)
      set_status blocked "corrected baseline state=$baseline_state"
      exit 3
      ;;
    *)
      set_status waiting_baseline "state=$baseline_state"
      sleep 60
      ;;
  esac
done

run_arm() {
  local tag=$1
  local config=$2
  local learning_rate=$3
  local artifact="trl-transition-v26-trustsql-result-only-grpo-sft2-60xk8-${VARIANT}-${tag}-seed20260809"
  local result="$ER/data/results/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}_version36_dev1534_greedy_${RUN_DATE}"
  local inner_status="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}_queue_table_rl_${RUN_DATE}.status"

  set_status running_arm "tag=$tag lr=$learning_rate"
  TRAIN_RUNTIME="$TR" EXPERIMENT_CONFIG="$config" RUN_SMOKE=0 \
  RUN_ROOT="$ROOT/$tag" ARTIFACT="$artifact" \
  STATUS="$inner_status" \
  RUN_LOG="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}_queue_table_rl_${RUN_DATE}.log" \
  VLLM_LOG="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}_train_vllm_${RUN_DATE}.log" \
  VLLM_PORT=8064 VLLM_GROUP_PORT=51264 EVAL_PORT=18100 \
  RESULT="$result" \
  EVAL_STATUS="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}.full_dev_greedy.status" \
  EVAL_RUN_LOG="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}.full_dev_greedy.log" \
  SERVED_MODEL="trustsql-result-only-grpo-scale60-${VARIANT}-${tag}" \
  SUMMARY="$ROOT/$tag/full_dev_analysis.json" \
  LOCK="$O/logs/trustsql_result_only_grpo_scale60_${VARIANT}_${tag}_queue_table_rl_${RUN_DATE}.lock" \
    bash "$INNER"

  test "$(state_of "$inner_status")" = complete
  test -f "$O/checkpoints/$artifact/training_precision.json"
  test -f "$result/all.jsonl"
  test "$(wc -l <"$result/all.jsonl")" -eq 1534
}

run_arm lr2e6 "$TR/src/rl/configs/experiments/trustsql_result_only_grpo_scale60_lr2e6.yaml" 2e-6
run_arm lr4e6 "$TR/src/rl/configs/experiments/trustsql_result_only_grpo_scale60_lr4e6.yaml" 4e-6

set_status summarizing "baseline=8e-7 candidates=2e-6,4e-6"
"$EVAL_PY" "$TR/src/rl/diagnostics/analyze_evaluation_results.py" \
  --examples "$ER/data/eval_inputs/bird_dev_20240627.jsonl" \
  --arm "sft2=$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl" \
  --arm "lr8e7=$BASELINE_RESULT" \
  --arm "lr2e6=$ER/data/results/trustsql_result_only_grpo_scale60_${VARIANT}_lr2e6_version36_dev1534_greedy_${RUN_DATE}/all.jsonl" \
  --arm "lr4e6=$ER/data/results/trustsql_result_only_grpo_scale60_${VARIANT}_lr4e6_version36_dev1534_greedy_${RUN_DATE}/all.jsonl" \
  --compare lr8e7:sft2 --compare lr2e6:sft2 --compare lr4e6:sft2 \
  --compare lr2e6:lr8e7 --compare lr4e6:lr8e7 --compare lr4e6:lr2e6 \
  --expected-count 1534 --protocol-version version36 \
  --temperature 0 --top-p 1 --denotation-comparison bird-set \
  --output "$SUMMARY" --overwrite

set_status complete "summary=$SUMMARY"
