#!/usr/bin/env bash
# TRUST-SQL-style result-only GRPO translated to the frozen atomic mixed-60 cohort.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_trustsql_result_only_scale60_20260809}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
EVAL_PY=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${SFT2_ADAPTER:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
EXPECTED_SFT2_SHA=${EXPECTED_SFT2_SHA:-d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e}
TASKS=${TASKS_JSONL:-$O/phase8_controlled_20260801/balanced_mixed60_dense_uniform_seed101/tasks.jsonl}
CONFIG=${EXPERIMENT_CONFIG:-$TR/src/rl/configs/experiments/trustsql_result_only_grpo_scale60.yaml}
ROOT=${RUN_ROOT:-$O/trustsql_result_only_grpo_scale60_20260809}
ARTIFACT=${ARTIFACT:-trl-transition-v26-trustsql-result-only-grpo-sft2-60xk8-seed20260809}
SMOKE_OUT=${SMOKE_OUT:-$O/checkpoints/${ARTIFACT}-smoke2}
TRAIN_OUT=$O/checkpoints/$ARTIFACT
STATUS=${STATUS:-$O/logs/trustsql_result_only_grpo_scale60_queue_table_rl_20260809.status}
RUN_LOG=${RUN_LOG:-$O/logs/trustsql_result_only_grpo_scale60_queue_table_rl_20260809.log}
VLLM_LOG=${VLLM_LOG:-$O/logs/trustsql_result_only_grpo_scale60_train_vllm_20260809.log}
VLLM_PORT=${VLLM_PORT:-8062}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51262}
EVAL_PORT=${EVAL_PORT:-18098}
RESULT=${RESULT:-$ER/data/results/trustsql_result_only_grpo_scale60_version36_dev1534_greedy_20260809}
EVAL_STATUS=${EVAL_STATUS:-$O/logs/trustsql_result_only_grpo_scale60.full_dev_greedy.status}
EVAL_RUN_LOG=${EVAL_RUN_LOG:-$O/logs/trustsql_result_only_grpo_scale60.full_dev_greedy.log}
SERVED_MODEL=${SERVED_MODEL:-trustsql-result-only-grpo-scale60}
SUMMARY=${SUMMARY:-$ROOT/full_dev_analysis.json}
TRAIN_AUDIT=${TRAIN_AUDIT:-$ROOT/grpo_training_audit.json}
LORA_COMPARISON=${LORA_COMPARISON:-$ROOT/final_lora_update.json}
READINESS=${READINESS:-$ROOT/grpo_baseline_readiness.json}
LOCK=${LOCK:-$O/logs/trustsql_result_only_grpo_scale60_queue_table_rl_20260809.lock}
RUN_SMOKE=${RUN_SMOKE:-1}
SMOKE_GROUP_SIZE=${SMOKE_GROUP_SIZE:-2}
SMOKE_EXAMPLE_INDEX=${SMOKE_EXAMPLE_INDEX:-}
SMOKE_MAX_AGENT_STEPS=${SMOKE_MAX_AGENT_STEPS:-30}
EXPECTED_PROTOCOL_VERSION=${EXPECTED_PROTOCOL_VERSION:-version36}
EXPECTED_PROTOCOL_HASH=${EXPECTED_PROTOCOL_HASH:-20a8d3b4356d883c}
POLICY_REDUCTION=${POLICY_REDUCTION:-trajectory_token_mean}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$(( $1 + 1 ))p"
}
wait_two_gpus() {
  local used0 used1
  while true; do
    used0=$(gpu_used 0); used1=$(gpu_used 1)
    if [[ -n "$used0" && -n "$used1" && "$used0" -le 512 && "$used1" -le 512 ]]; then
      return 0
    fi
    set_status waiting_gpu "gpu0_mib=${used0:-unknown} gpu1_mib=${used1:-unknown}"
    sleep 30
  done
}

vllm_pid=""
stop_training_vllm() {
  if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    # The TRL launcher creates an engine core and resource tracker. Run it in a
    # dedicated session and terminate the whole session between smoke/formal/eval
    # so an orphan cannot retain GPU1 memory.
    kill -TERM -- "-$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  vllm_pid=""
}
start_training_vllm() {
  wait_two_gpus
  set_status starting_vllm "gpu=1 port=$VLLM_PORT"
  cd "$TR"
  setsid env CUDA_VISIBLE_DEVICES=1 \
    PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
    MODEL_PATH="$MODEL" VLLM_PORT="$VLLM_PORT" \
    VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=8192 \
    bash src/rl/frameworks/trl/start_vllm_server.sh >"$VLLM_LOG" 2>&1 &
  vllm_pid=$!
  local ready=0
  for _ in {1..120}; do
    kill -0 "$vllm_pid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || { set_status failed "training vllm readiness failure"; exit 4; }
}
on_exit() {
  local code=$?
  trap - EXIT INT TERM
  stop_training_vllm
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT INT TERM

mkdir -p "$O/logs" "$ROOT"
exec 9>"$LOCK"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

test -f "$TASKS"
test -f "$CONFIG"
test -f "$SFT2/adapter_model.safetensors"
test "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" = "$EXPECTED_SFT2_SHA"
test "$(grep -cve '^[[:space:]]*$' "$TASKS")" -eq 60

if [[ "$RUN_SMOKE" == 1 && ! -f "$SMOKE_OUT/final/adapter_model.safetensors" ]]; then
  if [[ -d "$SMOKE_OUT" ]] && [[ -n "$(find "$SMOKE_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    set_status blocked "incomplete smoke output: $SMOKE_OUT"
    exit 3
  fi
  start_training_vllm
  smoke_selection=(--limit 2)
  if [[ -n "$SMOKE_EXAMPLE_INDEX" ]]; then
    smoke_selection=(--example-index "$SMOKE_EXAMPLE_INDEX")
  fi
  set_status smoke "selection=${SMOKE_EXAMPLE_INDEX:-first2} k=$SMOKE_GROUP_SIZE max_steps=$SMOKE_MAX_AGENT_STEPS reward=1,0.2,0 policy_reduction=$POLICY_REDUCTION"
  CUDA_VISIBLE_DEVICES=0 PROJECT_DIR="$TR" PYTHON="$PY" \
  MODEL_PATH="$MODEL" ADAPTER_PATH="$SFT2" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$SMOKE_OUT" EXPERIMENT_CONFIG="" \
  VLLM_PORT="$VLLM_PORT" VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
      "${smoke_selection[@]}" --reward-mode result-only \
      --result-reward-profile execution-ladder \
      --policy-reduction "$POLICY_REDUCTION" \
      --optimizer-steps 1 --ppo-iterations 1 \
      --prompts-per-update 1 --group-size "$SMOKE_GROUP_SIZE" \
      --gradient-accumulation-steps 1 \
      --optimizer-name adamw_torch --learning-rate 8e-7 \
      --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
      --lr-scheduler-type constant --warmup-ratio 0 \
      --kl-beta 0 --clip-epsilon 0.2 --clip-epsilon-high 0.28 \
      --max-agent-steps "$SMOKE_MAX_AGENT_STEPS" --temperature 0.8 --top-p 1 \
      --transition-micro-batch-size 1 --save-steps 1 --seed 20260809
  test -f "$SMOKE_OUT/final/adapter_model.safetensors"
  test -f "$SMOKE_OUT/training_precision.json"
  "$PY" - "$SMOKE_OUT" "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" "$SMOKE_GROUP_SIZE" "$POLICY_REDUCTION" <<'PY'
import json
import sys
from pathlib import Path

from src.rl.frameworks.trl.transition_batch import standardized_group_advantages

root = Path(sys.argv[1])
expected_version = sys.argv[2]
expected_hash = sys.argv[3]
expected_group_size = int(sys.argv[4])
expected_policy_reduction = sys.argv[5]
manifest = json.loads((root / "run_manifest.json").read_text())
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_hash
assert manifest["policy_reduction"] == expected_policy_reduction
rows = [json.loads(line) for line in (root / "rollouts.jsonl").open() if line.strip()]
assert len(rows) == expected_group_size
assert {row["protocol_version"] for row in rows} == {expected_version}
assert {row["protocol_hash"] for row in rows} == {expected_hash}
rewards = [float(row["result_reward"]["value"]) for row in rows]
assert len(set(rewards)) > 1, rewards
advantages = standardized_group_advantages(rewards, [True] * len(rewards))
assert not any(
    value != 0.0 and abs(value) < 1e-12 for value in advantages
), advantages
state = json.loads((root / "checkpoint-1" / "trainer_state.json").read_text())
step = state["log_history"][-1]
assert float(step["rollout/nonzero_advantage_fraction"]) > 0.0, step
assert float(step["grad_norm"]) > 0.0, step
PY
  "$PY" "$TR/src/rl/diagnostics/compare_lora_updates.py" \
    --reference "$SFT2" --checkpoint "$SMOKE_OUT/final" \
    --output "$SMOKE_OUT/lora_update_comparison.json" >/dev/null
  "$PY" - "$SMOKE_OUT/lora_update_comparison.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
checkpoint = result["checkpoints"][0]
assert float(result["raw_adapter"][checkpoint]["update_norm"]) > 0.0
assert float(result["effective_lora"][checkpoint]["update_norm"]) > 0.0
PY
  stop_training_vllm
fi

if [[ ! -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
  resume_args=()
  if [[ -d "$TRAIN_OUT" ]] && [[ -n "$(find "$TRAIN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    latest=$(find "$TRAIN_OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
      | sort -t- -k2,2n | tail -n 1)
    [[ -n "$latest" ]] || { set_status blocked "formal output has no checkpoint: $TRAIN_OUT"; exit 3; }
    for required in adapter_model.safetensors optimizer.pt trainer_state.json; do
      test -f "$TRAIN_OUT/$latest/$required" || {
        set_status blocked "incomplete resume checkpoint: $TRAIN_OUT/$latest/$required"; exit 3;
      }
    done
    resume_args=(--resume-from-checkpoint "$TRAIN_OUT/$latest")
    set_status resuming "checkpoint=$latest"
  fi
  start_training_vllm
  set_status training "questions=60 epochs=3 k=8 effective_prompts=30 trajectories_per_update=240 reward=1,0.2,0 policy_reduction=$POLICY_REDUCTION"
  CUDA_VISIBLE_DEVICES=0 PROJECT_DIR="$TR" PYTHON="$PY" \
  MODEL_PATH="$MODEL" ADAPTER_PATH="$SFT2" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" \
  VLLM_PORT="$VLLM_PORT" VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
      --transition-micro-batch-size 1 --save-steps 1 --seed 20260809 \
      "${resume_args[@]}"
  test -f "$TRAIN_OUT/final/adapter_model.safetensors"
  test -f "$TRAIN_OUT/training_precision.json"
  "$PY" - "$TRAIN_OUT/run_manifest.json" "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" "$POLICY_REDUCTION" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["protocol_version"] == sys.argv[2]
assert manifest["protocol_hash"] == sys.argv[3]
assert manifest["policy_reduction"] == sys.argv[4]
PY
  stop_training_vllm
fi

test -f "$TRAIN_OUT/final/adapter_model.safetensors"
test -f "$TRAIN_OUT/training_precision.json"
"$PY" "$TR/src/rl/diagnostics/analyze_grpo_training.py" \
  --run-dir "$TRAIN_OUT" --output "$TRAIN_AUDIT" --overwrite >/dev/null
"$PY" "$TR/src/rl/diagnostics/compare_lora_updates.py" \
  --reference "$SFT2" --checkpoint "$TRAIN_OUT/final" \
  --output "$LORA_COMPARISON" >/dev/null

if [[ ! -f "$RESULT/all.jsonl" || "$(wc -l <"$RESULT/all.jsonl")" -ne 1534 ]]; then
  set_status evaluating "questions=1534 greedy=1 gpu=0 concurrency=24"
  ADAPTER="$TRAIN_OUT/final" SERVED_MODEL="$SERVED_MODEL" \
  RESULT_DIR="$RESULT" STATUS="$EVAL_STATUS" \
  RUN_LOG="$EVAL_RUN_LOG" \
  PORT="$EVAL_PORT" EVAL_GPU_ID=0 RUNTIME="$ER" OUTPUT_ROOT="$O" \
    bash "$TR/src/rl/experiments/run_full_dev_greedy_table_rl.sh"
fi

set_status summarizing "candidate_vs=sft2,exp14,exp15,exp19"
analysis_args=(
  --examples "$ER/data/eval_inputs/bird_dev_20240627.jsonl"
  --arm "candidate=$RESULT/all.jsonl"
  --arm "sft2=$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl"
  --arm "exp14=$ER/data/results/stage1_exp14_fixed_process_rank_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl"
  --arm "exp15=$ER/data/results/stage1_exp15_fixed_prefix_action_dpo_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl"
  --compare candidate:sft2 --compare candidate:exp14 --compare candidate:exp15
  --expected-count 1534 --protocol-version "$EXPECTED_PROTOCOL_VERSION"
  --protocol-hash "$EXPECTED_PROTOCOL_HASH"
  --temperature 0 --top-p 1 --denotation-comparison bird-set
  --output "$SUMMARY" --overwrite
)
EXP19_RESULT=${EXP19_RESULT:-$ER/data/results/exp19_result_only_grpo_scale60_version36_dev1534_greedy_20260809/all.jsonl}
if [[ -f "$EXP19_RESULT" ]] && [[ "$(wc -l <"$EXP19_RESULT")" -eq 1534 ]]; then
  analysis_args+=(--arm "exp19=$EXP19_RESULT" --compare candidate:exp19)
fi
"$EVAL_PY" "$TR/src/rl/diagnostics/analyze_evaluation_results.py" "${analysis_args[@]}"
"$PY" "$TR/src/rl/diagnostics/audit_grpo_baseline_readiness.py" \
  --training-audit "$TRAIN_AUDIT" \
  --lora-comparison "$LORA_COMPARISON" \
  --evaluation-analysis "$SUMMARY" \
  --candidate candidate --baseline sft2 --expected-eval-count 1534 \
  --output "$READINESS" --overwrite >/dev/null

set_status complete "train=$TRAIN_OUT result=$RESULT summary=$SUMMARY readiness=$READINESS"
