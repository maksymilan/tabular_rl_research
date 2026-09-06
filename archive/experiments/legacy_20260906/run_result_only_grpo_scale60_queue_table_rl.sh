#!/usr/bin/env bash
# Online result-only K=4 GRPO baseline over the frozen balanced mixed-60 task identities.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_result_only_grpo_scale60_20260809}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
EVAL_PY=${EVAL_PYTHON:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${SFT2_ADAPTER:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
EXPECTED_SFT2_SHA=${EXPECTED_SFT2_SHA:-d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e}
TASKS=${TASKS_JSONL:-$O/phase8_controlled_20260801/balanced_mixed60_dense_uniform_seed101/tasks.jsonl}
CONFIG=${EXPERIMENT_CONFIG:-$TR/src/rl/configs/experiments/exp19_result_only_grpo_scale60.yaml}
ROOT=${RUN_ROOT:-$O/result_only_grpo_scale60_20260809}
ARTIFACT=${ARTIFACT:-trl-transition-v26-exp19-result-only-grpo-sft2-60xk4-seed20260809}
TRAIN_OUT=$O/checkpoints/$ARTIFACT
STATUS=${STATUS:-$O/logs/result_only_grpo_scale60_queue_table_rl_20260809.status}
RUN_LOG=${RUN_LOG:-$O/logs/result_only_grpo_scale60_queue_table_rl_20260809.log}
VLLM_LOG=${VLLM_LOG:-$O/logs/result_only_grpo_scale60_train_vllm_20260809.log}
VLLM_PORT=${VLLM_PORT:-8060}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51260}
EVAL_PORT=${EVAL_PORT:-18096}
RESULT=$ER/data/results/exp19_result_only_grpo_scale60_version36_dev1534_greedy_20260809
EVAL_STATUS=$O/logs/exp19_result_only_grpo_scale60.full_dev_greedy.status
SUMMARY=$ROOT/full_dev_analysis.json

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
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  vllm_pid=""
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
exec 9>"$O/logs/result_only_grpo_scale60_queue_table_rl_20260809.lock"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1

test -f "$TASKS"
test -f "$CONFIG"
test -f "$SFT2/adapter_model.safetensors"
test "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" = "$EXPECTED_SFT2_SHA"
test "$(grep -cve '^[[:space:]]*$' "$TASKS")" -eq 60

if [[ ! -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
  if [[ -d "$TRAIN_OUT" ]] && [[ -n "$(find "$TRAIN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    set_status blocked "incomplete nonempty train output requires checkpoint audit: $TRAIN_OUT"
    exit 3
  fi
  wait_two_gpus
  set_status starting_vllm "gpu=1 port=$VLLM_PORT"
  cd "$TR"
  CUDA_VISIBLE_DEVICES=1 \
  PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL" VLLM_PORT="$VLLM_PORT" \
  VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=8192 \
    bash src/rl/frameworks/trl/start_vllm_server.sh >"$VLLM_LOG" 2>&1 &
  vllm_pid=$!
  ready=0
  for _ in {1..120}; do
    kill -0 "$vllm_pid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || { set_status failed "training vllm readiness failure"; exit 4; }

  set_status training "questions=60 online_k4=1 whole_trajectory_advantage=1 lr=1e-6"
  CUDA_VISIBLE_DEVICES=0 PROJECT_DIR="$TR" PYTHON="$PY" \
  MODEL_PATH="$MODEL" ADAPTER_PATH="$SFT2" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" \
  VLLM_PORT="$VLLM_PORT" VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
      --transition-micro-batch-size 1 --save-steps 5 --seed 20260809
  test -f "$TRAIN_OUT/final/adapter_model.safetensors"
  stop_training_vllm
fi

if [[ ! -f "$RESULT/all.jsonl" || "$(wc -l <"$RESULT/all.jsonl")" -ne 1534 ]]; then
  set_status evaluating "questions=1534 greedy=1 gpu=0 concurrency=24"
  ADAPTER="$TRAIN_OUT/final" SERVED_MODEL=exp19-result-only-grpo-scale60 \
  RESULT_DIR="$RESULT" STATUS="$EVAL_STATUS" \
  RUN_LOG="$O/logs/exp19_result_only_grpo_scale60.full_dev_greedy.log" \
  PORT="$EVAL_PORT" EVAL_GPU_ID=0 RUNTIME="$ER" OUTPUT_ROOT="$O" \
    bash "$TR/src/rl/experiments/run_full_dev_greedy_table_rl.sh"
fi

set_status summarizing "candidate_vs=sft2,exp14,exp15,scale20"
"$EVAL_PY" "$TR/src/rl/diagnostics/analyze_evaluation_results.py" \
  --examples "$ER/data/eval_inputs/bird_dev_20240627.jsonl" \
  --arm "candidate=$RESULT/all.jsonl" \
  --arm "sft2=$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl" \
  --arm "exp14=$ER/data/results/stage1_exp14_fixed_process_rank_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
  --arm "exp15=$ER/data/results/stage1_exp15_fixed_prefix_action_dpo_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
  --arm "scale20=$ER/data/results/routed_coupled_scale20_lr6e6_version36_dev1534_greedy_20260809/all.jsonl" \
  --compare candidate:sft2 --compare candidate:exp14 \
  --compare candidate:exp15 --compare candidate:scale20 \
  --expected-count 1534 --protocol-version version36 \
  --temperature 0 --top-p 1 --denotation-comparison bird-set \
  --output "$SUMMARY" --overwrite

set_status complete "train=$TRAIN_OUT result=$RESULT summary=$SUMMARY"
