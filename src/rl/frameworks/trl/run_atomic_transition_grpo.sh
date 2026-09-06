#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local base model}
ADAPTER_PATH=${ADAPTER_PATH:?set ADAPTER_PATH to the frozen SFT adapter}
EXAMPLES_JSON=${EXAMPLES_JSON:?set EXAMPLES_JSON to an immutable RL task artifact}
OUTPUT_DIR=${OUTPUT_DIR:?set an isolated OUTPUT_DIR}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-}
REWARD_MODE=${REWARD_MODE:-result-only}
PROCESS_REWARD_CONFIG=${PROCESS_REWARD_CONFIG:-$PROJECT_DIR/src/rl/configs/atomic_process_reward.json}
PROCESS_ADMISSION_POLICY=${PROCESS_ADMISSION_POLICY:-counterfactual-completeness}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-}
FIXED_ROLLOUT_POOL=${FIXED_ROLLOUT_POOL:-}
FIXED_POOL_MANIFEST=${FIXED_POOL_MANIFEST:-}
EXCLUDE_EMPTY_REFERENCE_RESULTS=${EXCLUDE_EMPTY_REFERENCE_RESULTS:-0}
OPTIMIZER_STEPS=${OPTIMIZER_STEPS:-200}
PPO_ITERATIONS=${PPO_ITERATIONS:-2}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-1}
GROUP_SIZE=${GROUP_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-3e-7}
KL_BETA=${KL_BETA:-0.001}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_PORT:-8000}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51216}
# Avoid leaking vLLM's internal distributed-master `VLLM_PORT` variable into
# the trainer process; the server port is passed explicitly to TRL below.
unset VLLM_PORT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/src/rl:$PROJECT_DIR/src/eval:$PROJECT_DIR/src/harness:$PROJECT_DIR/src/sft:${PYTHONPATH:-}"

experiment_args=()
if [[ -n "$EXPERIMENT_CONFIG" ]]; then
  experiment_args+=(--experiment-config "$EXPERIMENT_CONFIG")
else
  experiment_args+=(
    --reward-mode "$REWARD_MODE"
    --optimizer-steps "$OPTIMIZER_STEPS"
    --ppo-iterations "$PPO_ITERATIONS"
    --prompts-per-update "$PROMPTS_PER_UPDATE"
    --group-size "$GROUP_SIZE"
    --learning-rate "$LEARNING_RATE"
    --kl-beta "$KL_BETA"
    --temperature 1.0
    --top-p 1.0
  )
  if [[ "$EXCLUDE_EMPTY_REFERENCE_RESULTS" == "1" ]]; then
    experiment_args+=(--exclude-empty-reference-results)
  fi
  if [[ "$REWARD_MODE" == "process" ]]; then
    experiment_args+=(--process-reward-config "$PROCESS_REWARD_CONFIG")
    experiment_args+=(--process-admission-policy "$PROCESS_ADMISSION_POLICY")
  fi
fi
if [[ -n "$COUNTERFACTUAL_SUITE_MANIFEST" ]]; then
  experiment_args+=(--counterfactual-suite-manifest "$COUNTERFACTUAL_SUITE_MANIFEST")
fi
if [[ -n "$FIXED_ROLLOUT_POOL" || -n "$FIXED_POOL_MANIFEST" ]]; then
  [[ -n "$FIXED_ROLLOUT_POOL" && -n "$FIXED_POOL_MANIFEST" ]] || {
    printf 'FIXED_ROLLOUT_POOL and FIXED_POOL_MANIFEST must be set together\n' >&2
    exit 2
  }
  experiment_args+=(--fixed-rollout-pool "$FIXED_ROLLOUT_POOL")
  experiment_args+=(--fixed-pool-manifest "$FIXED_POOL_MANIFEST")
fi

cd "$PROJECT_DIR"
"$PYTHON" src/rl/frameworks/trl/run_transition_grpo.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --examples-json "$EXAMPLES_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --vllm-host "$VLLM_HOST" \
  --vllm-port "$VLLM_SERVER_PORT" \
  --vllm-group-port "$VLLM_GROUP_PORT" \
  "${experiment_args[@]}" \
  "$@"
