#!/usr/bin/env bash
# Controlled BIRD result-only/process RL launcher over the same SFT-derived task cohort.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/sft/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:?set ADAPTER_PATH to the frozen SFT initialization adapter}
EXAMPLES_JSON=${EXAMPLES_JSON:?set EXAMPLES_JSON to a build_sft_task_set.py artifact}
OUTPUT_DIR=${OUTPUT_DIR:?set an isolated OUTPUT_DIR}
REWARD_MODE=${REWARD_MODE:-process}
PROCESS_REWARD_CONFIG=${PROCESS_REWARD_CONFIG:-$PROJECT_DIR/src/rl/configs/atomic_process_reward.json}
COUNTERFACTUAL_SUITE_MANIFEST=${COUNTERFACTUAL_SUITE_MANIFEST:-}
KL_BETA=${KL_BETA:-0}
STEPS=${STEPS:-200}
GROUP_SIZE=${GROUP_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}

if [[ "$REWARD_MODE" != "result-only" && "$REWARD_MODE" != "process" ]]; then
  printf 'REWARD_MODE must be result-only or process, got %s\n' "$REWARD_MODE" >&2
  exit 2
fi

mkdir -p /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_DIR/src/rl:$PROJECT_DIR/src/eval:$PROJECT_DIR/src/harness:$PROJECT_DIR/src/sft:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
reward_args=(--reward-mode "$REWARD_MODE")
if [[ "$REWARD_MODE" == "process" ]]; then
  if [[ -z "$COUNTERFACTUAL_SUITE_MANIFEST" ]]; then
    printf 'COUNTERFACTUAL_SUITE_MANIFEST is required for process RL\n' >&2
    exit 2
  fi
  reward_args+=(--process-reward-config "$PROCESS_REWARD_CONFIG")
  reward_args+=(--counterfactual-suite-manifest "$COUNTERFACTUAL_SUITE_MANIFEST")
fi

"$PYTHON" src/rl/frameworks/accelerate/group_reinforce.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --examples-json "$EXAMPLES_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --group-size "$GROUP_SIZE" \
  --rollout-batch-size "$GROUP_SIZE" \
  --logprob-micro-batch-size 2 \
  --train-turns all \
  --learning-rate "$LEARNING_RATE" \
  --lr-scheduler-type "$LR_SCHEDULER_TYPE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --kl-beta "$KL_BETA" \
  --denotation-comparison bird-set \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --max-steps 30 \
  --max-new-tokens 1024 \
  --max-context-tokens 8192 \
  "${reward_args[@]}" \
  "$@"
