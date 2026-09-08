#!/usr/bin/env bash
# User-approved 60-question diagnostic using the frozen remote trainer.
set -Eeuo pipefail
CONTROL_ROOT=${CONTROL_ROOT:?set CONTROL_ROOT to the staged control bundle}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new diagnostic output root}
RUNTIME=${RUNTIME:?set RUNTIME to the isolated frozen training runtime}
SOURCE_RUNTIME=${SOURCE_RUNTIME:?set SOURCE_RUNTIME}
PREVIOUS_TRAIN=${PREVIOUS_TRAIN:?set PREVIOUS_TRAIN to the alpha=0.5 training directory}
EVAL_RUNTIME=${EVAL_RUNTIME:?set EVAL_RUNTIME to the previously used evaluator root}
CONFIG=$RUNTIME/src/rl/configs/experiments/qwen3_v26_saam_alpha025_balanced60.yaml
TASKS=$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_smc_balanced60_train_v1.jsonl
PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python
export PYTHONDONTWRITEBYTECODE=1
source "$CONTROL_ROOT/src/rl/frameworks/launcher/launch_common.sh"
audit() {
  PYTHONPATH="$CONTROL_ROOT/src" "$PYTHON" "$CONTROL_ROOT/archive/diagnostics/20260908_saam_alpha025.py" "$1" \
    --run-root "$RUN_ROOT" --runtime "$RUNTIME" --source-runtime "$SOURCE_RUNTIME" \
    --previous "$PREVIOUS_TRAIN" --config "$CONFIG" --tasks "$TASKS"
}

[[ "${1:---run}" == --preflight ]] && { audit preflight; exit; }
[[ "${1:---run}" == --run ]] || exit 2
[[ -f "$RUN_ROOT/preflight.json" ]] || { printf 'preflight receipt required\n' >&2; exit 2; }
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || exit 2
trap 'set_status "$RUN_ROOT" failed "exit=$?"' ERR
gpu_idle 0 && gpu_idle 1
set_status "$RUN_ROOT" training "alpha=0.25; 60 tasks; 30x8; 4 updates"
env RUNTIME="$RUNTIME" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH=/home/dengyan/models/Qwen3-8B-TrustSQL-baseline \
  ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/checkpoint-6380 \
  TASKS="$TASKS" CONFIG="$CONFIG" OUTPUT_DIR="$RUN_ROOT/train" LOG_ROOT="$RUN_ROOT/logs" \
  TRAIN_GPU=0 VLLM_GPU=1 VLLM_PORT=18285 VLLM_GROUP_PORT=51385 \
  EXPECTED_RECORDS=60 PROMPTS_PER_UPDATE=30 OPTIMIZER_STEPS=4 SPAN_BALANCE_ALPHA=0.25 \
  bash "$RUNTIME/src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_gate60_table_rl.sh"
set_status "$RUN_ROOT" validating "checking effective alpha, actual step and final adapter"
audit trained

set_status "$RUN_ROOT" evaluating "BIRD-dev1534 greedy; GPU0 and GPU1; checkpoint-4"
env PYTHON_BIN="$PYTHON" RUNTIME=/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de \
  ADAPTER="$RUN_ROOT/train/final" SOURCE_CHECKPOINT="$RUN_ROOT/train/checkpoint-4" CHECKPOINT_GLOBAL_STEP=4 \
  RUN_DIR="$RUN_ROOT/evaluation" GPU0=0 GPU1=1 PORT0=18310 PORT1=18311 \
  WRAPPER="$EVAL_RUNTIME/src/rl/evaluation/formal_v26_rollout_passk.py" \
  SHARDER="$EVAL_RUNTIME/src/rl/evaluation/make_eval_shards.py" \
  MERGER="$EVAL_RUNTIME/src/rl/evaluation/merge_eval_shards.py" \
  bash "$CONTROL_ROOT/src/rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh"

set_status "$RUN_ROOT" reporting "paired comparison with SFT6380 and alpha=0.5"
PYTHONPATH="$CONTROL_ROOT/src" "$PYTHON" "$CONTROL_ROOT/src/rl/scenarios/diagnostics/analyze_evaluation_results.py" \
  --examples "$RUN_ROOT/evaluation/input/bird_dev_20240627.table_rl.jsonl" \
  --arm "sft6380=$RUN_ROOT/reference/sft6380.all.jsonl" \
  --arm "alpha025=$RUN_ROOT/evaluation/results/merged/all.jsonl" \
  --arm "alpha05=/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_saam_fourlevel_spanbalanced_smc_balanced60_table_rl_eval_20260907/results/merged/all.jsonl" \
  --compare alpha025:sft6380 --compare alpha025:alpha05 --expected-count 1534 \
  --include-per-example --output "$RUN_ROOT/paired_analysis.json"
set_status "$RUN_ROOT" complete "training and dual-GPU evaluation complete; owned vLLM released"
