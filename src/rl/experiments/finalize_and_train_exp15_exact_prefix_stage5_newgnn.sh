#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_exact_prefix_branch_20260802}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_branch_20260802}
STAGE_DIR=${STAGE_DIR:-$OUTPUT_ROOT/scale48_stage5}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
STATUS=$OUTPUT_ROOT/logs/stage5_finalize_and_train.status
LOG=$OUTPUT_ROOT/logs/stage5_finalize_and_train.log
TRAIN_GPU=${TRAIN_GPU:-6}
mkdir -p "$OUTPUT_ROOT/logs"

set_status() {
  printf 'state=%s detail=%s time=%s\n' "$1" "$2" "$(date -Is)" >"$STATUS"
}
on_exit() {
  exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 ]]; then
    set_status failed "exit_code=$exit_code log=$LOG"
  fi
  exit "$exit_code"
}
trap on_exit EXIT
exec >>"$LOG" 2>&1

set_status waiting_for_generation "shards=0,1 anchors=236"
while true; do
  all_complete=true
  for shard_id in 0 1; do
    shard_state=$(sed -n 's/^state=\([^ ]*\).*/\1/p' \
      "$OUTPUT_ROOT/logs/stage5_shard${shard_id}.status" 2>/dev/null || true)
    if [[ "$shard_state" == failed ]]; then
      set_status failed "generation_shard=$shard_id"
      exit 3
    fi
    if [[ "$shard_state" != complete ]]; then
      all_complete=false
    fi
  done
  if [[ "$all_complete" == true ]]; then
    break
  fi
  sleep 30
done

set_status finalizing "anchors=236"
cd "$RUNTIME"
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
"$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
  --source-pool "$OUTPUT_ROOT/input/validated_trajectories.jsonl" \
  --source-manifest "$OUTPUT_ROOT/input/manifest.json" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$STAGE_DIR" \
  --limit-trajectories 0 \
  --anchors-per-trajectory 5 \
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

set_status building_union "original_plus_expanded_stage5"
TRAINING_DIR=$STAGE_DIR/training
mkdir -p "$TRAINING_DIR"
"$PYTHON_BIN" src/rl/action_dpo/build_expanded_exact_prefix_union.py \
  --original-dataset "$OUTPUT_ROOT/input/original_exp15/fixed_prefix_train.verified.jsonl" \
  --original-audit "$OUTPUT_ROOT/input/original_exp15/verification_audit.json" \
  --expanded-dataset "$STAGE_DIR/exact_prefix_pairs.jsonl" \
  --expanded-manifest "$STAGE_DIR/manifest.json" \
  --source-replay-audit "$STAGE_DIR/source_replay_audit.json" \
  --output-dataset "$TRAINING_DIR/expanded_exp15_union.verified.jsonl" \
  --output-audit "$TRAINING_DIR/expanded_exp15_union.verification_audit.json" \
  --min-expanded-pairs 36 \
  --min-expanded-questions 20 \
  --min-new-unique-pairs 24

chmod 444 \
  "$STAGE_DIR/plan.json" \
  "$STAGE_DIR/source_replay_audit.json" \
  "$STAGE_DIR/manifest.json" \
  "$STAGE_DIR/exact_prefix_pairs.jsonl" \
  "$STAGE_DIR/exact_prefix_pairs.online_supported.jsonl" \
  "$STAGE_DIR/exact_prefix_pairs.online_consistent.jsonl" \
  "$TRAINING_DIR/expanded_exp15_union.verified.jsonl" \
  "$TRAINING_DIR/expanded_exp15_union.verification_audit.json"

set_status launching_training "gpu=$TRAIN_GPU"
chmod +x "$OUTPUT_ROOT/run_exp15_exact_prefix_expanded_train_newgnn.sh"
DATA_DIR="$TRAINING_DIR" \
ARTIFACT=trl-transition-v26-exp15-exact-prefix-stage5-expanded-union-action-dpo-sft2-seed101-20260802 \
"$OUTPUT_ROOT/run_exp15_exact_prefix_expanded_train_newgnn.sh" "$TRAIN_GPU"
set_status complete "training_status=$OUTPUT_ROOT/logs/expanded_exp15_train.status"
trap - EXIT
