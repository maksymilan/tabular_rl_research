#!/usr/bin/env bash
set -Eeuo pipefail

# Epochs run serially; the shared runner owns two independent single-GPU services.
RUNTIME_DIR=/home/dengyan/tabular_rl_outputs/runtime/qwen3_4b_cumulative_20260909
CONTROL="$RUNTIME_DIR/dp_actionable_20260911/control"
TRAIN_STATUS=/home/dengyan/tabular_rl_outputs/logs/qwen3_4b_atomic_v26_cumulative_formal_b1_20260910.status.json
MODEL=/home/dengyan/models/Qwen3-4B-TrustSQL-baseline
CHECKPOINT_ROOT=/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1
SOURCE=/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl
DATABASES=/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases
RUNTIME=/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python
RUN_ROOT=/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_atomic_v26
REFERENCE=/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sft6380_actionable_feedback_dev1534_20260908_2333_r2/evaluation/results
QUEUE_STATUS="$RUNTIME_DIR/queue_epoch3_epoch4.status"
export PYTHONPATH="$CONTROL/src" PYTHONDONTWRITEBYTECODE=1
export TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda
trap 'printf "failed exit=%s %s\n" "$?" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$QUEUE_STATUS"' ERR

"$PYTHON" - "$TRAIN_STATUS" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding='utf-8'))
if s.get('success') is not True or int(s.get('expected_global_step', -1)) != 6380:
    raise SystemExit(f'training status did not pass: {s}')
PY

run_epoch() {
  local epoch="$1" step="$2" adapter_sha="$3"
  local dir="$RUN_ROOT/epoch${epoch}_checkpoint${step}_dp_actionable_r1"
  printf 'preflight_epoch%s %s\n' "$epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$QUEUE_STATUS"
  "$PYTHON" -m rl.scenarios.evaluation.run_sft_feedback_regression preflight \
    --run-root "$dir" --control-root "$CONTROL" --asset-contract "$CONTROL/asset_contract_4b.json" \
    --runtime "$RUNTIME" --model "$MODEL" --adapter "$CHECKPOINT_ROOT/checkpoint-$step" \
    --checkpoint-global-step "$step" --adapter-sha256 "$adapter_sha" \
    --source "$SOURCE" --databases "$DATABASES" \
    --baseline "$REFERENCE/merged/all.jsonl" --baseline-manifest "$REFERENCE/gpu0/manifest.json" \
    --baseline-sha256 740d6a723a5fcccdc456baeb2a404a7f2e37fc396de3d1dec8b99a195e69dfe2 \
    --max-tokens 2048 --workers 32 --max-num-seqs 32 --max-model-len 32768 \
    --max-num-batched-tokens 8192 --gpus 0 1 --ports 18330 18331
  printf 'evaluating_epoch%s %s\n' "$epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$QUEUE_STATUS"
  "$PYTHON" -m rl.scenarios.evaluation.run_sft_feedback_regression run \
    --run-root "$dir" --control-root "$CONTROL"
}

run_epoch 3 4785 b95be67f99afbdea05e45ebac744a18f6ac0ca876685d05b903d63e980fd5fa8
run_epoch 4 6380 0bc7b8b644ab8aba0c8f0762ff58bba66f5b7e917182bf5ada10e6b3ce522d19
printf 'completed_epoch3_epoch4 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$QUEUE_STATUS"
