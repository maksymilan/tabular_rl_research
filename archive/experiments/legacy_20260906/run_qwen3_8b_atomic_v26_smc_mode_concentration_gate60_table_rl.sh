#!/usr/bin/env bash
# Isolated SMC mode-concentration diagnostic on table_rl.
# This wrapper reuses the guarded gate60 launcher but selects the new
# probability-ranked correct/wrong trajectory advantage profile.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER="$HERE/run_qwen3_8b_atomic_v26_saam_fourlevel_gate60_table_rl.sh"
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_smc_mode_concentration_20260906}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_smc_mode_concentration_gate60.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/qwen3_v26_smc_mode_concentration_gate60_table_rl_20260906/train}

exec env \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RUNTIME="$RUNTIME" \
  PROJECT_DIR="$RUNTIME" \
  MODEL_PATH="${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}" \
  ADAPTER_PATH="${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/checkpoint-6380}" \
  PROTOCOL_RUNTIME="${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}" \
  TASKS="${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}" \
  CONFIG="$CONFIG" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  RESULT_ADVANTAGE_PROFILE=smc-mode-concentration \
  CREDIT_ASSIGNMENT=trajectory \
  EXPECTED_RECORDS=60 \
  PROMPTS_PER_UPDATE=30 \
  OPTIMIZER_STEPS=4 \
  bash "$BASE_LAUNCHER"
