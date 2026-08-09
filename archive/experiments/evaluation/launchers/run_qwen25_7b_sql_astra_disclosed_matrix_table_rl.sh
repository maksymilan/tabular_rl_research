#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/dengyan/tabular_rl_outputs/sql_astra_reproduction_matrix_20260805}"
LAUNCHER="$PROJECT_DIR/archive/experiments/evaluation/launchers/run_qwen25_7b_sql_astra_disclosed_arm_table_rl.sh"
LOG_ROOT="${LOG_ROOT:-/home/dengyan/tabular_rl_outputs/logs/sql_astra_reproduction_matrix_20260805}"

mkdir -p "$LOG_ROOT"

env \
  PROJECT_DIR="$PROJECT_DIR" \
  GPU_ID=0 \
  PORT=18320 \
  RUN_LABEL=greedy_t0_p1 \
  TEMPERATURE=0 \
  TOP_P=1 \
  SMOKE_N=32 \
  bash "$LAUNCHER" &
gpu0_pid=$!

env \
  PROJECT_DIR="$PROJECT_DIR" \
  GPU_ID=1 \
  PORT=18321 \
  RUN_LABEL=validation_t06_p095 \
  TEMPERATURE=0.6 \
  TOP_P=0.95 \
  SMOKE_N=32 \
  bash "$LAUNCHER"

env \
  PROJECT_DIR="$PROJECT_DIR" \
  GPU_ID=1 \
  PORT=18321 \
  RUN_LABEL=validation_t10_p095 \
  TEMPERATURE=1.0 \
  TOP_P=0.95 \
  SMOKE_N=0 \
  bash "$LAUNCHER"

wait "$gpu0_pid"
