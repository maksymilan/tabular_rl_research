#!/usr/bin/env bash
set -euo pipefail

runtime=/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728
output_root=/home/dengyan/tabular_rl_outputs
diagnostic_root=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_scale20_20260808
indices=data/eval_inputs/routed_coupled_scale20_holdout90_seed20260811.indices.json
runner="$runtime/src/rl/experiments/run_equal300_greedy_table_rl.sh"

mkdir -p "$diagnostic_root"

nohup env \
  RUNTIME="$runtime" OUTPUT_ROOT="$output_root" \
  HARDWARE_PROFILE="$diagnostic_root/rtx3090_24gb.sh" \
  ADAPTER="$diagnostic_root/retry1_lr6e6_adapters/lr_6e-06_neg_0p5_qbatch_5_category_mean_weighted" \
  SERVED_MODEL=routed-coupled-scale20-lr6e6-holdout90 \
  RESULT_DIR="$runtime/data/results/routed_coupled_scale20_lr6e6_version36_holdout90_greedy_20260809" \
  STATUS="$diagnostic_root/scale20_holdout90.status" \
  RUN_LOG="$diagnostic_root/scale20_holdout90.run.log" \
  INDICES_FILE="$indices" EXPECTED=90 EVAL_GPU_ID=0 PORT=18131 \
  RTX3090_GREEDY_WORKERS=16 RTX3090_EVAL_MAX_INFLIGHT=16 \
  SERVER_CONFIG_ID=vllm-version36-routed-coupled-scale20-holdout90-c16 \
  bash "$runner" \
  >"$diagnostic_root/scale20_holdout90.launch.log" 2>&1 &
echo $! >"$diagnostic_root/scale20_holdout90.pid"

nohup env \
  RUNTIME="$runtime" OUTPUT_ROOT="$output_root" \
  HARDWARE_PROFILE="$diagnostic_root/rtx3090_24gb.sh" \
  ADAPTER=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_balance_smoke_20260808/qbatch5_keyweighted_lr12e6_adapters/lr_1e-05_neg_0p5_qbatch_5_category_mean_weighted \
  SERVED_MODEL=routed-coupled-scale5-keyweighted-holdout90 \
  RESULT_DIR="$runtime/data/results/routed_coupled_scale5_keyweighted_version36_holdout90_greedy_20260809" \
  STATUS="$diagnostic_root/scale5_holdout90.status" \
  RUN_LOG="$diagnostic_root/scale5_holdout90.run.log" \
  INDICES_FILE="$indices" EXPECTED=90 EVAL_GPU_ID=1 PORT=18132 \
  RTX3090_GREEDY_WORKERS=16 RTX3090_EVAL_MAX_INFLIGHT=16 \
  SERVER_CONFIG_ID=vllm-version36-routed-coupled-scale5-holdout90-c16 \
  bash "$runner" \
  >"$diagnostic_root/scale5_holdout90.launch.log" 2>&1 &
echo $! >"$diagnostic_root/scale5_holdout90.pid"

echo "launched scale20=$(<"$diagnostic_root/scale20_holdout90.pid") scale5=$(<"$diagnostic_root/scale5_holdout90.pid")"
