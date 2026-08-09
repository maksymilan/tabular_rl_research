#!/usr/bin/env bash
set -euo pipefail

runtime=/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728
output_root=/home/dengyan/tabular_rl_outputs
diagnostic_root=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_scale20_20260808
runner=/home/dengyan/tabular_rl_outputs/rl_runtime_streaming_teacher_union_scale120_v2_20260804/src/rl/experiments/run_full_dev_greedy_table_rl.sh
hardware_profile="$diagnostic_root/rtx3090_24gb.sh"

mkdir -p "$diagnostic_root"
test -f "$hardware_profile"

nohup env \
  RUNTIME="$runtime" OUTPUT_ROOT="$output_root" \
  HARDWARE_PROFILE="$hardware_profile" \
  ADAPTER=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_balance_smoke_20260808/qbatch5_keyweighted_lr12e6_adapters/lr_1e-05_neg_0p5_qbatch_5_category_mean_weighted \
  ADAPTER_SHA256=93c04a0230140efcfac07653b9fe40fb8b1bba8cdae637f9af1d2bbc107526af \
  SERVED_MODEL=routed-coupled-scale5-keyweighted-full-dev \
  RESULT_DIR="$runtime/data/results/routed_coupled_scale5_keyweighted_version36_dev1534_greedy_20260809" \
  STATUS="$diagnostic_root/scale5_full_dev.status" \
  RUN_LOG="$diagnostic_root/scale5_full_dev.run.log" \
  VLLM_LOG="$diagnostic_root/scale5_full_dev.vllm.log" \
  VLLM_PID_FILE="$diagnostic_root/scale5_full_dev.vllm.pid" \
  EVAL_LOG="$diagnostic_root/scale5_full_dev.eval.log" \
  EVAL_GPU_ID=0 PORT=18141 \
  RTX3090_GREEDY_WORKERS=24 RTX3090_EVAL_MAX_INFLIGHT=24 \
  SERVER_CONFIG_ID=vllm-version36-routed-coupled-scale5-full-dev-c24-logprobs20 \
  bash "$runner" \
  >"$diagnostic_root/scale5_full_dev.launch.log" 2>&1 &
echo $! >"$diagnostic_root/scale5_full_dev.pid"

nohup env \
  RUNTIME="$runtime" OUTPUT_ROOT="$output_root" \
  HARDWARE_PROFILE="$hardware_profile" \
  ADAPTER=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_scale20_20260808/retry1_lr6e6_adapters/lr_6e-06_neg_0p5_qbatch_5_category_mean_weighted \
  ADAPTER_SHA256=3fe16025871ae816014f91d026237644e63555cbc2c6cfa73eb95c610273a286 \
  SERVED_MODEL=routed-coupled-scale20-lr6e6-full-dev \
  RESULT_DIR="$runtime/data/results/routed_coupled_scale20_lr6e6_version36_dev1534_greedy_20260809" \
  STATUS="$diagnostic_root/scale20_full_dev.status" \
  RUN_LOG="$diagnostic_root/scale20_full_dev.run.log" \
  VLLM_LOG="$diagnostic_root/scale20_full_dev.vllm.log" \
  VLLM_PID_FILE="$diagnostic_root/scale20_full_dev.vllm.pid" \
  EVAL_LOG="$diagnostic_root/scale20_full_dev.eval.log" \
  EVAL_GPU_ID=1 PORT=18142 \
  RTX3090_GREEDY_WORKERS=24 RTX3090_EVAL_MAX_INFLIGHT=24 \
  SERVER_CONFIG_ID=vllm-version36-routed-coupled-scale20-full-dev-c24-logprobs20 \
  bash "$runner" \
  >"$diagnostic_root/scale20_full_dev.launch.log" 2>&1 &
echo $! >"$diagnostic_root/scale20_full_dev.pid"

printf 'launched scale5=%s scale20=%s\n' \
  "$(<"$diagnostic_root/scale5_full_dev.pid")" \
  "$(<"$diagnostic_root/scale20_full_dev.pid")"
