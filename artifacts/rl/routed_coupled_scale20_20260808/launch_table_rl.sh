#!/usr/bin/env bash
set -euo pipefail

task_root=/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_scale20_20260808
python_bin=/home/dengyan/miniconda3/envs/trl-table/bin/python
runner="$task_root/run_routed_coupled_reward_smoke.py"
data_root=/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_dense_uniform_seed101
base_model=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
sft2_adapter=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682

mkdir -p "$task_root"

common_args=(
  "$runner"
  --transitions "$data_root/transitions.jsonl"
  --features "$data_root/process_features.jsonl"
  --base-model "$base_model"
  --adapter "$sft2_adapter"
  --train-example-index 3044
  --train-example-index 1188
  --train-example-index 1295
  --train-example-index 5027
  --train-example-index 3102
  --train-example-index 3700
  --train-example-index 4692
  --train-example-index 1626
  --train-example-index 4579
  --train-example-index 2837
  --train-example-index 5040
  --train-example-index 2376
  --train-example-index 2263
  --train-example-index 4050
  --train-example-index 3345
  --train-example-index 3395
  --train-example-index 5623
  --train-example-index 104
  --train-example-index 4505
  --train-example-index 733
  --retention-example-index 3797
  --retention-example-index 4800
  --retention-example-index 2810
  --negative-scale 0.5
  --question-batch-size 5
  --loss-normalization category_mean
  --category-weight correct_key_evidence=2
  --category-weight correct_key_backslice=2
  --category-weight correct_terminal=0.25
  --category-weight severe_local_error=1
  --think-weight 0.5
  --weight-decay 0.01
  --seed 20260808
  --device cuda:0
)

nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$python_bin" "${common_args[@]}" \
  --learning-rate 1.2e-5 \
  --output "$task_root/retry1_lr12e6_train20.json" \
  --save-root "$task_root/retry1_lr12e6_adapters" \
  >"$task_root/retry1_lr12e6_train20.log" 2>&1 &
echo $! >"$task_root/retry1_lr12e6_train20.pid"

nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$python_bin" "${common_args[@]}" \
  --learning-rate 6e-6 \
  --output "$task_root/retry1_lr6e6_train20.json" \
  --save-root "$task_root/retry1_lr6e6_adapters" \
  >"$task_root/retry1_lr6e6_train20.log" 2>&1 &
echo $! >"$task_root/retry1_lr6e6_train20.pid"

echo "launched retry1_lr12e6=$(<"$task_root/retry1_lr12e6_train20.pid") retry1_lr6e6=$(<"$task_root/retry1_lr6e6_train20.pid")"
