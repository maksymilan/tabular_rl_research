#!/bin/zsh
set -u

ROOT=/Users/hudou/Research/tabular_rl_research
cd "$ROOT" || exit 1
mkdir -p logs/sft_scale1000c

# Keep the machine awake while launchd owns this batch. Each generator is independently resumable.
caffeinate -dimsu -w $$ &

.venv/bin/python -u src/sft/rollout_external_data.py \
  --examples-file data/eval_inputs/bird_train_sft1_scale1000c_buckets/easy.jsonl \
  --limit 400 --model deepseek-v4-flash \
  --out data/trajectories/bird_scale1000c_flash_easy_success.jsonl \
  --workers 4 --max-steps 30 --max-errors-per-type 3 --attempts-per-example 1 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full \
  --resume > logs/sft_scale1000c/easy.log 2>&1 &
easy_pid=$!

.venv/bin/python -u src/sft/rollout_external_data.py \
  --examples-file data/eval_inputs/bird_train_sft1_scale1000c_buckets/medium.jsonl \
  --limit 300 --model deepseek-v4-flash \
  --out data/trajectories/bird_scale1000c_flash_medium_success.jsonl \
  --workers 4 --max-steps 30 --max-errors-per-type 3 --attempts-per-example 2 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full \
  --resume > logs/sft_scale1000c/medium.log 2>&1 &
medium_pid=$!

.venv/bin/python -u src/sft/rollout_external_data.py \
  --examples-file data/eval_inputs/bird_train_sft1_scale1000c_buckets/hard.jsonl \
  --limit 300 --model deepseek-v4-flash \
  --out data/trajectories/bird_scale1000c_flash_hard_success.jsonl \
  --workers 4 --max-steps 40 --max-errors-per-type 4 --attempts-per-example 3 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full \
  --resume > logs/sft_scale1000c/hard.log 2>&1 &
hard_pid=$!

wait "$easy_pid" "$medium_pid" "$hard_pid"
