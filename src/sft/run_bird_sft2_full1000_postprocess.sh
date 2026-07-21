#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
LOG=${LOG:-/tmp/bird_sft2_full1000_postprocess.log}
cd "$PROJECT_DIR"
exec >>"$LOG" 2>&1

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] starting SFT2 full1000 replay and quality gates"
/usr/bin/nice -n 10 .venv/bin/python -u src/sft/prepare_bird_sft2_onpolicy.py \
  --tasks data/eval_inputs/bird_train_sft2_student_pilot1000.jsonl \
  --passk-all data/results/bird_sft2_student_full1000_k4_canonical_all.jsonl \
  --raw-success-out data/trajectories/bird_sft2_student_full1000_success_raw.jsonl \
  --accepted-success-out data/trajectories/bird_sft2_student_full1000_success_accepted.jsonl \
  --rejected-out data/trajectories/bird_sft2_student_full1000_success_rejected.jsonl \
  --fallback-tasks-out data/eval_inputs/bird_sft2_flash_fallback_full1000_all368.jsonl \
  --fallback-index-out data/eval_inputs/bird_sft2_flash_fallback_full1000_all368.index.jsonl \
  --manifest data/trajectories/bird_sft2_student_full1000_prepare.manifest.json \
  --fallback-per-difficulty 10000 --seed 20260720 --max-steps 20 --max-think-words 300

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] building rolling SFT records"
/usr/bin/nice -n 10 .venv/bin/python -u src/sft/build_rolling_sft_data.py \
  --input data/trajectories/bird_sft2_student_full1000_success_accepted.jsonl \
  --out data/sft/bird_sft2_student_full1000_success_rolling4_full_resident.jsonl \
  --index-out data/sft/bird_sft2_student_full1000_success_rolling4_full_resident.index.jsonl \
  --dataset-name bird_sft2_student_full1000_success_rolling4_full_resident \
  --history-turns 4 --rolling-prompt-variant full --rolling-observation-style resident
echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] SFT2 full1000 postprocess complete"
