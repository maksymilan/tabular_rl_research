#!/usr/bin/env bash
# Matched learning-rate comparison for the denotation-nonempty simple Process RL pilot.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/rl_runtime_sft2_result_only_v26_20260729
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs
TASKS="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"
ADAPTER="$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
REWARD_CONFIG="$RUNTIME/src/rl/configs/simple_process_reward.json"
RUN_STEM=qwen25-coder7b-sft2-step1682-simple-process-denotation-nonempty-first23-lr5e6-20260729
OUTPUT="$OUTPUT_ROOT/checkpoints/$RUN_STEM"
LOG="$OUTPUT_ROOT/logs/$RUN_STEM.log"
STATUS="$OUTPUT_ROOT/logs/$RUN_STEM.status"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

PYTHONPATH="$RUNTIME/src/rl:$RUNTIME/src/eval:$RUNTIME/src/harness:$RUNTIME/src/sft" \
  /home/dengyan/miniconda3/envs/sft/bin/python - "$TASKS" <<'PY'
import json
import pathlib
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload["examples"]
assert payload["dataset_split"] == "train"
assert len(rows) == payload["count"] == 23
assert len({int(row["example_index"]) for row in rows}) == 23
assert all(pathlib.Path(row["db_path"]).is_file() for row in rows)
assert all(row["selection"]["bucket"] == "mixed_attempt_outcomes" for row in rows)
PY

mkdir -p "$OUTPUT_ROOT/logs"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$REWARD_CONFIG"
if [[ -e "$OUTPUT" ]]; then
  set_status failed "isolated output already exists"
  exit 3
fi
if nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  set_status failed "GPU1 already has an active compute process"
  exit 4
fi

set_status training "tasks=23 steps=23 group_size=4 gpu=1 lr=5e-6"
set +e
(
  cd "$RUNTIME"
  env \
    PROJECT_DIR="$RUNTIME" \
    CUDA_VISIBLE_DEVICES=1 \
    MODEL_PATH=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
    ADAPTER_PATH="$ADAPTER" \
    EXAMPLES_JSON="$TASKS" \
    OUTPUT_DIR="$OUTPUT" \
    REWARD_MODE=process \
    PROCESS_REWARD_CONFIG="$REWARD_CONFIG" \
    PROCESS_ADMISSION_POLICY=denotation-nonempty \
    EXCLUDE_EMPTY_REFERENCE_RESULTS=1 \
    KL_BETA=0 \
    STEPS=23 \
    GROUP_SIZE=4 \
    LEARNING_RATE=5e-6 \
    LR_SCHEDULER_TYPE=cosine \
    WARMUP_RATIO=0.03 \
    bash src/rl/frameworks/accelerate/run_atomic_group_reinforce.sh \
      --seed 20260729 \
      --save-every 5
) >"$LOG" 2>&1
exit_code=$?
set -e

if [[ "$exit_code" -eq 0 ]]; then
  set_status complete "tasks=23 steps=23 group_size=4 lr=5e-6"
else
  set_status failed "exit=$exit_code"
fi
exit "$exit_code"
