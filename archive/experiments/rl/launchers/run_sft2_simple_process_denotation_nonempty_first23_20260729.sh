#!/usr/bin/env bash
# One-step smoke followed by one deterministic pass over the first 23 retained mixed-outcome tasks.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/rl_runtime_sft2_result_only_v26_20260729
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs
TASKS="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"
ADAPTER="$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
REWARD_CONFIG="$RUNTIME/src/rl/configs/simple_process_reward.json"
RUN_STEM=qwen25-coder7b-sft2-step1682-simple-process-denotation-nonempty-first23-20260729
SMOKE_OUTPUT="$OUTPUT_ROOT/checkpoints/${RUN_STEM}-smoke1"
FORMAL_OUTPUT="$OUTPUT_ROOT/checkpoints/$RUN_STEM"
SMOKE_LOG="$OUTPUT_ROOT/logs/${RUN_STEM}-smoke1.log"
FORMAL_LOG="$OUTPUT_ROOT/logs/${RUN_STEM}.log"
STATUS="$OUTPUT_ROOT/logs/${RUN_STEM}.status"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

validate_task_pool() {
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
}

run_training() {
  local output=$1
  local steps=$2
  local save_every=$3
  local log=$4
  (
    cd "$RUNTIME"
    env \
      PROJECT_DIR="$RUNTIME" \
      CUDA_VISIBLE_DEVICES=0 \
      MODEL_PATH=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
      ADAPTER_PATH="$ADAPTER" \
      EXAMPLES_JSON="$TASKS" \
      OUTPUT_DIR="$output" \
      REWARD_MODE=process \
      PROCESS_REWARD_CONFIG="$REWARD_CONFIG" \
      PROCESS_ADMISSION_POLICY=denotation-nonempty \
      EXCLUDE_EMPTY_REFERENCE_RESULTS=1 \
      KL_BETA=0 \
      STEPS="$steps" \
      GROUP_SIZE=4 \
      LEARNING_RATE=1e-6 \
      LR_SCHEDULER_TYPE=cosine \
      WARMUP_RATIO=0.03 \
      bash src/rl/frameworks/accelerate/run_atomic_group_reinforce.sh \
        --seed 20260729 \
        --save-every "$save_every"
  ) >"$log" 2>&1
}

validate_smoke() {
  PYTHONPATH="$RUNTIME/src/rl:$RUNTIME/src/eval:$RUNTIME/src/harness:$RUNTIME/src/sft" \
    /home/dengyan/miniconda3/envs/sft/bin/python - "$SMOKE_OUTPUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metrics = [json.loads(line) for line in (root / "metrics.jsonl").open() if line.strip()]
rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").open() if line.strip()]
assert len(metrics) == 1, len(metrics)
assert len(rollouts) == 4, len(rollouts)
event = metrics[0]
assert event["step"] == 1
assert event["reward_mode"] == "process"
assert event["optimization_error"] is None
assert event["denotation_comparison"] == "bird-set"
assert event["updated"], event
assert any(sample.get("process_admission_policy") == "denotation-nonempty" for sample in rollouts)
reference_filter = json.load((root / "reference_result_filter.json").open())
assert reference_filter["retained_count"] == 23
assert reference_filter["excluded_count"] == 0
assert (root / "checkpoint-1" / "adapter_model.safetensors").is_file()
assert (root / "checkpoint-1" / "trainer_state.pt").is_file()
PY
}

mkdir -p "$OUTPUT_ROOT/logs"
validate_task_pool
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$REWARD_CONFIG"

if [[ -e "$SMOKE_OUTPUT" || -e "$FORMAL_OUTPUT" ]]; then
  set_status failed "isolated output already exists"
  exit 3
fi
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  set_status failed "GPU0 already has an active compute process"
  exit 4
fi

set_status smoke "tasks=23 steps=1 group_size=4 gpu=0"
run_training "$SMOKE_OUTPUT" 1 1 "$SMOKE_LOG"
validate_smoke

set_status training "tasks=23 steps=23 group_size=4 gpu=0 lr=1e-6"
run_training "$FORMAL_OUTPUT" 23 5 "$FORMAL_LOG"

set_status complete "tasks=23 steps=23 group_size=4"
