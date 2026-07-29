#!/usr/bin/env bash
# Wait for the isolated one-group smoke, validate it, then launch the frozen 100-task result-only run.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/rl_runtime_sft2_result_only_v26_20260729
SMOKE_OUTPUT=/home/dengyan/tabular_rl_outputs/checkpoints/qwen25-coder7b-sft2-step1682-result-only-sft1-passk4-mixed100-smoke1-retry1-20260729
SMOKE_STATUS=/home/dengyan/tabular_rl_outputs/logs/qwen25-coder7b-sft2-step1682-result-only-sft1-passk4-mixed100-smoke1-retry1-20260729.status
OUTPUT=/home/dengyan/tabular_rl_outputs/checkpoints/qwen25-coder7b-sft2-step1682-result-only-sft1-passk4-mixed100-20260729
LOG=/home/dengyan/tabular_rl_outputs/logs/qwen25-coder7b-sft2-step1682-result-only-sft1-passk4-mixed100-20260729.log
STATUS=/home/dengyan/tabular_rl_outputs/logs/qwen25-coder7b-sft2-step1682-result-only-sft1-passk4-mixed100-20260729.status
RUN_MANIFEST="$RUNTIME/data/rl/bird_sft2_step1682_result_only_sft1_passk4_mixed100_20260729.run_manifest.json"

while [[ ! -s "$SMOKE_STATUS" ]]; do
  sleep 30
done

grep -qx 'exit_code=0' <(head -n 1 "$SMOKE_STATUS")

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
assert event["reward_mode"] == "result-only"
assert len(event["rewards"]) == 4
assert set(event["rewards"]).issubset({0, 1, 0.0, 1.0})
assert event["optimization_error"] is None
assert event["denotation_comparison"] == "bird-set"
assert event["context_mode"] == "rolling-legal-history"
assert event["history_turns"] == 4
assert (root / "checkpoint-1" / "adapter_model.safetensors").is_file()
assert (root / "checkpoint-1" / "trainer_state.pt").is_file()
PY

if [[ -e "$OUTPUT" ]]; then
  printf 'refusing to reuse formal output: %s\n' "$OUTPUT" >&2
  exit 3
fi

if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing to launch while GPU0 still has an active compute process\n' >&2
  exit 4
fi

mkdir -p "$OUTPUT"
cp "$RUN_MANIFEST" "$OUTPUT/run_manifest.json"
printf 'started_at=%s\n' "$(date -Is)" > "$OUTPUT/run.started"

set +e
(
  cd "$RUNTIME"
  env \
    PROJECT_DIR="$RUNTIME" \
    CUDA_VISIBLE_DEVICES=0 \
    MODEL_PATH=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
    ADAPTER_PATH=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682 \
    EXAMPLES_JSON="$RUNTIME/data/rl/bird_sft2_step1682_result_only_sft1_passk4_mixed100_seed20260729_table_rl.json" \
    OUTPUT_DIR="$OUTPUT" \
    REWARD_MODE=result-only \
    KL_BETA=0 \
    STEPS=100 \
    GROUP_SIZE=4 \
    LEARNING_RATE=1e-6 \
    LR_SCHEDULER_TYPE=cosine \
    WARMUP_RATIO=0.03 \
    bash src/rl/frameworks/accelerate/run_atomic_group_reinforce.sh \
      --seed 20260729 \
      --save-every 20
) >"$LOG" 2>&1
exit_code=$?
set -e

printf 'exit_code=%s\nfinished_at=%s\n' "$exit_code" "$(date -Is)" > "$STATUS"
exit "$exit_code"
