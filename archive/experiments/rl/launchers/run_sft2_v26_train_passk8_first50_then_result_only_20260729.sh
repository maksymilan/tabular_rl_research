#!/usr/bin/env bash
# Sample the frozen SFT2 policy on 50 BIRD-train tasks, select true mixed-outcome tasks, then train.
set -euo pipefail

RUNTIME=/home/dengyan/tabular_rl_outputs/rl_runtime_sft2_result_only_v26_20260729
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs
TASKS="$RUNTIME/data/rl/bird_sft2_step1682_sft1_passk4_mixed_first50_seed20260729_table_rl.json"
PASSK_RESULT="$RUNTIME/data/results/qwen25_coder7b_sft2_step1682_version26_bird_train_first50_passk8"
SELECTED="$RUNTIME/data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes.json"
TRAIN_OUTPUT="$OUTPUT_ROOT/checkpoints/qwen25-coder7b-sft2-step1682-result-only-sft2-passk8-mixed-first50-20260729"
ADAPTER="$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
SERVED_MODEL=qwen25-coder7b-sft2-step1682-version26-train-first50-passk8
STATUS="$OUTPUT_ROOT/logs/bird_sft2_v26_train_passk8_first50_then_result_only.status"
CHAIN_LOG="$OUTPUT_ROOT/logs/bird_sft2_v26_train_passk8_first50_then_result_only.chain.log"
VLLM_LOG="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.log"
VLLM_PID_FILE="$OUTPUT_ROOT/logs/$SERVED_MODEL.vllm.pid"
EVAL_LOG="$OUTPUT_ROOT/logs/$SERVED_MODEL.eval.log"
TRAIN_LOG="$OUTPUT_ROOT/logs/qwen25-coder7b-sft2-step1682-result-only-sft2-passk8-mixed-first50-20260729.log"
PORT=18053
EXPECTED=50

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

completed_rows() {
  if [[ -f "$PASSK_RESULT/all.jsonl" ]]; then
    wc -l <"$PASSK_RESULT/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

validate_passk() {
  /home/dengyan/miniconda3/envs/sft/bin/python - "$PASSK_RESULT/all.jsonl" "$EXPECTED" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
assert len(rows) == expected, len(rows)
indices = [int(row["example_index"]) for row in rows]
assert len(set(indices)) == expected, len(set(indices))
assert all((row.get("dataset_split") or "train") == "train" for row in rows)
infra = [
    row["example_index"]
    for row in rows
    if row.get("failure_type") in {
        "api_error",
        "transport_error",
        "provider_carrier_error",
    }
]
assert not infra, infra[:20]
PY
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$CHAIN_LOG" 2>&1
cd "$RUNTIME"

test -f "$TASKS"
test -f "$ADAPTER/adapter_model.safetensors"

if [[ "$(completed_rows)" -ne "$EXPECTED" ]]; then
  for attempt in 1 2 3; do
    set_status sampling "attempt=$attempt rows=$(completed_rows)/$EXPECTED gpu=1 k=8 concurrency=24"
    set +e
    PROJECT_DIR="$RUNTIME" \
    PYTHON_BIN=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python \
    BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
    ADAPTER="$ADAPTER" \
    SERVED_MODEL="$SERVED_MODEL" \
    GPU_ID=1 \
    PORT="$PORT" \
    RESULT_DIR="$PASSK_RESULT" \
    EXAMPLES_JSON="$TASKS" \
    N="$EXPECTED" \
    N_SAMPLES=8 \
    PASS_K=1,2,4,8 \
    WORKERS=3 \
    SAMPLE_WORKERS=8 \
    MAX_INFLIGHT=24 \
    MAX_NUM_SEQS=24 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_STEPS=30 \
    MAX_TOKENS=1024 \
    TEMPERATURE=0.7 \
    TOP_P=0.95 \
    HISTORY_TURNS=4 \
    EVAL_ENABLE_THINKING=0 \
    SERVER_CONFIG_ID=vllm-generation-config-vllm-sft2-step1682-version26-train-first50-passk8 \
    ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 \
    TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
    ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
    VLLM_LOG="$VLLM_LOG" \
    VLLM_PID_FILE="$VLLM_PID_FILE" \
    EVAL_LOG="$EVAL_LOG" \
      bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
    exit_code=$?
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
      break
    fi
    set_status sampling_retry "attempt=$attempt exit=$exit_code rows=$(completed_rows)/$EXPECTED"
    sleep 60
  done
fi

validate_passk
set_status selecting "rows=$EXPECTED mode=mixed_attempt_outcomes"
PYTHONPATH="$RUNTIME/src/rl:$RUNTIME/src/eval:$RUNTIME/src/harness:$RUNTIME/src/sft" \
  /home/dengyan/miniconda3/envs/sft/bin/python \
  src/rl/select_passk_rl_examples.py \
    --input "$PASSK_RESULT/all.jsonl" \
    --tasks "$TASKS" \
    --output "$SELECTED" \
    --target-pass-k 8 \
    --mode mixed_attempt_outcomes \
    --no-shuffle

selected_count="$(
  /home/dengyan/miniconda3/envs/sft/bin/python - "$SELECTED" <<'PY'
import json
import pathlib
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload["examples"]
assert payload["dataset_split"] == "train"
assert len(rows) == payload["count"]
assert len({int(row["example_index"]) for row in rows}) == len(rows)
assert all(row["selection"]["bucket"] == "mixed_attempt_outcomes" for row in rows)
assert all(pathlib.Path(row["db_path"]).is_file() for row in rows)
print(len(rows))
PY
)"
if [[ "$selected_count" -lt 1 ]]; then
  set_status failed "no mixed-outcome tasks among $EXPECTED sampled train tasks"
  exit 5
fi
if [[ -e "$TRAIN_OUTPUT" ]]; then
  set_status failed "training output already exists: $TRAIN_OUTPUT"
  exit 6
fi

set_status training "tasks=$selected_count steps=$selected_count gpu=1 group_size=4"
set +e
(
  cd "$RUNTIME"
  env \
    PROJECT_DIR="$RUNTIME" \
    CUDA_VISIBLE_DEVICES=1 \
    MODEL_PATH=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
    ADAPTER_PATH="$ADAPTER" \
    EXAMPLES_JSON="$SELECTED" \
    OUTPUT_DIR="$TRAIN_OUTPUT" \
    REWARD_MODE=result-only \
    KL_BETA=0 \
    STEPS="$selected_count" \
    GROUP_SIZE=4 \
    LEARNING_RATE=1e-6 \
    LR_SCHEDULER_TYPE=cosine \
    WARMUP_RATIO=0.03 \
    bash src/rl/frameworks/accelerate/run_atomic_group_reinforce.sh \
      --seed 20260729 \
      --save-every 10
) >"$TRAIN_LOG" 2>&1
train_exit=$?
set -e

if [[ "$train_exit" -eq 0 ]]; then
  set_status complete "sampled=$EXPECTED mixed=$selected_count trained_steps=$selected_count"
else
  set_status failed "training_exit=$train_exit mixed=$selected_count"
fi
exit "$train_exit"
