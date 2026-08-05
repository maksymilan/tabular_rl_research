#!/usr/bin/env bash
# Generate the frozen 60-question x K4 SFT2 policy pool on GPU1.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_rank_score_v3_20260731}
POOL_DIR=${POOL_DIR:-$OUTPUT_ROOT/phase8_controlled_20260801/fixed_pool_60_seed101}
TASKS=${TASKS:-$POOL_DIR/tasks.jsonl}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/fixed_pool60_generation_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/fixed_pool60_generation_table_rl_20260801.log}
LOCK=${LOCK:-$OUTPUT_ROOT/logs/fixed_pool60_generation_table_rl_20260801.lock}
GPU_ID=${GPU_ID:-1}
TASK_BATCH_SIZE=${TASK_BATCH_SIZE:-4}
TASK_ID_FILE=${TASK_ID_FILE:-}
EXCLUDE_TASK_ID_FILE=${EXCLUDE_TASK_ID_FILE:-}
NO_FINALIZE=${NO_FINALIZE:-0}
EXPECTED_ADAPTER_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
EXPECTED_TASKS_SHA=4763da481f351e4a5e0d9d441e75b588ec547ffbb41d06876cc7baa4bfd5d85b

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$POOL_DIR/groups"
exec 8>"$LOCK"
if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1
cd "$RUNTIME"
[[ "$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | awk '{print $1}')" == "$EXPECTED_ADAPTER_SHA" ]]
[[ "$(sha256sum "$TASKS" | awk '{print $1}')" == "$EXPECTED_TASKS_SHA" ]]
[[ "$(grep -o 'PROTOCOL_VERSION = "version26"' src/sft/protocol.py | wc -l)" -eq 1 ]]

if [[ -f "$POOL_DIR/manifest.pending.json" ]]; then
  if "$PYTHON_BIN" - "$POOL_DIR/manifest.pending.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest["protocol_version"] == "version26"
assert manifest["tasks"] == 60
assert manifest["group_size"] == 4
assert manifest["trajectories"] == 240
assert manifest["temperature"] == 0.7
assert manifest["top_p"] == 0.95
assert manifest["max_steps"] == 30
PY
  then
    set_status complete "already generated tasks=60 trajectories=240"
    exit 0
  fi
fi

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
  if [[ -n "$used" && "$used" -le 512 ]]; then break; fi
  groups=$(find "$POOL_DIR/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
  set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown} groups=$groups/60"
  sleep 30
done

groups=$(find "$POOL_DIR/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
set_status generating "gpu=$GPU_ID groups=$groups/60 trajectories_target=240"
worker_args=()
if [[ -n "$TASK_ID_FILE" ]]; then worker_args+=(--task-id-file "$TASK_ID_FILE"); fi
if [[ -n "$EXCLUDE_TASK_ID_FILE" ]]; then worker_args+=(--exclude-task-id-file "$EXCLUDE_TASK_ID_FILE"); fi
if [[ "$NO_FINALIZE" -eq 1 ]]; then worker_args+=(--no-finalize); fi
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
  "$PYTHON_BIN" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
  --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" \
  --tasks "$TASKS" --output-dir "$POOL_DIR" --group-size 4 \
  --temperature 0.7 --top-p 0.95 --max-steps 30 --max-new-tokens 1024 \
  --max-context-tokens 8192 --history-turns 4 --seed 101 \
  --gpu-memory-utilization 0.82 --task-batch-size "$TASK_BATCH_SIZE" \
  "${worker_args[@]}"

if [[ "$NO_FINALIZE" -eq 1 ]]; then
  groups=$(find "$POOL_DIR/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
  set_status worker_complete "gpu=$GPU_ID groups=$groups/60 no_finalize=1"
  exit 0
fi

"$PYTHON_BIN" - "$POOL_DIR/manifest.pending.json" "$POOL_DIR/trajectories.jsonl" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
assert manifest["protocol_version"] == "version26"
assert manifest["tasks"] == 60 and manifest["trajectories"] == 240
assert len(rows) == 240
assert [row["sequence"] for row in rows] == list(range(240))
assert all(len(row["policy_turns"]) == len(row["sample"]["audit_record"]["turns"]) for row in rows)
PY
set_status complete "tasks=60 trajectories=240 pending_counterfactual_validation=1"
