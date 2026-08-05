#!/usr/bin/env bash
# Generate the deterministic 4-per-difficulty admission reserve on table_rl GPU1.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
C=${CANDIDATE_DIR:-$O/phase8_controlled_20260801/fixed_pool_60_seed101_admission_reserve_v1}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER=${ADAPTER_PATH:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
STATUS=${STATUS:-$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.log}
LOCK=${LOCK:-$O/logs/fixed_pool_admission_reserve_generation_table_rl_20260801.lock}
GPU_ID=${GPU_ID:-1}
EXPECTED_ADAPTER_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
EXPECTED_TASKS_SHA=88261b88c159f4597c92203fb484b21d0ed3d6421da47d3831e28a4a0cb0f351

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$O/logs" "$C/groups"
exec 8>"$LOCK"
flock -n 8 || exit 0
exec >>"$RUN_LOG" 2>&1
cd "$R"

[[ "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" == "$EXPECTED_ADAPTER_SHA" ]]
[[ "$(sha256sum "$C/tasks.jsonl" | awk '{print $1}')" == "$EXPECTED_TASKS_SHA" ]]
[[ "$(grep -o 'PROTOCOL_VERSION = "version26"' src/sft/protocol.py | wc -l)" -eq 1 ]]

if [[ -f "$C/manifest.pending.json" ]]; then
  "$PYTHON_BIN" - "$C/manifest.pending.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["protocol_version"] == "version26"
assert p["tasks"] == 12 and p["trajectories"] == 48 and p["group_size"] == 4
assert p["temperature"] == 0.7 and p["top_p"] == 0.95 and p["max_steps"] == 30
PY
  set_status complete "already generated tasks=12 trajectories=48"
  exit 0
fi

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
  if [[ -n "$used" && "$used" -le 512 ]]; then break; fi
  groups=$(find "$C/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
  set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown} groups=$groups/12"
  sleep 30
done

groups=$(find "$C/groups" -maxdepth 1 -name 'bird_train_*.json' 2>/dev/null | wc -l)
set_status generating "gpu=$GPU_ID groups=$groups/12 trajectories_target=48"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
  "$PYTHON_BIN" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
  --model-path "$MODEL" --adapter-path "$ADAPTER" \
  --tasks "$C/tasks.jsonl" --output-dir "$C" --group-size 4 \
  --temperature 0.7 --top-p 0.95 --max-steps 30 --max-new-tokens 1024 \
  --max-context-tokens 8192 --history-turns 4 --seed 101 \
  --gpu-memory-utilization 0.82 --task-batch-size 4

set_status complete "tasks=12 trajectories=48 pending_admission_audit=1"
