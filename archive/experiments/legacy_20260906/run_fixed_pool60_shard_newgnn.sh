#!/usr/bin/env bash
# Generate one disjoint fixed-pool task shard on an explicitly free NewGNN GPU.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_phase8_parallel_20260801}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${ADAPTER_PATH:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
TASKS=${TASKS:-$O/phase8_parallel_20260801/tasks.jsonl}
for name in GPU_ID TASK_ID_FILE OUTPUT_DIR STATUS RUN_LOG; do
  [[ -n "${!name:-}" ]] || { printf 'missing %s\n' "$name" >&2; exit 2; }
done
if [[ "$GPU_ID" != 6 && "$GPU_ID" != 7 ]]; then
  printf 'NewGNN phase8 worker is restricted to GPU6/7, got %s\n' "$GPU_ID" >&2
  exit 2
fi
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT
mkdir -p "$O/logs" "$OUTPUT_DIR/groups"
exec >>"$RUN_LOG" 2>&1

used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
compute_pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
if [[ "$used" -gt 512 || -n "$compute_pids" ]]; then
  set_status blocked "gpu=$GPU_ID mib=$used compute_pids=${compute_pids:-none}"
  exit 3
fi
[[ "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" == d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e ]]
[[ "$(grep -c '^PROTOCOL_VERSION = "version26"' "$R/src/sft/protocol.py")" -eq 1 ]]

set_status generating "gpu=$GPU_ID tasks=$(wc -l < "$TASK_ID_FILE") task_batch_size=2"
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
  "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
  --model-path "$MODEL" --adapter-path "$SFT2" --tasks "$TASKS" \
  --output-dir "$OUTPUT_DIR" --group-size 4 --temperature 0.7 --top-p 0.95 \
  --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
  --history-turns 4 --seed 101 --gpu-memory-utilization 0.82 \
  --task-batch-size 2 --task-id-file "$TASK_ID_FILE" --no-finalize

"$PY" - "$TASKS" "$TASK_ID_FILE" "$OUTPUT_DIR/groups" <<'PY'
import json,sys
from pathlib import Path
tasks_path, ids_path, groups_path = map(Path, sys.argv[1:])
order = {
    row["example_id"]: index
    for index, row in enumerate(json.loads(line) for line in tasks_path.open() if line.strip())
}
ids = [line.strip() for line in ids_path.open() if line.strip()]
for task_id in ids:
    rows = json.loads((groups_path / f"{task_id}.json").read_text())
    assert len(rows) == 4
    assert [int(row["sequence"]) for row in rows] == [order[task_id] * 4 + i for i in range(4)]
    assert [int(row["sample"]["audit_record"]["sample_index"]) for row in rows] == list(range(4))
print(json.dumps({"tasks": len(ids), "trajectories": len(ids) * 4}))
PY
set_status complete "gpu=$GPU_ID shard_tasks=$(wc -l < "$TASK_ID_FILE") trajectories=$((4 * $(wc -l < "$TASK_ID_FILE")))"
