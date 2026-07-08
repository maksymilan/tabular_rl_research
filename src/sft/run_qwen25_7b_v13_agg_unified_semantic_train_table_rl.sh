#!/usr/bin/env bash
# Launch Qwen2.5-7B QLoRA training on table_rl for v13 aggregate-unified semantic-only data.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
LOG_DIR=${LOG_DIR:-/home/dengyan/tabular_rl_outputs/logs}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/qwen2.5_7b_qlora_v13_agg_unified_semantic_epoch4_table_rl.yaml}
GPU_ID=${GPU_ID:-0}

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

RUN_ID=${RUN_ID:-qwen25_7b_v13_agg_unified_semantic_906_epoch4_$(date +%Y%m%d_%H%M%S)}
LOG="$LOG_DIR/${RUN_ID}.log"
PID_FILE="$LOG_DIR/${RUN_ID}.pid"

cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"

(
  set +e
  /home/dengyan/miniconda3/envs/sft/bin/llamafactory-cli train "$CONFIG"
  status=$?

  /home/dengyan/miniconda3/envs/sft/bin/python - "$CONFIG" <<'PY'
import json
import os
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
output_dir = None
for raw in config_path.read_text().splitlines():
    line = raw.strip()
    if line.startswith("output_dir:"):
        output_dir = line.split(":", 1)[1].strip().strip('"').strip("'")
        break

if not output_dir:
    raise SystemExit("Could not find output_dir in config")

out = Path(output_dir)
inventory = []
for ckpt in sorted(out.glob("checkpoint-*")):
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        continue
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        continue
    epoch = state.get("epoch")
    global_step = state.get("global_step")
    inventory.append({
        "path": str(ckpt),
        "epoch": epoch,
        "global_step": global_step,
    })

(out / "checkpoint_inventory.json").write_text(
    json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
)

with_epoch = [item for item in inventory if isinstance(item.get("epoch"), (int, float))]
if with_epoch:
    epoch2 = min(with_epoch, key=lambda item: abs(float(item["epoch"]) - 2.0))
    epoch2_path = Path(epoch2["path"])
    (out / "epoch2_checkpoint.txt").write_text(str(epoch2_path) + "\n")
    stable = out.parent / "qwen2.5-7b-spider-v13-agg-unified-semantic-906-epoch2-qlora"
    if stable.is_symlink() or stable.exists():
        if stable.is_dir() and not stable.is_symlink():
            raise SystemExit(f"Stable epoch2 path exists and is not a symlink: {stable}")
        stable.unlink()
    os.symlink(epoch2_path, stable, target_is_directory=True)
    print(f"[checkpoint] epoch2={epoch2_path}")
    print(f"[checkpoint] epoch2_symlink={stable}")
else:
    print("[checkpoint] no checkpoint with epoch metadata found")
PY

  exit "$status"
) > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/qwen25_7b_v13_agg_unified_semantic_latest.run_id"

echo "run_id=$RUN_ID"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG"
echo "config=$CONFIG"
