#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_omnisql_7b_sft1_union_4epoch_qlora_6400.yaml}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/omnisql-sft}
GPU_ID=${GPU_ID:-5}
MAX_GPU_MEMORY_MIB=${MAX_GPU_MEMORY_MIB:-512}
RUN_ID=${RUN_ID:-bird_omnisql_sft1_union_4epoch_$(date +%Y%m%d_%H%M%S)}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/OmniSQL-7B-af4eed67}
DATASET_DIR=${DATASET_DIR:-/home/dengyan/tabular_rl_outputs/data/omnisql_sft1_union_sft1_sft2_20260801}
OUTPUT_DIR=${OUTPUT_DIR:-/home/dengyan/tabular_rl_outputs/checkpoints/omnisql-7b-bird-sft1-union-sft1-sft2-6400-4epoch-qlora}
DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/bird_omnisql_sft1_union_sft1_sft2_6400_training_view.jsonl}
DATASET_MANIFEST=${DATASET_MANIFEST:-${DATASET_FILE%.jsonl}.manifest.json}
DATASET_INFO=${DATASET_INFO:-$DATASET_DIR/dataset_info.json}
LOG_DIR="$OUTPUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
PID_FILE="$LOG_DIR/$RUN_ID.pid"
MANIFEST="$LOG_DIR/$RUN_ID.launch_manifest.json"

for required in \
  "$CONFIG" \
  "$ENV_DIR/bin/llamafactory-cli" \
  "$ENV_DIR/bin/python" \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/model.safetensors" \
  "$DATASET_FILE" \
  "$DATASET_MANIFEST" \
  "$DATASET_INFO"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty output directory: $OUTPUT_DIR" >&2
  exit 3
fi

gpu_memory_mib=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ ]]; then
  echo "could not determine memory use for physical GPU $GPU_ID" >&2
  exit 4
fi
if (( gpu_memory_mib > MAX_GPU_MEMORY_MIB )); then
  echo "physical GPU $GPU_ID is not idle: ${gpu_memory_mib} MiB used (limit ${MAX_GPU_MEMORY_MIB})" >&2
  exit 5
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

"$ENV_DIR/bin/python" - "$CONFIG" "$DATASET_FILE" "$MODEL_DIR" "$OUTPUT_DIR" "$GPU_ID" "$MANIFEST" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

config = Path(sys.argv[1])
dataset = Path(sys.argv[2])
model = Path(sys.argv[3])
output = Path(sys.argv[4])
gpu = sys.argv[5]
manifest = Path(sys.argv[6])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

training_config = yaml.safe_load(config.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(model),
    "dataset": dataset.stem,
    "dataset_dir": str(dataset.parent),
    "output_dir": str(output),
}
for key, value in expected.items():
    if training_config.get(key) != value:
        raise SystemExit(
            f"launch/config mismatch for {key}: {training_config.get(key)!r} != {value!r}"
        )

dataset_manifest = dataset.with_suffix(".manifest.json")
dataset_info = dataset.parent / "dataset_info.json"
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "config": str(config),
    "config_sha256": sha256(config),
    "dataset": str(dataset),
    "dataset_sha256": sha256(dataset),
    "dataset_manifest": str(dataset_manifest),
    "dataset_manifest_sha256": sha256(dataset_manifest),
    "dataset_info": str(dataset_info),
    "dataset_info_sha256": sha256(dataset_info),
    "model": str(model),
    "model_config_sha256": sha256(model / "config.json"),
    "output_dir": str(output),
    "physical_gpu": gpu,
    "python": sys.executable,
    "python_version": platform.python_version(),
}
for package in ("torch", "transformers", "peft", "datasets", "llamafactory"):
    try:
        module = __import__(package)
        payload[package] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[package] = f"import-error: {exc}"
payload["git_head"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=config.parents[3], text=True,
    capture_output=True, check=False,
).stdout.strip()
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"

echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/bird_omnisql_sft1_union_latest.run_id"
echo "run_id=$RUN_ID"
echo "pid=$$"
echo "physical_gpu=$GPU_ID"
echo "initial_gpu_memory_mib=$gpu_memory_mib"
echo "log=$LOG"
echo "config=$CONFIG"
echo "launch_manifest=$MANIFEST"

set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$LOG" 2>&1
status=$?
set -e

if find "$OUTPUT_DIR" -name trainer_state.json -print -quit | grep -q .; then
  "$ENV_DIR/bin/python" "$PROJECT_DIR/src/sft/summarize_sft_loss.py" \
    --output-dir "$OUTPUT_DIR" >> "$LOG" 2>&1 || true
fi
exit "$status"
