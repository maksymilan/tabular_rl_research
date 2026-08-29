#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/qwen3-atomic-sft}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_8b_epoch1_plus3_qlora_8192.yaml}
DATASET=${DATASET:-$OUTPUT_ROOT/data/qwen3_8b_checkpoint_relalg_atomic_v24_sft2_20260821_r2/checkpoint_relalg_atomic_v24_union_qwen3_8192_training_view.jsonl}
SOURCE_ADAPTER=${SOURCE_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-from-base-8192-qlora}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-epoch1-plus3-8192-qlora}
RUN_ID=${RUN_ID:-qwen3_checkpoint_relalg_atomic_v24_epoch1_plus3_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
WAIT_LOG=$LOG_DIR/${RUN_ID}.wait.log
TRAIN_LOG=$LOG_DIR/${RUN_ID}.log
STATUS=$LOG_DIR/${RUN_ID}.status.json
LAUNCH_MANIFEST=$LOG_DIR/${RUN_ID}.launch_manifest.json

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for path in "$CONFIG" "$DATASET" "$SOURCE_ADAPTER/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 3; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 4; }
for path in "$WAIT_LOG" "$TRAIN_LOG" "$STATUS" "$LAUNCH_MANIFEST"; do
  [[ ! -e "$path" ]] || { echo "refusing existing artifact: $path" >&2; exit 4; }
done

python3 - "$CONFIG" "$DATASET" "$SOURCE_ADAPTER" "$OUTPUT_DIR" "$LAUNCH_MANIFEST" <<'PY'
import hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

config, dataset, adapter, output, launch = map(Path, sys.argv[1:6])
cfg = yaml.safe_load(config.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
records = sum(bool(line.strip()) for line in dataset.open())
expected = {
    "adapter_name_or_path": str(adapter), "output_dir": str(output),
    "dataset_dir": str(dataset.parent), "dataset": dataset.stem,
    "num_train_epochs": 3.0, "learning_rate": 2e-5,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "cutoff_len": 8192, "mask_history": True,
}
for key, value in expected.items():
    if cfg.get(key) != value:
        raise SystemExit(f"config mismatch {key}: {cfg.get(key)!r} != {value!r}")
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "qwen3-8b-checkpoint-relalg-atomic-v24-epoch1-plus3",
    "initialization": "load epoch-1 Atomic-v24 LoRA weights; new optimizer/scheduler/global step",
    "source_adapter": str(adapter),
    "source_adapter_sha256": sha(adapter / "adapter_model.safetensors"),
    "dataset": str(dataset), "dataset_sha256": sha(dataset), "records": records,
    "config": str(config), "config_sha256": sha(config),
    "additional_epochs": 3, "intended_cumulative_epochs": 4,
    "world_size": 4, "effective_global_batch": 32,
    "expected_optimizer_steps": math.ceil(math.ceil(records / 4) / 8) * 3,
    "output_dir": str(output), "gpu_selection": "first four GPUs with <=512 MiB for two polls",
}
launch.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "$(date -Is) waiting for four idle GPUs" >> "$WAIT_LOG"
selected=""
while [[ -z "$selected" ]]; do
  mapfile -t first < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2+0 <= 512 {gsub(/ /,"",$1); print $1}')
  if (( ${#first[@]} >= 4 )); then
    sleep 30
    mapfile -t second < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2+0 <= 512 {gsub(/ /,"",$1); print $1}')
    candidates=()
    for gpu in "${first[@]}"; do
      for current in "${second[@]}"; do
        [[ "$gpu" == "$current" ]] && candidates+=("$gpu")
      done
    done
    if (( ${#candidates[@]} >= 4 )); then
      selected=$(IFS=,; echo "${candidates[*]:0:4}")
      break
    fi
  fi
  echo "$(date -Is) idle_count=${#first[@]}" >> "$WAIT_LOG"
  sleep 60
done

echo "$(date -Is) selected_gpus=$selected" >> "$WAIT_LOG"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
mkdir "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$selected" FORCE_TORCHRUN=1 NPROC_PER_NODE=4 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$TRAIN_LOG" 2>&1
code=$?
set -e
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$STATUS" "$RUN_ID" "$started" "$finished" "$code" "$selected" "$TRAIN_LOG" "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "run_id": sys.argv[2], "started_at_utc": sys.argv[3], "finished_at_utc": sys.argv[4],
    "exit_status": int(sys.argv[5]), "success": int(sys.argv[5]) == 0,
    "physical_gpus": sys.argv[6].split(","), "log": sys.argv[7], "output_dir": sys.argv[8],
}, indent=2) + "\n")
PY
exit "$code"
