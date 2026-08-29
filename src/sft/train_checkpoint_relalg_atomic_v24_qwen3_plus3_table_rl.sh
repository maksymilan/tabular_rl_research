#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/sft}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_8b_epoch1_plus3_table_rl_2gpu_qlora_8192.yaml}
DATASET=${DATASET:-$OUTPUT_ROOT/data/qwen3_8b_checkpoint_relalg_atomic_v24_sft2_20260821_r2/checkpoint_relalg_atomic_v24_union_qwen3_8192_training_view.jsonl}
SOURCE_ADAPTER=${SOURCE_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-from-base-8192-qlora}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-epoch1-plus3-8192-qlora}
RUN_ID=${RUN_ID:-qwen3_atomic_v24_epoch1_plus3_table_rl_2gpu_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
LOG=$LOG_DIR/$RUN_ID.log
STATUS=$LOG_DIR/$RUN_ID.status.json
MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for path in "$CONFIG" "$DATASET" "$SOURCE_ADAPTER/adapter_model.safetensors" "$SOURCE_ADAPTER/adapter_config.json"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 3; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 4; }
for path in "$LOG" "$STATUS" "$MANIFEST"; do
  [[ ! -e "$path" ]] || { echo "refusing existing artifact: $path" >&2; exit 4; }
done
for gpu in 0 1; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  (( used <= 512 )) || { echo "GPU $gpu is not idle: $used MiB" >&2; exit 5; }
done

"$ENV_DIR/bin/python" - "$CONFIG" "$DATASET" "$SOURCE_ADAPTER" "$OUTPUT_DIR" "$MANIFEST" <<'PY'
import hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

config, dataset, adapter, output, manifest = map(Path, sys.argv[1:6])
cfg = yaml.safe_load(config.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
expected_hashes = {
    dataset: "90ebbd965b3418f3701eac2f25231844183f73264e51514793ffd3f1df0d628f",
    adapter / "adapter_model.safetensors": "0cfadb83e10d2405697e2e7a45ecd2cb70dd9b709da67b349a1b441fbf7a3110",
    adapter / "adapter_config.json": "26efc2e8351db2ded019940ec433d74832dc23c5c9b4454d619cfea008c62db4",
}
for path, expected in expected_hashes.items():
    if sha(path) != expected:
        raise SystemExit(f"hash mismatch: {path}")
records = sum(bool(line.strip()) for line in dataset.open())
expected_cfg = {
    "adapter_name_or_path": str(adapter), "dataset": dataset.stem,
    "dataset_dir": str(dataset.parent), "output_dir": str(output),
    "gradient_accumulation_steps": 16, "per_device_train_batch_size": 1,
    "num_train_epochs": 3.0, "learning_rate": 2e-5, "cutoff_len": 8192,
}
for key, expected in expected_cfg.items():
    if cfg.get(key) != expected:
        raise SystemExit(f"config mismatch {key}: {cfg.get(key)!r} != {expected!r}")
steps_per_epoch = math.ceil(math.ceil(records / 2) / 16)
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": "table_rl", "physical_gpus": [0, 1], "world_size": 2,
    "experiment": "qwen3-8b-checkpoint-relalg-atomic-v24-epoch1-plus3",
    "initialization": "epoch-1 Atomic-v24 LoRA weights; new optimizer/scheduler/global step",
    "source_adapter": str(adapter), "source_adapter_sha256": expected_hashes[adapter / "adapter_model.safetensors"],
    "dataset": str(dataset), "dataset_sha256": expected_hashes[dataset], "records": records,
    "config": str(config), "config_sha256": sha(config),
    "additional_epochs": 3, "intended_cumulative_epochs": 4,
    "effective_global_batch": 32, "steps_per_epoch": steps_per_epoch,
    "expected_optimizer_steps": steps_per_epoch * 3, "output_dir": str(output),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
PY

ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
mkdir "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NPROC_PER_NODE=2 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$LOG" 2>&1
code=$?
set -e
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ $code -eq 0 ]]; then
  "$ENV_DIR/bin/python" - "$OUTPUT_DIR" <<'PY' || code=8
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
state = json.loads((root / "trainer_state.json").read_text())
if state.get("global_step") != 2130:
    raise SystemExit(f"global step mismatch: {state.get('global_step')}")
if not (root / "adapter_model.safetensors").is_file():
    raise SystemExit("missing final adapter")
PY
fi
"$ENV_DIR/bin/python" - "$STATUS" "$RUN_ID" "$started" "$finished" "$code" "$LOG" "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "run_id": sys.argv[2], "started_at_utc": sys.argv[3], "finished_at_utc": sys.argv[4],
    "exit_status": int(sys.argv[5]), "success": int(sys.argv[5]) == 0,
    "log": sys.argv[6], "output_dir": sys.argv[7], "expected_global_step": 2130,
}, indent=2) + "\n")
PY
exit "$code"
