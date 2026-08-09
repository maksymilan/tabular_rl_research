#!/usr/bin/env bash
set -euo pipefail

# Two-RTX-3090 Qwen3-4B QLoRA scale control. This deliberately uses the
# exact atomic-version26 SFT1 targets and hyperparameters of the completed
# Qwen3-8B control; it is not the paper's unavailable full-parameter SFT job.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
ENV_DIR="${ENV_DIR:-/home/dengyan/miniconda3/envs/qwen3-atomic-sft}"
MODEL_DIR="${MODEL_DIR:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}"
REFERENCE_TOKENIZER_DIR="${REFERENCE_TOKENIZER_DIR:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
DATASET_DIR="${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_atomic_v26_sft1_20260806}"
GPU_IDS="${GPU_IDS:-5,6}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-512}"
RUN_KIND="${RUN_KIND:-smoke}"

case "$RUN_KIND" in
  smoke)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/bird_external_teacher_qwen3_4b_sft1_smoke_qlora_6400.yaml}"
    DATASET_FILE="${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-4b-bird-atomic-v26-sft1-6400-qlora-smoke}"
    EXPECTED_RECORDS=32
    EXPECTED_STEPS=2
    ;;
  full)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/bird_external_teacher_qwen3_4b_sft1_qlora_6400.yaml}"
    DATASET_FILE="${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-4b-bird-atomic-v26-sft1-6400-qlora}"
    EXPECTED_RECORDS=4471
    EXPECTED_STEPS=560
    ;;
  *)
    echo "RUN_KIND must be smoke or full" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-qwen3_4b_atomic_v26_sft1_${RUN_KIND}_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
PID_FILE="$LOG_DIR/$RUN_ID.pid"
LAUNCH_MANIFEST="$LOG_DIR/$RUN_ID.launch_manifest.json"
STATUS_MANIFEST="$LOG_DIR/$RUN_ID.status.json"
DATASET_MANIFEST="${DATASET_FILE%.jsonl}.manifest.json"
DATASET_INFO="$DATASET_DIR/dataset_info.json"
PREPARATION_MANIFEST="$DATASET_DIR/preparation_manifest.json"
MODEL_SPECS="$SCRIPT_DIR/qwen3_model_specs.json"
MODEL_VERIFIER="$SCRIPT_DIR/verify_pinned_qwen3_model.py"

IFS=',' read -r -a physical_gpus <<<"$GPU_IDS"
if [[ ${#physical_gpus[@]} -ne 2 ]] \
  || [[ ! "${physical_gpus[0]}" =~ ^[0-9]+$ ]] \
  || [[ ! "${physical_gpus[1]}" =~ ^[0-9]+$ ]] \
  || [[ "${physical_gpus[0]}" == "${physical_gpus[1]}" ]]; then
  echo "GPU_IDS must name exactly two distinct physical GPUs" >&2
  exit 2
fi

for required in \
  "$CONFIG" "$ENV_DIR/bin/llamafactory-cli" "$ENV_DIR/bin/python" \
  "$MODEL_VERIFIER" "$MODEL_SPECS" "$DATASET_FILE" "$DATASET_MANIFEST" \
  "$DATASET_INFO" "$PREPARATION_MANIFEST"; do
  [[ -f "$required" ]] || {
    echo "missing required file: $required" >&2
    exit 3
  }
done
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "refusing to reuse output directory: $OUTPUT_DIR" >&2
  exit 4
}

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for artifact in "$LOG" "$PID_FILE" "$LAUNCH_MANIFEST" "$STATUS_MANIFEST"; do
  [[ ! -e "$artifact" ]] || {
    echo "refusing to overwrite run artifact: $artifact" >&2
    exit 4
  }
done

for gpu in "${physical_gpus[@]}"; do
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$used" =~ ^[0-9]+$ ]] || exit 5
  (( used <= MAX_GPU_MEMORY_MIB )) || {
    echo "physical GPU $gpu is not idle: ${used} MiB" >&2
    exit 6
  }
done

ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
MODEL_GATE="$LOG_DIR/$RUN_ID.model_gate.json"
"$ENV_DIR/bin/python" "$MODEL_VERIFIER" \
  --model-root "$MODEL_DIR" --model-size 4b --specs "$MODEL_SPECS" >"$MODEL_GATE"

"$ENV_DIR/bin/python" - \
  "$CONFIG" "$DATASET_FILE" "$DATASET_MANIFEST" "$DATASET_INFO" \
  "$PREPARATION_MANIFEST" "$MODEL_DIR" "$REFERENCE_TOKENIZER_DIR" \
  "$OUTPUT_DIR" "$EXPECTED_RECORDS" "$EXPECTED_STEPS" "$RUN_KIND" \
  "$MODEL_GATE" "$LAUNCH_MANIFEST" "$GPU_IDS" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

(config, dataset, dataset_manifest, dataset_info, preparation_manifest,
 model, reference_tokenizer, output, expected_records, expected_steps,
 run_kind, model_gate, launch_manifest, gpu_ids) = sys.argv[1:]
config = Path(config); dataset = Path(dataset); dataset_manifest = Path(dataset_manifest)
dataset_info = Path(dataset_info); preparation_manifest = Path(preparation_manifest)
model = Path(model); reference_tokenizer = Path(reference_tokenizer); output = Path(output)
model_gate = Path(model_gate); launch_manifest = Path(launch_manifest)
expected_records = int(expected_records); expected_steps = int(expected_steps)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
expected_cfg = {
    "model_name_or_path": str(model), "dataset_dir": str(dataset.parent),
    "output_dir": str(output), "template": "qwen3", "enable_thinking": True,
    "preserve_thinking": False, "mask_history": True, "cutoff_len": 6400,
    "quantization_bit": 4, "enable_liger_kernel": True,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4, "lora_rank": 16, "lora_alpha": 32,
    "lora_dropout": 0.05,
}
for key, value in expected_cfg.items():
    if cfg.get(key) != value:
        raise SystemExit(f"config drift for {key}: {cfg.get(key)!r} != {value!r}")
if run_kind == "full" and float(cfg.get("num_train_epochs")) != 2.0:
    raise SystemExit("full run must use exactly two epochs")
if run_kind == "smoke" and int(cfg.get("max_steps")) != 2:
    raise SystemExit("smoke run must use exactly two optimizer steps")

manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
records = manifest.get("records", manifest.get("repeats"))
dataset_hash = sha256(dataset)
if records != expected_records or manifest.get("output_sha256") != dataset_hash:
    raise SystemExit("dataset manifest record/hash gate failed")
if sum(bool(line.strip()) for line in dataset.open(encoding="utf-8")) != records:
    raise SystemExit("dataset line-count gate failed")
registry = json.loads(dataset_info.read_text(encoding="utf-8"))
entry = registry.get(cfg["dataset"])
if not isinstance(entry, dict) or entry.get("file_name") != dataset.name:
    raise SystemExit("dataset registry gate failed")
preparation = json.loads(preparation_manifest.read_text(encoding="utf-8"))
if (preparation.get("files") or {}).get(str(dataset)) != dataset_hash:
    raise SystemExit("preparation manifest does not bind this dataset")

tokenizer_files = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
tokenizer_hashes = {}
for name in tokenizer_files:
    current = sha256(model / name)
    reference = sha256(reference_tokenizer / name)
    if current != reference:
        raise SystemExit(f"4B/8B tokenizer mismatch: {name}")
    tokenizer_hashes[name] = current

versions = {}
for package in (
    "torch", "transformers", "peft", "bitsandbytes", "datasets",
    "accelerate", "llamafactory", "liger_kernel",
):
    try:
        versions[package] = importlib.metadata.version(package.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "not-installed"

payload = {
    "schema_version": "qwen3-4b-atomic-v26-sft1-launch-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "qwen3-4b-historical-atomic-version26-sft1-qlora-scale-control",
    "paper_training_match": False,
    "protocol_boundary": "frozen atomic version26; not current version54 native-tool-bundle",
    "model_repository": "Qwen/Qwen3-4B",
    "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "model_root": str(model),
    "model_gate": json.loads(model_gate.read_text(encoding="utf-8")),
    "shared_tokenizer_hashes": tokenizer_hashes,
    "config": str(config), "config_sha256": sha256(config),
    "dataset": str(dataset), "dataset_sha256": dataset_hash, "records": records,
    "dataset_manifest_sha256": sha256(dataset_manifest),
    "dataset_info_sha256": sha256(dataset_info),
    "preparation_manifest_sha256": sha256(preparation_manifest),
    "run_kind": run_kind, "physical_gpus": gpu_ids.split(","), "world_size": 2,
    "effective_global_batch": 16, "expected_optimizer_steps": expected_steps,
    "output_dir": str(output), "python": sys.executable,
    "python_version": platform.python_version(), "package_versions": versions,
}
launch_manifest.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mkdir "$OUTPUT_DIR"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"

printf '%s\n' "$$" >"$PID_FILE"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" >"$LOG" 2>&1
status=$?
set -e
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $status -eq 0 ]]; then
  "$ENV_DIR/bin/python" - "$OUTPUT_DIR" "$EXPECTED_STEPS" <<'PY' || status=8
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); expected = int(sys.argv[2])
states = [json.loads(path.read_text()) for path in root.rglob("trainer_state.json")]
actual = max((int(state.get("global_step", -1)) for state in states), default=-1)
if actual != expected:
    raise SystemExit(f"global_step mismatch: {actual} != {expected}")
if not list(root.rglob("adapter_model.safetensors")):
    raise SystemExit("adapter weights were not saved")
PY
fi

"$ENV_DIR/bin/python" - \
  "$STATUS_MANIFEST" "$RUN_ID" "$RUN_KIND" "$started_at" "$finished_at" \
  "$status" "$LOG" "$OUTPUT_DIR" "$EXPECTED_STEPS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2], "run_kind": sys.argv[3],
    "started_at_utc": sys.argv[4], "finished_at_utc": sys.argv[5],
    "exit_status": int(sys.argv[6]), "success": int(sys.argv[6]) == 0,
    "log": sys.argv[7], "output_dir": sys.argv[8],
    "expected_global_step": int(sys.argv[9]),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
exit "$status"
