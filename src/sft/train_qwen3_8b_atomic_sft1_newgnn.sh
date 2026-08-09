#!/usr/bin/env bash
set -euo pipefail

# Reproducible two-RTX-3090 launcher for the frozen atomic version26 Qwen3-8B
# SFT1 baseline.  RUN_KIND=smoke uses 32 copies of the longest token-audited
# record; RUN_KIND=full trains the complete 4,471-target dataset for two epochs.

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/qwen3-atomic-sft}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_atomic_v26_sft1_20260806}
GPU_IDS=${GPU_IDS:-5,6}
MAX_GPU_MEMORY_MIB=${MAX_GPU_MEMORY_MIB:-512}
RUN_KIND=${RUN_KIND:-smoke}

case "$RUN_KIND" in
  smoke)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen3_8b_sft1_smoke_qlora_6400.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora-smoke}
    EXPECTED_STEPS=2
    ;;
  full)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen3_8b_sft1_qlora_6400.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora}
    EXPECTED_STEPS=560
    ;;
  *)
    echo "RUN_KIND must be smoke or full" >&2
    exit 2
    ;;
esac

RUN_ID=${RUN_ID:-qwen3_8b_atomic_v26_sft1_${RUN_KIND}_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
LOG=$LOG_DIR/$RUN_ID.log
PID_FILE=$LOG_DIR/$RUN_ID.pid
LAUNCH_MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json
STATUS_MANIFEST=$LOG_DIR/$RUN_ID.status.json
DATASET_MANIFEST=${DATASET_FILE%.jsonl}.manifest.json
DATASET_INFO=$DATASET_DIR/dataset_info.json
PREPARATION_MANIFEST=$DATASET_DIR/preparation_manifest.json

IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
if [[ ${#physical_gpus[@]} -ne 2 ]] \
  || [[ ! "${physical_gpus[0]}" =~ ^[0-9]+$ ]] \
  || [[ ! "${physical_gpus[1]}" =~ ^[0-9]+$ ]] \
  || [[ "${physical_gpus[0]}" == "${physical_gpus[1]}" ]]; then
  echo "GPU_IDS must name exactly two distinct physical GPU indices, for example 5,6" >&2
  exit 2
fi

for required in \
  "$CONFIG" \
  "$ENV_DIR/bin/llamafactory-cli" \
  "$ENV_DIR/bin/python" \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/model.safetensors.index.json" \
  "$MODEL_DIR/tokenizer_config.json" \
  "$DATASET_FILE" \
  "$DATASET_MANIFEST" \
  "$DATASET_INFO" \
  "$PREPARATION_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 3
  fi
done
for shard in "$MODEL_DIR"/model-0000{1,2,3,4,5}-of-00005.safetensors; do
  if [[ ! -f "$shard" ]]; then
    echo "missing model shard: $shard" >&2
    exit 3
  fi
done

"$ENV_DIR/bin/python" - <<'PY'
import bitsandbytes
import liger_kernel
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("dependency gate passed", torch.__version__, bitsandbytes.__version__)
PY

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse existing output path: $OUTPUT_DIR" >&2
  exit 4
fi

gpu_memory_rows=()
for gpu in "${physical_gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ ! "$used" =~ ^[0-9]+$ ]]; then
    echo "could not determine memory use for physical GPU $gpu" >&2
    exit 5
  fi
  if (( used > MAX_GPU_MEMORY_MIB )); then
    echo "physical GPU $gpu is not idle: ${used} MiB used (limit ${MAX_GPU_MEMORY_MIB})" >&2
    exit 6
  fi
  gpu_memory_rows+=("$gpu:$used")
done

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for artifact in "$LOG" "$PID_FILE" "$LAUNCH_MANIFEST" "$STATUS_MANIFEST"; do
  if [[ -e "$artifact" ]]; then
    echo "refusing to overwrite existing run artifact: $artifact" >&2
    exit 4
  fi
done
if ! mkdir "$OUTPUT_DIR"; then
  echo "could not acquire exclusive output directory: $OUTPUT_DIR" >&2
  exit 4
fi
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

"$ENV_DIR/bin/python" - \
  "$CONFIG" "$DATASET_FILE" "$DATASET_MANIFEST" "$DATASET_INFO" \
  "$PREPARATION_MANIFEST" "$MODEL_DIR" "$OUTPUT_DIR" "$GPU_IDS" \
  "$RUN_KIND" "$LAUNCH_MANIFEST" <<'PY'
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
dataset_manifest_path = Path(sys.argv[3])
dataset_info = Path(sys.argv[4])
preparation_manifest = Path(sys.argv[5])
model = Path(sys.argv[6])
output = Path(sys.argv[7])
gpu_ids = sys.argv[8]
run_kind = sys.argv[9]
manifest = Path(sys.argv[10])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(model),
    "dataset": dataset.stem,
    "dataset_dir": str(dataset.parent),
    "output_dir": str(output),
    "template": "qwen3",
    "enable_thinking": True,
    "preserve_thinking": False,
    "mask_history": True,
    "cutoff_len": 6400,
    "quantization_bit": 4,
    "enable_liger_kernel": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
}
for key, value in expected.items():
    if cfg.get(key) != value:
        raise SystemExit(f"launch/config mismatch for {key}: {cfg.get(key)!r} != {value!r}")
dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
records = dataset_manifest.get("records", dataset_manifest.get("repeats"))
dataset_hash = sha256(dataset)
if Path(dataset_manifest.get("output", "")).resolve() != dataset.resolve():
    raise SystemExit("dataset manifest output-path gate failed")
if dataset_manifest.get("output_sha256") != dataset_hash:
    raise SystemExit("dataset manifest sha256 gate failed")
with dataset.open(encoding="utf-8") as handle:
    actual_records = sum(bool(line.strip()) for line in handle)
if actual_records != records:
    raise SystemExit(f"dataset line-count gate failed: {actual_records} != {records}")
dataset_registry = json.loads(dataset_info.read_text(encoding="utf-8"))
dataset_entry = dataset_registry.get(cfg["dataset"])
if not isinstance(dataset_entry, dict) or dataset_entry.get("file_name") != dataset.name:
    raise SystemExit("dataset_info registry gate failed")
preparation = json.loads(preparation_manifest.read_text(encoding="utf-8"))
if (preparation.get("files") or {}).get(str(dataset)) != dataset_hash:
    raise SystemExit("preparation-manifest dataset binding gate failed")
if run_kind == "full":
    if records != 4471 or float(cfg.get("num_train_epochs")) != 2.0:
        raise SystemExit("full-run record/epoch gate failed")
    expected_optimizer_steps = 560
elif run_kind == "smoke":
    if records != 32 or int(cfg.get("max_steps")) != 2:
        raise SystemExit("smoke record/step gate failed")
    expected_optimizer_steps = 2
else:
    raise SystemExit(f"unknown run kind: {run_kind}")

model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
if model_config.get("model_type") != "qwen3" or model_config.get("num_hidden_layers") != 36:
    raise SystemExit("model architecture gate failed")
expected_shard_sha256 = {
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
}
actual_shard_sha256 = {
    name: sha256(model / name) for name in sorted(expected_shard_sha256)
}
if actual_shard_sha256 != expected_shard_sha256:
    raise SystemExit("pinned model shard sha256 gate failed")

payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_kind": run_kind,
    "experiment": "qwen3-8b-historical-atomic-version26-sft1-qlora",
    "protocol_boundary": (
        "frozen version26 single-action think-json-v1; not current version54 native-tool-bundle"
    ),
    "model_repo": "Qwen/Qwen3-8B",
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "model": str(model),
    "model_config_sha256": sha256(model / "config.json"),
    "model_index_sha256": sha256(model / "model.safetensors.index.json"),
    "tokenizer_config_sha256": sha256(model / "tokenizer_config.json"),
    "model_shards_sha256": actual_shard_sha256,
    "config": str(config),
    "config_sha256": sha256(config),
    "dataset": str(dataset),
    "dataset_sha256": dataset_hash,
    "dataset_manifest": str(dataset_manifest_path),
    "dataset_manifest_sha256": sha256(dataset_manifest_path),
    "dataset_info": str(dataset_info),
    "dataset_info_sha256": sha256(dataset_info),
    "preparation_manifest": str(preparation_manifest),
    "preparation_manifest_sha256": sha256(preparation_manifest),
    "records": records,
    "physical_gpus": gpu_ids.split(","),
    "world_size": 2,
    "effective_global_batch": 16,
    "expected_optimizer_steps": expected_optimizer_steps,
    "launcher_sha256": sha256(config.parent.parent / "train_qwen3_8b_atomic_sft1_newgnn.sh"),
    "output_dir": str(output),
    "python": sys.executable,
    "python_version": platform.python_version(),
}
if run_kind == "full":
    payload["distributed_sampler"] = {
        "source_records": 4471,
        "samples_per_rank_per_epoch": 2236,
        "padded_duplicate_samples_per_epoch": 1,
        "optimizer_steps_per_epoch": 280,
        "last_optimizer_step_effective_global_examples": 8,
    }
for package in (
    "torch", "transformers", "peft", "bitsandbytes", "datasets", "llamafactory", "liger_kernel"
):
    try:
        module = __import__(package)
        payload[package] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[package] = f"import-error: {exc}"
payload["git_head"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=config.parents[3], text=True,
    capture_output=True, check=False,
).stdout.strip()
manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"

echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/qwen3_8b_atomic_v26_sft1_${RUN_KIND}_latest.run_id"
echo "run_id=$RUN_ID"
echo "pid=$$"
echo "physical_gpus=$GPU_IDS"
echo "initial_gpu_memory=${gpu_memory_rows[*]}"
echo "log=$LOG"
echo "config=$CONFIG"
echo "launch_manifest=$LAUNCH_MANIFEST"

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$LOG" 2>&1
status=$?
set -e
finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if find "$OUTPUT_DIR" -name trainer_state.json -print -quit | grep -q .; then
  "$ENV_DIR/bin/python" "$PROJECT_DIR/src/sft/summarize_sft_loss.py" \
    --output-dir "$OUTPUT_DIR" >> "$LOG" 2>&1 || true
fi

adapter_path=""
if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
  adapter_path=$OUTPUT_DIR/adapter_model.safetensors
else
  adapter_path=$(find "$OUTPUT_DIR" -type f -name adapter_model.safetensors -print | sort | tail -1)
fi
if [[ $status -eq 0 ]] && [[ -z "$adapter_path" ]]; then
  echo "training exited successfully but no adapter_model.safetensors was saved" >> "$LOG"
  status=7
fi
if [[ $status -eq 0 ]]; then
  if ! "$ENV_DIR/bin/python" - "$OUTPUT_DIR" "$EXPECTED_STEPS" >> "$LOG" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
states = []
for path in root.rglob("trainer_state.json"):
    state = json.loads(path.read_text(encoding="utf-8"))
    states.append((int(state.get("global_step", -1)), str(path)))
if not states:
    raise SystemExit("no trainer_state.json was saved")
actual = max(step for step, _ in states)
if actual != expected:
    raise SystemExit(f"final global_step mismatch: {actual} != {expected}; states={states}")
print(f"final global_step gate passed: {actual}")
PY
  then
    status=8
  fi
fi

"$ENV_DIR/bin/python" - \
  "$STATUS_MANIFEST" "$RUN_ID" "$RUN_KIND" "$started_at" "$finished_at" \
  "$status" "$LOG" "$OUTPUT_DIR" "$adapter_path" "$EXPECTED_STEPS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "run_kind": sys.argv[3],
    "started_at_utc": sys.argv[4],
    "finished_at_utc": sys.argv[5],
    "exit_status": int(sys.argv[6]),
    "success": int(sys.argv[6]) == 0,
    "log": sys.argv[7],
    "output_dir": sys.argv[8],
    "adapter_model": sys.argv[9] or None,
    "expected_global_step": int(sys.argv[10]),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "status=$status"
echo "status_manifest=$STATUS_MANIFEST"
exit "$status"
