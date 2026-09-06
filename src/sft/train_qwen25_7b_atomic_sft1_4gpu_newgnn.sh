#!/usr/bin/env bash
set -euo pipefail

# Four-RTX-3090 Qwen2.5-7B-Instruct comparison on the exact model-visible
# 4,471-record dataset used by the Qwen3-8B Atomic v26 SFT1 baseline.
# RUN_KIND=smoke runs two optimizer steps; RUN_KIND=full runs two epochs.

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/sft}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_atomic_v26_sft1_20260806}
QWEN25_AUDIT_DIR=${QWEN25_AUDIT_DIR:-$OUTPUT_ROOT/data/qwen25_7b_atomic_v26_sft1_20260904}
MAX_GPU_MEMORY_MIB=${MAX_GPU_MEMORY_MIB:-512}
RUN_KIND=${RUN_KIND:-smoke}

case "$RUN_KIND" in
  smoke)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen25_7b_atomic_v26_sft1_same4471_4gpu_smoke_qlora_6400.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen2.5-7b-bird-atomic-v26-sft1-same4471-6400-qlora-4gpu-smoke-20260904}
    EXPECTED_RECORDS=32
    EXPECTED_STEPS=2
    ;;
  full)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_external_teacher_qwen25_7b_atomic_v26_sft1_same4471_4gpu_qlora_6400.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen2.5-7b-bird-atomic-v26-sft1-same4471-6400-qlora-4gpu-20260904}
    EXPECTED_RECORDS=4471
    EXPECTED_STEPS=560
    ;;
  *)
    echo "RUN_KIND must be smoke or full" >&2
    exit 2
    ;;
esac

DATASET_MANIFEST=${DATASET_FILE%.jsonl}.manifest.json
DATASET_INFO=$DATASET_DIR/dataset_info.json
PREPARATION_MANIFEST=$DATASET_DIR/preparation_manifest.json
QWEN25_TOKEN_AUDIT=$QWEN25_AUDIT_DIR/bird_external_teacher_qwen25_7b_atomic_v26_sft1_6400_training_view.token_audit.json
QWEN25_AUDITED_DATA=$QWEN25_AUDIT_DIR/bird_external_teacher_qwen25_7b_atomic_v26_sft1_6400_training_view.jsonl
LOG_DIR=$OUTPUT_ROOT/logs

if [[ -z "${GPU_IDS:-}" ]]; then
  mapfile -t idle_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v limit="$MAX_GPU_MEMORY_MIB" '
          {gsub(/ /, "", $1); gsub(/ /, "", $2)}
          $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2 <= limit {print $1}
        ' \
      | head -n 4
  )
  if [[ ${#idle_gpus[@]} -ne 4 ]]; then
    echo "fewer than four GPUs are below the ${MAX_GPU_MEMORY_MIB} MiB idle threshold" >&2
    exit 6
  fi
  GPU_IDS=$(IFS=,; echo "${idle_gpus[*]}")
fi

IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
if [[ ${#physical_gpus[@]} -ne 4 ]]; then
  echo "GPU_IDS must name exactly four physical GPU indices" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${physical_gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || (( gpu < 0 || gpu > 7 )); then
    echo "invalid physical GPU index: $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "duplicate physical GPU index: $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

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
  "$PREPARATION_MANIFEST" \
  "$QWEN25_TOKEN_AUDIT" \
  "$QWEN25_AUDITED_DATA"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 3
  fi
done
for shard in "$MODEL_DIR"/model-0000{1,2,3,4}-of-00004.safetensors; do
  if [[ ! -f "$shard" ]]; then
    echo "missing model shard: $shard" >&2
    exit 3
  fi
done

"$ENV_DIR/bin/python" - <<'PY'
import bitsandbytes
import peft
import torch
import transformers
assert torch.cuda.is_available(), "CUDA is unavailable"
print(
    "dependency gate passed",
    torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__,
)
PY

SMOKE_STATUS_RECEIPT=""
if [[ "$RUN_KIND" == "full" ]]; then
  smoke_latest=$LOG_DIR/qwen25_7b_atomic_v26_sft1_same4471_4gpu_smoke_latest.run_id
  if [[ ! -f "$smoke_latest" ]]; then
    echo "missing successful four-GPU smoke run id: $smoke_latest" >&2
    exit 3
  fi
  smoke_run_id=$(tr -d '[:space:]' < "$smoke_latest")
  SMOKE_STATUS_RECEIPT=$LOG_DIR/$smoke_run_id.status.json
  if [[ ! -f "$SMOKE_STATUS_RECEIPT" ]]; then
    echo "missing four-GPU smoke status receipt: $SMOKE_STATUS_RECEIPT" >&2
    exit 3
  fi
  "$ENV_DIR/bin/python" - "$SMOKE_STATUS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("run_kind") != "smoke" or status.get("success") is not True:
    raise SystemExit("four-GPU smoke did not finish successfully")
if status.get("expected_global_step") != 2:
    raise SystemExit("four-GPU smoke step gate failed")
adapter = status.get("adapter_model")
if not adapter or not Path(adapter).is_file():
    raise SystemExit("four-GPU smoke adapter receipt is missing")
PY
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse existing output path: $OUTPUT_DIR" >&2
  exit 4
fi

gpu_memory_rows=()
gpu_uuid_rows=()
for gpu in "${physical_gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  if [[ ! "$used" =~ ^[0-9]+$ ]]; then
    echo "could not determine memory use for physical GPU $gpu" >&2
    exit 5
  fi
  if (( used > MAX_GPU_MEMORY_MIB )) || [[ -n "$pids" ]]; then
    echo "physical GPU $gpu is not idle: ${used} MiB used, pids=${pids:-none}" >&2
    exit 6
  fi
  uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  gpu_memory_rows+=("$gpu:$used")
  gpu_uuid_rows+=("$gpu:$uuid")
done

RUN_ID=${RUN_ID:-qwen25_7b_atomic_v26_sft1_same4471_4gpu_${RUN_KIND}_$(date +%Y%m%d_%H%M%S)}
LOG=$LOG_DIR/$RUN_ID.log
PID_FILE=$LOG_DIR/$RUN_ID.pid
LAUNCH_MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json
IMPLEMENTATION_LOCK=$LOG_DIR/$RUN_ID.implementation_lock.json
PRECISION_AUDIT=$LOG_DIR/$RUN_ID.precision_audit.json
STATUS_MANIFEST=$LOG_DIR/$RUN_ID.status.json

mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for artifact in \
  "$LOG" "$PID_FILE" "$LAUNCH_MANIFEST" "$IMPLEMENTATION_LOCK" \
  "$PRECISION_AUDIT" "$STATUS_MANIFEST"; do
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
  "$PREPARATION_MANIFEST" "$QWEN25_TOKEN_AUDIT" "$QWEN25_AUDITED_DATA" \
  "$MODEL_DIR" "$OUTPUT_DIR" "$GPU_IDS" "$RUN_KIND" "$EXPECTED_RECORDS" \
  "$EXPECTED_STEPS" "$LAUNCH_MANIFEST" "$IMPLEMENTATION_LOCK" \
  "$PRECISION_AUDIT" "${gpu_memory_rows[*]}" "${gpu_uuid_rows[*]}" \
  "$SMOKE_STATUS_RECEIPT" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

(
    config_path, dataset_path, dataset_manifest_path, dataset_info_path,
    preparation_manifest_path, qwen25_audit_path, qwen25_audited_data_path,
    model_path, output_path, gpu_ids, run_kind, expected_records,
    expected_steps, launch_manifest_path, implementation_lock_path,
    precision_audit_path, initial_gpu_memory, physical_gpu_uuids, smoke_status_path,
) = sys.argv[1:]

config = Path(config_path)
dataset = Path(dataset_path)
dataset_manifest = Path(dataset_manifest_path)
dataset_info = Path(dataset_info_path)
preparation_manifest = Path(preparation_manifest_path)
qwen25_audit = Path(qwen25_audit_path)
qwen25_audited_data = Path(qwen25_audited_data_path)
model = Path(model_path)
output = Path(output_path)
expected_records = int(expected_records)
expected_steps = int(expected_steps)
launch_manifest = Path(launch_manifest_path)
implementation_lock = Path(implementation_lock_path)
precision_audit = Path(precision_audit_path)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
expected_config = {
    "model_name_or_path": str(model),
    "dataset_dir": str(dataset.parent),
    "output_dir": str(output),
    "template": "qwen",
    "cutoff_len": 6400,
    "mask_history": True,
    "quantization_bit": 4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "gradient_checkpointing": True,
}
for key, value in expected_config.items():
    if cfg.get(key) != value:
        raise SystemExit(f"launch/config mismatch for {key}: {cfg.get(key)!r} != {value!r}")
if cfg.get("enable_thinking") not in (None, False):
    raise SystemExit("Qwen2.5 control must not enable the Qwen3 thinking template")
if run_kind == "full":
    if cfg.get("dataset") != "bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view":
        raise SystemExit("full dataset-name gate failed")
    if float(cfg.get("num_train_epochs")) != 2.0 or cfg.get("max_steps") is not None:
        raise SystemExit("full schedule gate failed")
elif run_kind == "smoke":
    if cfg.get("dataset") != "bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke":
        raise SystemExit("smoke dataset-name gate failed")
    if int(cfg.get("max_steps", -1)) != 2:
        raise SystemExit("smoke schedule gate failed")
else:
    raise SystemExit(f"unknown run kind: {run_kind}")

dataset_hash = sha256(dataset)
manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
if manifest.get("output_sha256") != dataset_hash:
    raise SystemExit("dataset manifest SHA-256 gate failed")
records = manifest.get("records", manifest.get("repeats"))
if records != expected_records:
    raise SystemExit(f"dataset record-count gate failed: {records} != {expected_records}")
registry = json.loads(dataset_info.read_text(encoding="utf-8"))
entry = registry.get(cfg["dataset"])
if not isinstance(entry, dict) or entry.get("file_name") != dataset.name:
    raise SystemExit("dataset_info registry gate failed")

preparation = json.loads(preparation_manifest.read_text(encoding="utf-8"))
if preparation.get("prompt_sha256") != "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316":
    raise SystemExit("student prompt identity gate failed")
if preparation.get("records") != 4471 or preparation.get("complete_episodes") != 678:
    raise SystemExit("preparation cohort identity gate failed")
full_dataset = dataset.parent / "bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view.jsonl"
if sha256(full_dataset) != "c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14":
    raise SystemExit("pinned 4,471-record training-view SHA-256 gate failed")

audit = json.loads(qwen25_audit.read_text(encoding="utf-8"))
if audit.get("template") != "qwen" or audit.get("cutoff_len") != 6400:
    raise SystemExit("Qwen2.5 tokenizer audit configuration gate failed")
if audit.get("records") != 4471 or audit.get("kept_by_filter_policy") != 4471:
    raise SystemExit("Qwen2.5 tokenizer audit record gate failed")
if audit.get("dropped_by_filter_policy") != 0 or audit.get("dropped_truncated_final_target") != 0:
    raise SystemExit("Qwen2.5 tokenizer audit truncation gate failed")
if sha256(qwen25_audited_data) != sha256(full_dataset):
    raise SystemExit("Qwen2.5 audited data is not byte-identical to the Qwen3 training view")

model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
if model_config.get("architectures") != ["Qwen2ForCausalLM"]:
    raise SystemExit("model architecture gate failed")
if model_config.get("model_type") != "qwen2" or model_config.get("num_hidden_layers") != 28:
    raise SystemExit("model config identity gate failed")
expected_shard_sha256 = {
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
}
actual_shard_sha256 = {name: sha256(model / name) for name in expected_shard_sha256}
if actual_shard_sha256 != expected_shard_sha256:
    raise SystemExit("pinned model shard SHA-256 gate failed")

launcher = config.parent.parent / "train_qwen25_7b_atomic_sft1_4gpu_newgnn.sh"
package_versions = {}
for package in (
    "torch", "transformers", "peft", "bitsandbytes", "datasets", "llamafactory", "accelerate"
):
    try:
        module = __import__(package)
        package_versions[package] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        package_versions[package] = f"import-error: {exc}"

created_at = datetime.now(timezone.utc).isoformat()
common = {
    "created_at_utc": created_at,
    "experiment": "qwen2.5-7b-atomic-v26-sft1-same4471-4gpu-control",
    "comparison_boundary": (
        "Qwen2.5-7B-Instruct model-family control on the Qwen3 SFT1 model-visible data; "
        "not the current Qwen3-8B mainline model identity"
    ),
    "protocol": "version26",
    "protocol_hash": "4da19387399bd3a5",
    "carrier": "think-json-v1",
    "student_prompt_sha256": "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
    "model_repo": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "model": str(model),
    "model_config_sha256": sha256(model / "config.json"),
    "model_index_sha256": sha256(model / "model.safetensors.index.json"),
    "tokenizer_config_sha256": sha256(model / "tokenizer_config.json"),
    "model_shards_sha256": actual_shard_sha256,
    "dataset": str(dataset),
    "dataset_sha256": dataset_hash,
    "full_training_view_sha256": sha256(full_dataset),
    "qwen25_token_audit": str(qwen25_audit),
    "qwen25_token_audit_sha256": sha256(qwen25_audit),
    "records": records,
    "physical_gpus": gpu_ids.split(","),
    "physical_gpu_uuids": physical_gpu_uuids.split(),
    "initial_gpu_memory_mib": initial_gpu_memory.split(),
    "world_size": 4,
    "effective_global_batch": 16,
    "expected_optimizer_steps": expected_steps,
    "config": str(config),
    "config_sha256": sha256(config),
    "launcher": str(launcher),
    "launcher_sha256": sha256(launcher),
    "dataset_manifest_sha256": sha256(dataset_manifest),
    "dataset_info_sha256": sha256(dataset_info),
    "preparation_manifest_sha256": sha256(preparation_manifest),
    "output_dir": str(output),
    "python": sys.executable,
    "python_version": platform.python_version(),
    "packages": package_versions,
    "git_head": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.parents[3], text=True,
        capture_output=True, check=False,
    ).stdout.strip(),
}
if run_kind == "full":
    smoke_status = Path(smoke_status_path)
    common["four_gpu_smoke_status"] = str(smoke_status)
    common["four_gpu_smoke_status_sha256"] = sha256(smoke_status)
    common["distributed_sampler"] = {
        "source_records": 4471,
        "samples_per_rank_per_epoch": 1118,
        "padded_duplicate_samples_per_epoch": 1,
        "optimizer_steps_per_epoch": 280,
        "last_optimizer_step_effective_global_examples": 8,
    }
launch_manifest.write_text(
    json.dumps({**common, "run_kind": run_kind}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
implementation_lock.write_text(
    json.dumps(
        {
            "created_at_utc": created_at,
            "config_sha256": common["config_sha256"],
            "launcher_sha256": common["launcher_sha256"],
            "dataset_sha256": common["dataset_sha256"],
            "model_revision": common["model_revision"],
            "model_shards_sha256": actual_shard_sha256,
            "protocol_hash": common["protocol_hash"],
            "student_prompt_sha256": common["student_prompt_sha256"],
            "world_size": 4,
            "effective_global_batch": 16,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
precision_audit.write_text(
    json.dumps(
        {
            "created_at_utc": created_at,
            "base_model_storage": "bitsandbytes-nf4-4bit",
            "quantization_bit": 4,
            "compute_dtype": "bfloat16",
            "bf16": True,
            "upcast_layernorm": True,
            "gradient_checkpointing": True,
            "optimizer": "paged_adamw_8bit",
            "torch": package_versions["torch"],
            "bitsandbytes": package_versions["bitsandbytes"],
            "model_declared_dtype": model_config.get("torch_dtype"),
            "cuda_kernel_gate": "torch 2.6.0+cu124 BF16 matmul synchronized on RTX 3090 before launch",
            "four_gpu_longest_record_smoke_status": smoke_status_path or None,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
for path in (launch_manifest, implementation_lock, precision_audit):
    path.chmod(0o444)
PY

# Model hashing above is intentionally exhaustive and can take about a minute.
# Re-check the selected devices immediately before starting the GPU process.
for gpu in "${physical_gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > MAX_GPU_MEMORY_MIB )) || [[ -n "$pids" ]]; then
    echo "physical GPU $gpu was claimed during preflight: ${used:-unknown} MiB, pids=${pids:-none}" >&2
    exit 6
  fi
done

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"

echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/qwen25_7b_atomic_v26_sft1_same4471_4gpu_${RUN_KIND}_latest.run_id"
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
path.chmod(0o444)
PY

exit "$status"
