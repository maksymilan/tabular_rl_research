#!/usr/bin/env bash
# RFT (on-policy self-training) stage launcher on the 8B Atomic v26 mainline.
#
# Continues the frozen cumulative SFT adapter ``checkpoint-6380`` on verified episodes that the
# same checkpoint sampled itself.  ``SMOKE_STEPS>0`` derives a bounded smoke configuration and
# writes it into the run root; a smoke run is explicitly marked and must never be reported as a
# result.  The formal run is selected by leaving ``SMOKE_STEPS=0``.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR}
RUN_ROOT=${RUN_ROOT:?set a fresh RUN_ROOT}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/bird_rft_qwen3_8b_atomic_v26_c_tier_12288_qlora.yaml}
DATASET_DIR=${DATASET_DIR:-/home/dengyan/tabular_rl_outputs/data/qwen3_v26_rft_20260917}
ADAPTER=${ADAPTER:-/home/dengyan/tabular_rl_outputs/checkpoints/checkpoint-6380}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/sft}
GPU_IDS=${GPU_IDS:-0,1}
SMOKE_STEPS=${SMOKE_STEPS:-0}

if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT already exists, refusing to reuse: $RUN_ROOT" >&2
  exit 2
fi
[[ -f "$CONFIG" ]] || { echo "ERROR: missing config $CONFIG" >&2; exit 2; }
[[ -f "$DATASET_DIR/dataset_info.json" ]] || { echo "ERROR: dataset_info.json not registered" >&2; exit 2; }
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "ERROR: adapter incomplete: $ADAPTER" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
LOG=$RUN_ROOT/logs/train.log
exec >>"$LOG" 2>&1
log() { echo "[$(date -Is)] $*"; }

IFS=',' read -r GPU0 GPU1 <<<"$GPU_IDS"
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  used0=$(nvidia-smi --id="$GPU0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  used1=$(nvidia-smi --id="$GPU1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used0 <= 512 && used1 <= 512 )); then break; fi
  sleep 30
done
(( used0 <= 512 && used1 <= 512 )) || { log "ERROR: GPUs not idle (${used0}/${used1} MiB)"; exit 75; }

EFFECTIVE_CONFIG=$RUN_ROOT/derived_config.yaml
if (( SMOKE_STEPS > 0 )); then
  "$ENV_DIR/bin/python" - "$CONFIG" "$EFFECTIVE_CONFIG" "$RUN_ROOT/smoke" "$SMOKE_STEPS" <<'PY'
import sys
from pathlib import Path
import yaml

source, target, output_dir, steps = sys.argv[1:5]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["output_dir"] = str(Path(output_dir) / "train")
config["max_steps"] = int(steps)
config["num_train_epochs"] = 1.0
config["save_strategy"] = "no"
config["logging_steps"] = 1
Path(target).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
else
  cp "$CONFIG" "$EFFECTIVE_CONFIG"
fi

"$ENV_DIR/bin/python" - "$EFFECTIVE_CONFIG" "$DATASET_DIR" "$ADAPTER" "$RUN_ROOT/preflight.json" "$SMOKE_STEPS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

config_path, dataset_dir, adapter, out, smoke_steps = sys.argv[1:6]
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
dataset_dir = Path(dataset_dir)
registry = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
entry = registry[config["dataset"]]
data_path = dataset_dir / entry["file_name"]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = {
    "schema_version": "qwen3-atomic-v26-rft-preflight-v1",
    "stage": "smoke" if int(smoke_steps) > 0 else "formal",
    "smoke_steps": int(smoke_steps) or None,
    "config": str(config_path),
    "config_sha256": sha(config_path),
    "dataset": config["dataset"],
    "dataset_file": str(data_path),
    "dataset_sha256": sha(data_path),
    "dataset_dir_sha256": sha(dataset_dir / "dataset_info.json"),
    "adapter": adapter,
    "adapter_sha256": sha(Path(adapter) / "adapter_model.safetensors"),
    "base_model": config["model_name_or_path"],
    "learning_rate": config["learning_rate"],
    "num_train_epochs": config["num_train_epochs"],
    "cutoff_len": config["cutoff_len"],
    "lora_rank": config["lora_rank"],
    "quantization_bit": config["quantization_bit"],
    "mask_history": config["mask_history"],
    "gradient_checkpointing": config["gradient_checkpointing"],
}
Path(out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"

log "rft $( (( SMOKE_STEPS > 0 )) && echo smoke || echo formal ) start config=$EFFECTIVE_CONFIG gpus=$GPU_IDS"
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$EFFECTIVE_CONFIG"
rc=$?
set -e
log "rft finished rc=$rc"
exit $rc
