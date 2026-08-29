#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/sft}
CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_projection_v2_epoch1_plus3_table_rl_2gpu_qlora_8192.yaml}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_checkpoint_relalg_atomic_v24_sft2_20260825_projection_v2}
DATASET=$DATASET_DIR/checkpoint_relalg_atomic_v24_union_qwen3_8192_training_view_v2.jsonl
CARRIER_AUDIT=$DATASET_DIR/qwen3_carrier_audit.json
SOURCE_ADAPTER=${SOURCE_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-projection-v2-from-base-8192-qlora}
SOURCE_STATUS=${SOURCE_STATUS:-$OUTPUT_ROOT/logs/qwen3_atomic_v24_projection_v2_full_table_rl_20260825_1043.status.json}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-projection-v2-epoch1-plus3-8192-qlora}
RUN_ID=${RUN_ID:-qwen3_atomic_v24_projection_v2_epoch1_plus3_table_rl_20260825}
GPU_IDS=${GPU_IDS:-0,1}
LOG_DIR=$OUTPUT_ROOT/logs
WAIT_LOG=$LOG_DIR/$RUN_ID.wait.log
TRAIN_LOG=$LOG_DIR/$RUN_ID.log
STATUS=$LOG_DIR/$RUN_ID.status.json
MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json

mkdir -p "$LOG_DIR"
for path in "$WAIT_LOG" "$TRAIN_LOG" "$STATUS" "$MANIFEST"; do
  [[ ! -e "$path" ]] || { echo "refusing existing artifact: $path" >&2; exit 4; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 4; }

echo "$(date -Is) waiting for clean projection-v2 epoch-1 status" >> "$WAIT_LOG"
while [[ ! -e "$SOURCE_STATUS" ]]; do sleep 60; done
"$ENV_DIR/bin/python" - "$SOURCE_STATUS" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if not d.get('success') or d.get('expected_global_step') != 709:
    raise SystemExit('epoch-1 did not complete successfully; refusing plus3')
PY

for path in "$CONFIG" "$DATASET" "$CARRIER_AUDIT" "$SOURCE_ADAPTER/adapter_model.safetensors" \
  "$SOURCE_ADAPTER/adapter_config.json"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 3; }
done
while true; do
  ready=1
  IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
  [[ ${#physical_gpus[@]} -eq 2 ]] || exit 5
  for gpu in "${physical_gpus[@]}"; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    (( used <= 512 )) || ready=0
  done
  (( ready == 1 )) && break
  sleep 30
done

"$ENV_DIR/bin/python" - "$CONFIG" "$DATASET" "$CARRIER_AUDIT" "$SOURCE_ADAPTER" "$OUTPUT_DIR" "$MANIFEST" <<'PY'
import hashlib,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
config,dataset,audit,adapter,output,manifest=map(Path,sys.argv[1:7])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
dataset_sha='ea74fdb6f4cc3f446f86ea6934c7615d6df6b7f86450c738d0356dbf778db09c'
prompt_sha='92d4e226d0f5bfa82687c2034f04b1343e28601b27f41c853688110a66b2caea'
if sha(dataset)!=dataset_sha: raise SystemExit('dataset hash mismatch')
a=json.loads(audit.read_text())
if not a.get('passed') or a.get('training_view_sha256')!=dataset_sha or a.get('qwen3_student_prompt_sha256')!=prompt_sha:
 raise SystemExit('carrier audit mismatch')
cfg=yaml.safe_load(config.read_text())
expected={'adapter_name_or_path':str(adapter),'dataset':dataset.stem,'dataset_dir':str(dataset.parent),
 'output_dir':str(output),'num_train_epochs':3.0,'learning_rate':2e-5,'gradient_accumulation_steps':16,
 'per_device_train_batch_size':1,'template':'qwen3','enable_thinking':True,'preserve_thinking':False,
 'mask_history':True,'cutoff_len':8192}
for k,v in expected.items():
 if cfg.get(k)!=v: raise SystemExit(f'config mismatch {k}: {cfg.get(k)!r}')
payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'host':'table_rl','world_size':2,
 'experiment':'qwen3-8b-checkpoint-relalg-atomic-v24-projection-v2-epoch1-plus3',
 'initialization':'clean projection-v2 epoch-1 adapter; new optimizer/scheduler/global step',
 'source_adapter':str(adapter),'source_adapter_sha256':sha(adapter/'adapter_model.safetensors'),
 'dataset':str(dataset),'dataset_sha256':dataset_sha,'records':22661,'qwen3_prompt_sha256':prompt_sha,
 'config':str(config),'config_sha256':sha(config),'additional_epochs':3,'intended_cumulative_epochs':4,
 'effective_global_batch':32,'expected_optimizer_steps':2127,'output_dir':str(output)}
manifest.write_text(json.dumps(payload,indent=2)+'\n')
PY

mkdir -p /home/dengyan/cuda_link
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
mkdir "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$TRAIN_LOG" 2>&1
code=$?
set -e
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ $code -eq 0 ]]; then
 "$ENV_DIR/bin/python" - "$OUTPUT_DIR" <<'PY' || code=8
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); d=json.loads((r/'trainer_state.json').read_text())
if d.get('global_step')!=2127: raise SystemExit('global step mismatch')
if not (r/'adapter_model.safetensors').is_file(): raise SystemExit('missing final adapter')
PY
fi
"$ENV_DIR/bin/python" - "$STATUS" "$RUN_ID" "$started" "$finished" "$code" "$TRAIN_LOG" "$OUTPUT_DIR" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'run_id':sys.argv[2],'started_at_utc':sys.argv[3],
 'finished_at_utc':sys.argv[4],'exit_status':int(sys.argv[5]),'success':int(sys.argv[5])==0,
 'log':sys.argv[6],'output_dir':sys.argv[7],'expected_global_step':2127},indent=2)+'\n')
PY
exit "$code"
