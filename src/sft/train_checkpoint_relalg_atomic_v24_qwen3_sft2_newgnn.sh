#!/usr/bin/env bash
set -euo pipefail

# Four-GPU fresh QLoRA training from the frozen Qwen3 TrustSQL base.
# RUN_KIND=smoke performs two optimizer steps on repeated longest audited rows;
# RUN_KIND=full performs one epoch over the admitted Atomic-v24 training view.

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/qwen3-atomic-sft}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_checkpoint_relalg_atomic_v24_sft2_20260821_r2}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_GPU_MEMORY_MIB=${MAX_GPU_MEMORY_MIB:-512}
RUN_KIND=${RUN_KIND:-smoke}

case "$RUN_KIND" in
  smoke)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_8b_from_base_smoke_qlora_8192.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/checkpoint_relalg_atomic_v24_union_qwen3_8192_longest_smoke.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-from-base-8192-qlora-smoke}
    EXPECTED_STEPS=2
    ;;
  full)
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_8b_from_base_qlora_8192.yaml}
    DATASET_FILE=${DATASET_FILE:-$DATASET_DIR/checkpoint_relalg_atomic_v24_union_qwen3_8192_training_view.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-from-base-8192-qlora}
    EXPECTED_STEPS=auto
    ;;
  *) echo "RUN_KIND must be smoke or full" >&2; exit 2 ;;
esac

RUN_ID=${RUN_ID:-qwen3_checkpoint_relalg_atomic_v24_sft2_${RUN_KIND}_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
LOG=$LOG_DIR/$RUN_ID.log
PID_FILE=$LOG_DIR/$RUN_ID.pid
LAUNCH_MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json
STATUS_MANIFEST=$LOG_DIR/$RUN_ID.status.json
DATASET_MANIFEST=${DATASET_FILE%.jsonl}.manifest.json
DATASET_INFO=$DATASET_DIR/dataset_info.json
PREPARATION_MANIFEST=$DATASET_DIR/preparation_manifest.json

IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
if [[ ${#physical_gpus[@]} -ne 4 ]]; then
  echo "GPU_IDS must name exactly four GPUs" >&2; exit 2
fi
for required in "$CONFIG" "$ENV_DIR/bin/llamafactory-cli" "$MODEL_DIR/config.json" \
  "$DATASET_FILE" "$DATASET_MANIFEST" "$DATASET_INFO" "$PREPARATION_MANIFEST"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 3; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 4; }
mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for artifact in "$LOG" "$PID_FILE" "$LAUNCH_MANIFEST" "$STATUS_MANIFEST"; do
  [[ ! -e "$artifact" ]] || { echo "refusing existing run artifact: $artifact" >&2; exit 4; }
done
for gpu in "${physical_gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] || exit 5
  (( used <= MAX_GPU_MEMORY_MIB )) || { echo "GPU $gpu is not idle: $used MiB" >&2; exit 6; }
done
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so

EXPECTED_STEPS=$(
  "$ENV_DIR/bin/python" - "$CONFIG" "$DATASET_FILE" "$DATASET_MANIFEST" \
    "$DATASET_INFO" "$PREPARATION_MANIFEST" "$MODEL_DIR" \
    "$OUTPUT_DIR" "$RUN_KIND" "$EXPECTED_STEPS" "$LAUNCH_MANIFEST" <<'PY'
import hashlib, json, math, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

config, dataset, dataset_manifest, dataset_info, prep, model, output = map(Path, sys.argv[1:8])
run_kind, expected_arg, launch_path = sys.argv[8], sys.argv[9], Path(sys.argv[10])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
cfg=yaml.safe_load(config.read_text())
manifest=json.loads(dataset_manifest.read_text())
records=sum(bool(x.strip()) for x in dataset.open())
if manifest.get('output_sha256') != sha(dataset) or manifest.get('records', manifest.get('repeats')) != records:
    raise SystemExit('dataset manifest binding failed')
if cfg.get('adapter_name_or_path') is not None:
    raise SystemExit('fresh-base run must not load an existing adapter')
expected_cfg={
 'model_name_or_path':str(model),'dataset':dataset.stem,'dataset_dir':str(dataset.parent),
 'output_dir':str(output),'template':'qwen3','enable_thinking':True,'preserve_thinking':False,
 'mask_history':True,'cutoff_len':8192,'quantization_bit':4,'per_device_train_batch_size':1,
 'gradient_accumulation_steps':8,'learning_rate':1e-4,
}
for k,v in expected_cfg.items():
    if cfg.get(k)!=v: raise SystemExit(f'config mismatch {k}: {cfg.get(k)!r} != {v!r}')
if run_kind=='smoke':
    if records!=32 or int(cfg.get('max_steps'))!=2: raise SystemExit('smoke size/steps mismatch')
    expected=2
else:
    if float(cfg.get('num_train_epochs'))!=1.0: raise SystemExit('full epoch mismatch')
    expected=math.ceil(math.ceil(records/4)/8)
preparation=json.loads(prep.read_text())
if preparation.get('training_view_sha256') != sha(dataset) and run_kind=='full':
    raise SystemExit('preparation manifest does not bind training view')
payload={
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_kind':run_kind,
 'experiment':'qwen3-8b-checkpoint-relalg-atomic-v24-from-base',
 'protocol_boundary':'checkpoint-relalg atomic-v24-frozen-v1 Text-JSON, checkpoint disabled',
 'initialization':'fresh LoRA adapter from Qwen3-8B TrustSQL base; no prior adapter, optimizer, scheduler, or global step',
 'model':str(model),'model_config_sha256':sha(model/'config.json'),
 'adapter':None,'adapter_model_sha256':None,
 'config':str(config),'config_sha256':sha(config),'dataset':str(dataset),
 'dataset_sha256':sha(dataset),'dataset_manifest_sha256':sha(dataset_manifest),
 'dataset_info_sha256':sha(dataset_info),'preparation_manifest_sha256':sha(prep),
 'records':records,'physical_gpus':sys.argv[0:0] or ['0','1','2','3'],
 'world_size':4,'effective_global_batch':32,'expected_optimizer_steps':expected,
 'output_dir':str(output),'python':sys.executable,'python_version':platform.python_version(),
}
for name in ('torch','transformers','peft','bitsandbytes','datasets','llamafactory','liger_kernel'):
    try:
        module=__import__(name); payload[name]=getattr(module,'__version__','unknown')
    except Exception as exc: payload[name]=f'import-error:{exc}'
launch_path.write_text(json.dumps(payload,indent=2)+'\n')
print(expected)
PY
)

mkdir "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS" FORCE_TORCHRUN=1 NPROC_PER_NODE=4 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 LIBRARY_PATH="/home/dengyan/cuda_link:${LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:${PATH}"
echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "$LOG_DIR/qwen3_checkpoint_relalg_atomic_v24_sft2_${RUN_KIND}_latest.run_id"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$ENV_DIR/bin/llamafactory-cli" train "$CONFIG" > "$LOG" 2>&1
status=$?
set -e
finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ $status -eq 0 ]]; then
  "$ENV_DIR/bin/python" - "$OUTPUT_DIR" "$EXPECTED_STEPS" <<'PY' || status=8
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=int(sys.argv[2])
states=[json.loads(p.read_text()).get('global_step') for p in root.rglob('trainer_state.json')]
if not states or max(states)!=expected: raise SystemExit(f'global step mismatch: {states} vs {expected}')
if not (root/'adapter_model.safetensors').exists(): raise SystemExit('missing final adapter')
PY
fi
"$ENV_DIR/bin/python" - "$STATUS_MANIFEST" "$RUN_ID" "$RUN_KIND" "$started_at" "$finished_at" "$status" "$LOG" "$OUTPUT_DIR" "$EXPECTED_STEPS" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 'run_id':sys.argv[2],'run_kind':sys.argv[3],'started_at_utc':sys.argv[4],
 'finished_at_utc':sys.argv[5],'exit_status':int(sys.argv[6]),'success':int(sys.argv[6])==0,
 'log':sys.argv[7],'output_dir':sys.argv[8],'expected_global_step':int(sys.argv[9])},indent=2)+'\n')
PY
echo "status=$status log=$LOG output=$OUTPUT_DIR"
exit "$status"
