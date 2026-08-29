#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/sft}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_checkpoint_relalg_atomic_v24_sft3_20260825_projection_v3}
MAIN_DATASET_NAME=checkpoint_relalg_atomic_v24_union_qwen3_projection_v3_16384
MAIN_DATASET_MANIFEST=$DATASET_DIR/$MAIN_DATASET_NAME.manifest.json
CARRIER_AUDIT=$DATASET_DIR/qwen3_projection_v3_audit.json
TOKEN_AUDIT=$DATASET_DIR/$MAIN_DATASET_NAME.token_audit.json
DATASET_INFO=$DATASET_DIR/dataset_info.json
RUN_KIND=${RUN_KIND:-smoke}
GPU_IDS=${GPU_IDS:-0,1}

case "$RUN_KIND" in
  smoke)
    DATASET_NAME=${MAIN_DATASET_NAME}_smoke_longest32
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_projection_v3_smoke_table_rl_2gpu_qlora_16384.yaml}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-projection-v3-smoke-16384-qlora}
    ;;
  full)
    DATASET_NAME=$MAIN_DATASET_NAME
    CONFIG=${CONFIG:-$PROJECT_DIR/src/sft/configs/checkpoint_relalg_atomic_v24_qwen3_projection_v3_from_base_table_rl_2gpu_qlora_16384.yaml}
    OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/checkpoints/qwen3-8b-checkpoint-relalg-atomic-v24-projection-v3-from-base-16384-qlora}
    ;;
  *) echo "RUN_KIND must be smoke or full" >&2; exit 2 ;;
esac
DATASET=$DATASET_DIR/$DATASET_NAME.jsonl
DATASET_MANIFEST=$DATASET_DIR/$DATASET_NAME.manifest.json

RUN_ID=${RUN_ID:-qwen3_atomic_v24_projection_v3_${RUN_KIND}_table_rl_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$OUTPUT_ROOT/logs
LOG=$LOG_DIR/$RUN_ID.log
STATUS=$LOG_DIR/$RUN_ID.status.json
MANIFEST=$LOG_DIR/$RUN_ID.launch_manifest.json

for path in "$CONFIG" "$DATASET" "$DATASET_MANIFEST" "$MAIN_DATASET_MANIFEST" "$CARRIER_AUDIT" "$TOKEN_AUDIT" \
  "$DATASET_INFO" "$MODEL_DIR/config.json" "$ENV_DIR/bin/llamafactory-cli"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 3; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 4; }
mkdir -p "$LOG_DIR" /home/dengyan/cuda_link
for path in "$LOG" "$STATUS" "$MANIFEST"; do
  [[ ! -e "$path" ]] || { echo "refusing existing artifact: $path" >&2; exit 4; }
done

IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
[[ ${#physical_gpus[@]} -eq 2 ]] || { echo "GPU_IDS must name exactly two GPUs" >&2; exit 5; }
for gpu in "${physical_gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] || exit 5
  (( used <= 512 )) || { echo "GPU $gpu is not idle: $used MiB" >&2; exit 6; }
done

read -r RECORDS EXPECTED_STEPS DATASET_SHA PROMPT_SHA < <(
  "$ENV_DIR/bin/python" - "$CONFIG" "$DATASET" "$DATASET_MANIFEST" "$MAIN_DATASET_MANIFEST" \
    "$CARRIER_AUDIT" "$TOKEN_AUDIT" "$DATASET_INFO" "$MODEL_DIR" "$OUTPUT_DIR" "$RUN_KIND" "$MANIFEST" <<'PY'
import hashlib,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
config,dataset,dataset_manifest,main_manifest,audit,token_audit,dataset_info,model,output=map(Path,sys.argv[1:10])
run_kind,manifest=sys.argv[10],Path(sys.argv[11])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
selected=json.loads(dataset_manifest.read_text()); binding=json.loads(main_manifest.read_text())
carrier=json.loads(audit.read_text())
tokens=json.loads(token_audit.read_text()); registry=json.loads(dataset_info.read_text())
combined=binding['outputs']['combined']; dataset_sha=sha(dataset)
main_records=combined['records']
if binding.get('schema_version')!='checkpoint-relalg-qwen3-projection-v3-dataset-v1':raise SystemExit('dataset manifest schema mismatch')
if run_kind=='smoke':
 if selected.get('schema_version')!='checkpoint-relalg-qwen3-longest-smoke-v1':raise SystemExit('smoke manifest schema mismatch')
 if selected.get('output_sha256')!=dataset_sha or selected.get('source_training_view_sha256')!=combined.get('training_view_sha256'):raise SystemExit('smoke manifest hash mismatch')
 records=selected['records']
else:
 if combined.get('training_view_sha256')!=dataset_sha:raise SystemExit('dataset manifest hash mismatch')
 records=combined['records']
if not carrier.get('passed') or carrier.get('training_view_sha256')!=combined.get('training_view_sha256'):raise SystemExit('carrier audit mismatch')
prompt_sha=carrier.get('qwen3_student_prompt_sha256')
if carrier.get('provider_carrier_leaks')!=0 or carrier.get('records')!=main_records:raise SystemExit('carrier audit counters mismatch')
if tokens.get('records')!=main_records or tokens.get('cutoff_len')!=16384:raise SystemExit('token audit identity mismatch')
if tokens.get('kept_by_filter_policy')!=main_records or tokens.get('dropped_by_filter_policy')!=0:raise SystemExit('token audit is not fully admitted')
name=('checkpoint_relalg_atomic_v24_union_qwen3_projection_v3_16384_smoke_longest32' if run_kind=='smoke' else 'checkpoint_relalg_atomic_v24_union_qwen3_projection_v3_16384')
if registry.get(name,{}).get('file_name')!=dataset.name:raise SystemExit('dataset registry mismatch')
cfg=yaml.safe_load(config.read_text())
expected={'model_name_or_path':str(model),'dataset':name,'dataset_dir':str(dataset.parent),'output_dir':str(output),
 'template':'qwen3','enable_thinking':True,'preserve_thinking':False,'mask_history':True,'cutoff_len':16384,
 'quantization_bit':4,'per_device_train_batch_size':1}
for key,value in expected.items():
 if cfg.get(key)!=value:raise SystemExit(f'config mismatch {key}: {cfg.get(key)!r}')
if cfg.get('adapter_name_or_path') is not None:raise SystemExit('fresh run cannot load an adapter')
expected_steps=2 if run_kind=='smoke' else math.ceil(records/32)
if run_kind=='smoke':
 if cfg.get('max_steps')!=2 or records!=32:raise SystemExit('smoke config mismatch')
else:
 if cfg.get('gradient_accumulation_steps')!=16 or cfg.get('num_train_epochs')!=1.0:raise SystemExit('full schedule mismatch')
payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_kind':run_kind,
 'experiment':'qwen3-8b-checkpoint-relalg-atomic-v24-projection-v3-from-base','host':'table_rl',
 'physical_gpus':[0,1],'world_size':2,'protocol_boundary':'checkpoint-relalg atomic-v24-frozen-v1; checkpoint disabled; Qwen3 inline carrier v1',
 'data_policy':binding['quality_policy_version'],'initialization':'fresh from Qwen3 TrustSQL base; no adapter',
 'model':str(model),'model_config_sha256':sha(model/'config.json'),'config':str(config),'config_sha256':sha(config),
 'dataset':str(dataset),'dataset_sha256':dataset_sha,'dataset_manifest_sha256':sha(dataset_manifest),
 'main_dataset_manifest_sha256':sha(main_manifest),
 'carrier_audit_sha256':sha(audit),'token_audit_sha256':sha(token_audit),'records':records,
 'context_bucket_records':binding['selection']['context_bucket_records'],'artifact_tier_records':binding['selection']['artifact_tier_records'],
 'qwen3_prompt_sha256':prompt_sha,'effective_global_batch':(2 if run_kind=='smoke' else 32),
 'expected_optimizer_steps':expected_steps,'output_dir':str(output)}
manifest.write_text(json.dumps(payload,indent=2)+'\n')
print(records,expected_steps,dataset_sha,prompt_sha)
PY
)

ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/dengyan/cuda_link/libcuda.so
mkdir "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 HF_HUB_OFFLINE=1
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
  "$ENV_DIR/bin/python" - "$OUTPUT_DIR" "$EXPECTED_STEPS" <<'PY' || code=8
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);expected=int(sys.argv[2])
state=json.loads((root/'trainer_state.json').read_text())
if state.get('global_step')!=expected:raise SystemExit(f'global step mismatch: {state.get("global_step")}')
if not (root/'adapter_model.safetensors').is_file():raise SystemExit('missing final adapter')
PY
fi
"$ENV_DIR/bin/python" - "$STATUS" "$RUN_ID" "$started" "$finished" "$code" "$LOG" "$OUTPUT_DIR" "$EXPECTED_STEPS" "$RECORDS" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'run_id':sys.argv[2],'started_at_utc':sys.argv[3],
 'finished_at_utc':sys.argv[4],'exit_status':int(sys.argv[5]),'success':int(sys.argv[5])==0,
 'log':sys.argv[6],'output_dir':sys.argv[7],'expected_global_step':int(sys.argv[8]),
 'records':int(sys.argv[9])},indent=2)+'\n')
PY
exit "$code"
