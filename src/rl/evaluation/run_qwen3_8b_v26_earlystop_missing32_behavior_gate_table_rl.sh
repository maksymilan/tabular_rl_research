#!/usr/bin/env bash
# Diagnostic-only missing32 K8 behavior gate.  Dry-run is the default.  The
# checkpoint watcher is CPU-only; GPU generation requires explicit modes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|watch-checkpoint6|prepare-cohort|generate-sft1|generate-checkpoint6|audit]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|watch-checkpoint6|prepare-cohort|generate-sft1|generate-checkpoint6|audit) ;;
  *) printf 'unsupported mode: %s\n' "$MODE" >&2; exit 2 ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_earlystop_mixed180_grpo_20260813}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TRAIN_RUN=${TRAIN_RUN:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_grpo_20260813/train180_two_pass_seed20260812}
EARLYSTOP_ROOT=${EARLYSTOP_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813}
EARLYSTOP_MANIFEST=${EARLYSTOP_MANIFEST:-$EARLYSTOP_ROOT/cohort/earlystop_mixed180_manifest.json}
SOURCE_TASKS=${SOURCE_TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_earlystop_missing32_cp6_20260813}
COHORT_DIR=$RUN_ROOT/cohort
TASKS=$COHORT_DIR/missing32.jsonl
COHORT_MANIFEST=$COHORT_DIR/missing32_manifest.json
SFT1_OUT=$RUN_ROOT/sft1_k8_seed20260817
CHECKPOINT_OUT=$RUN_ROOT/checkpoint6_k8_seed20260817
CHECKPOINT6=$TRAIN_RUN/diagnostic_checkpoint-6
SOURCE_CHECKPOINT6=$TRAIN_RUN/checkpoint-6
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
EVAL_GPU_ID=${EVAL_GPU_ID:-0}

PREPARER=$RUNTIME/src/rl/evaluation/prepare_earlystop_missing32_gate.py
PRESERVER=$RUNTIME/src/rl/evaluation/preserve_diagnostic_checkpoint.py
AUDITOR=$RUNTIME/src/rl/evaluation/audit_earlystop_missing32_behavior_gate.py
GENERATOR=$RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py

EXPECTED_SOURCE_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e
EXPECTED_EARLYSTOP_MANIFEST_SHA256=e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch expected=$expected actual=$actual"
}
require_code() {
  [[ -f "$1" && ! -L "$1" ]] || die "missing code: $1"
}
require_gpu_idle() {
  local used apps
  used=$(nvidia-smi -i "$EVAL_GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  apps=$(nvidia-smi -i "$EVAL_GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$used" =~ ^[0-9]+$ && "$used" -le 512 && -z "$apps" ]] \
    || die "GPU$EVAL_GPU_ID is not idle (used_mib=$used apps=$apps)"
}
verify_checkpoint_receipt() {
  "$PYTHON" - "$CHECKPOINT6" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); receipt=json.loads((root/'preservation_receipt.json').read_text())
assert receipt['schema_version']=='diagnostic-checkpoint-preservation-v1'
assert receipt['status']=='preserved' and receipt['source_mutated'] is False
assert receipt['global_step']==6 and receipt['max_steps']==12
for name, record in receipt['files'].items():
    path=root/name
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size==record['size_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
state=json.loads((root/'trainer_state.json').read_text())
assert state['global_step']==6 and state['max_steps']==12
PY
}

case "$MODE" in
  dry-run)
    printf '%s\n' \
      'early behavior gate dry-run (no files, processes, GPU inspection, or remote operations)' \
      'cohort=BIRD-train original600 minus earlystop568 group identities; no reward files read' \
      'arms=fresh SFT1 vs diagnostic checkpoint-6; K8 seed=20260817 temperature=0.8' \
      'runtime=version26 max_new_tokens=2048 max_context_tokens=16384' \
      'gate=mean>=+5pp AND one-sided80%-cluster-bootstrap-lower>0 AND legal_delta>=-5/256' \
      'lifecycle=watch-checkpoint6 -> prepare-cohort -> generate-sft1 -> generate-checkpoint6 -> audit' \
      'formal_dev_consumed=false; formal_checkpoint_selection_authority=false' \
      'deployment=table_rl single idle GPU; NewGNN requires staging the missing database first'
    ;;
  watch-checkpoint6)
    require_code "$PRESERVER"
    [[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
    exec env PYTHONPATH= "$PYTHON" "$PRESERVER" \
      --source "$SOURCE_CHECKPOINT6" --destination "$CHECKPOINT6" \
      --expected-step 6 --expected-max-steps 12 \
      --stable-seconds 5 --poll-seconds 2 --timeout-seconds 172800
    ;;
  prepare-cohort)
    require_code "$PREPARER"
    require_sha "$SOURCE_TASKS" "$EXPECTED_SOURCE_TASKS_SHA256" source600
    require_sha "$EARLYSTOP_MANIFEST" "$EXPECTED_EARLYSTOP_MANIFEST_SHA256" earlystop568_manifest
    [[ ! -e "$COHORT_DIR" ]] || die "cohort output exists: $COHORT_DIR"
    exec env PYTHONPATH= "$PYTHON" "$PREPARER" \
      --source-tasks "$SOURCE_TASKS" --earlystop-manifest "$EARLYSTOP_MANIFEST" \
      --output-dir "$COHORT_DIR"
    ;;
  generate-sft1|generate-checkpoint6)
    require_code "$GENERATOR"
    require_sha "$SOURCE_TASKS" "$EXPECTED_SOURCE_TASKS_SHA256" source600
    require_sha "$EARLYSTOP_MANIFEST" "$EXPECTED_EARLYSTOP_MANIFEST_SHA256" earlystop568_manifest
    env PYTHONPATH= "$PYTHON" "$PREPARER" --verify \
      --source-tasks "$SOURCE_TASKS" --earlystop-manifest "$EARLYSTOP_MANIFEST" \
      --output-dir "$COHORT_DIR" >/dev/null
    [[ -d "$MODEL_PATH" && ! -L "$MODEL_PATH" ]] || die "invalid model path"
    [[ -d "$PROTOCOL_RUNTIME" && ! -L "$PROTOCOL_RUNTIME" ]] || die "invalid protocol runtime"
    if [[ "$MODE" == generate-sft1 ]]; then
      adapter=$SFT1_ADAPTER; output=$SFT1_OUT
      require_sha "$adapter/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
    else
      adapter=$CHECKPOINT6; output=$CHECKPOINT_OUT
      verify_checkpoint_receipt
    fi
    [[ ! -e "$output" ]] || die "generation output exists: $output"
    require_gpu_idle
    exec env CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" HF_HUB_OFFLINE=1 \
      TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" PYTHONPATH="$RUNTIME/src/rl" \
      "$PYTHON" "$GENERATOR" --model-path "$MODEL_PATH" --adapter-path "$adapter" \
      --tasks "$TASKS" --output-dir "$output" --group-size 8 \
      --temperature 0.8 --top-p 1 --max-steps 30 --max-new-tokens 2048 \
      --max-context-tokens 16384 --history-turns 4 --enable-thinking \
      --seed 20260817 --gpu-memory-utilization 0.82 \
      --scheduler dynamic --question-window 4
    ;;
  audit)
    require_code "$AUDITOR"
    verify_checkpoint_receipt
    checkpoint_sha=$(sha256_file "$CHECKPOINT6/adapter_model.safetensors")
    [[ -f "$SFT1_OUT/manifest.pending.json" && -f "$CHECKPOINT_OUT/manifest.pending.json" ]] \
      || die 'both generation arms must be finalized'
    exec env PYTHONPATH="$RUNTIME" "$PYTHON" "$AUDITOR" \
      --cohort-manifest "$COHORT_MANIFEST" --tasks "$TASKS" \
      --sft1-dir "$SFT1_OUT" --checkpoint-dir "$CHECKPOINT_OUT" \
      --expected-checkpoint-adapter-sha256 "$checkpoint_sha" \
      --output "$RUN_ROOT/paired_gate.json"
    ;;
esac
