#!/usr/bin/env bash
# Independent diagnostic SFT1-vs-checkpoint-4 missing32 gate on NewGNN GPU6.
# Default is dry-run.  It never reads or selects on checkpoint-6/final results.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|preflight|generate-sft1|generate-checkpoint4|audit|run]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|preflight|generate-sft1|generate-checkpoint4|audit|run) ;;
  *) printf 'unsupported mode: %s\n' "$MODE" >&2; exit 2 ;;
esac

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$O/rl_runtime_qwen3_8b_v26_earlystop_mixed180_grpo_20260813}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$O/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1=${SFT1_ADAPTER:-$O/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
RUN_ROOT=${RUN_ROOT:-$O/evaluations/qwen3_8b_atomic_v26_earlystop_missing32_cp4_20260813}
CP4=${CHECKPOINT4_ADAPTER:-$RUN_ROOT/checkpoint-4}
SOURCE_CP6_COHORT=${SOURCE_CP6_COHORT_DIR:-$RUN_ROOT/cohort}
COHORT=${COHORT_DIR:-$RUN_ROOT/cohort_checkpoint4_seed20260816}
TASKS=$COHORT/missing32.jsonl
COHORT_MANIFEST=$COHORT/missing32_manifest.json
SFT1_OUT=$RUN_ROOT/sft1_k8_seed20260816
CP4_OUT=$RUN_ROOT/checkpoint4_k8_seed20260816
AUDIT_OUT=$RUN_ROOT/checkpoint4_paired_gate.json
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
GPU_ID=${GPU_ID:-6}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_STABLE_SECONDS=${GPU_STABLE_SECONDS:-5}

PREPARER=$RUNTIME/src/rl/scenarios/evaluation/prepare_earlystop_missing32_gate.py
AUDITOR=$RUNTIME/src/rl/scenarios/evaluation/audit_earlystop_missing32_behavior_gate.py
GENERATOR=$RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py

EXPECTED_PREPARER_SHA256=ba5df2809f93101684e55552ba05d0e5698e4984c813694fcbf6a9e1636a48b3
EXPECTED_AUDITOR_SHA256=c6d5490c59be42dab52872a21914c82304b636b7cb53ac95ab1771862688d081
EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191
EXPECTED_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_TASKS_SHA256=014e8ddc5cd6948510c9ef8dde0067915522b49d99e31472b75f4afaacde97bd
EXPECTED_COHORT_MANIFEST_SHA256=6355325166bc4deb55852040146c90d6e958abe5a663abe262f5a08fd3a144a6
EXPECTED_BASE_MODEL_IDENTITY_SHA256=85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_CP4_SHA256=677746a38f9f16542b06ed4345e947207f5bd81c240efa4d56dd8fd95efdc2ee
EXPECTED_CP4_CONFIG_SHA256=43860399e9bddc89fad3b52d0380ccee0e461725c6fda81a5bcd880dd5cb2d7d
EXPECTED_CP4_TOKENIZER_CONFIG_SHA256=443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0
EXPECTED_CHAT_TEMPLATE_SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
EXPECTED_STAGED_DB_SHA256=1a7c2e8e0868a3c61f0cf6dc936483b31e05d86226400273280a5975c5fc99d5
STAGED_DB_PATH=${STAGED_DB_PATH:-/home/dengyan/tabular_rl_project/data/bird/train/train_databases/european_football_1/european_football_1.sqlite}

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch expected=$expected actual=$actual"
}

dry_run() {
  printf '%s\n' \
    'checkpoint-4 missing32 gate dry-run (no files, GPU inspection, or processes)' \
    'host=NewGNN gpu=6 diagnostic_only=true formal_dev_consumed=false' \
    'arms=SFT1 vs checkpoint-4(global_step=4); checkpoint-6/final independent and ignored' \
    'tasks=32 K8 seed=20260816 temperature=0.8 top_p=1' \
    'protocol=version26 max_new_tokens=2048 max_context_tokens=16384' \
    'outputs use checkpoint-4 schema/labels; cannot be consumed as checkpoint-6 evidence' \
    'modes=preflight -> generate-sft1 -> generate-checkpoint4 -> audit (or explicit run)'
}

verify_runtime_tree() {
  env PYTHONPATH= "$PY" - "$PROTOCOL_RUNTIME" "$EXPECTED_RUNTIME_TREE_SHA256" <<'PY'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]; files=[]
for relative in ('src/eval','src/sft','src/harness'):
    directory=root/relative
    assert directory.is_dir() and not directory.is_symlink(), directory
    files += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
digest=hashlib.sha256()
for path in sorted(files,key=lambda p:p.relative_to(root).as_posix()):
    rel=path.relative_to(root).as_posix(); digest.update(rel.encode()); digest.update(b'\0')
    digest.update(path.read_bytes()); digest.update(b'\0')
assert digest.hexdigest()==expected,(digest.hexdigest(),expected)
PY
}

verify_model() {
  env PYTHONPATH= "$PY" - "$MODEL" "$EXPECTED_BASE_MODEL_IDENTITY_SHA256" "$EXPECTED_CHAT_TEMPLATE_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected_aggregate=sys.argv[2]; expected_template=sys.argv[3]
expected={
'config.json':'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
'generation_config.json':'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
'merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
'model-00001-of-00005.safetensors':'31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f',
'model-00002-of-00005.safetensors':'5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282',
'model-00003-of-00005.safetensors':'c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836',
'model-00004-of-00005.safetensors':'b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a',
'model-00005-of-00005.safetensors':'20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff',
'model.safetensors.index.json':'f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc',
'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
observed={}
for name,want in expected.items():
    path=root/name; assert path.is_file() and not path.is_symlink(),path
    got=hashlib.sha256(path.read_bytes()).hexdigest(); assert got==want,(name,got,want); observed[name]=got
aggregate=hashlib.sha256()
for name in sorted(observed,key=lambda value:value.encode()): aggregate.update(f'{name}\t{observed[name]}\n'.encode())
assert aggregate.hexdigest()==expected_aggregate
template=json.loads((root/'tokenizer_config.json').read_text())['chat_template']
assert hashlib.sha256(template.encode()).hexdigest()==expected_template
PY
}

verify_cp4() {
  require_sha "$CP4/adapter_model.safetensors" "$EXPECTED_CP4_SHA256" cp4_adapter
  require_sha "$CP4/adapter_config.json" "$EXPECTED_CP4_CONFIG_SHA256" cp4_adapter_config
  require_sha "$CP4/tokenizer_config.json" "$EXPECTED_CP4_TOKENIZER_CONFIG_SHA256" cp4_tokenizer_config
  require_sha "$CP4/chat_template.jinja" "$EXPECTED_CHAT_TEMPLATE_SHA256" cp4_chat_template
  "$PY" - "$CP4" "$EXPECTED_CHAT_TEMPLATE_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
state=json.loads((root/'trainer_state.json').read_text())
assert int(state['global_step'])==4 and int(state['max_steps'])==12
template=(root/'chat_template.jinja').read_bytes()
assert hashlib.sha256(template).hexdigest()==expected
PY
}

verify_tasks_and_databases() {
  require_sha "$TASKS" "$EXPECTED_TASKS_SHA256" missing32_tasks
  require_sha "$COHORT_MANIFEST" "$EXPECTED_COHORT_MANIFEST_SHA256" missing32_manifest
  env PYTHONPATH="$RUNTIME" "$PY" "$PREPARER" --checkpoint4-sibling \
    --source-cohort-dir "$SOURCE_CP6_COHORT" --output-dir "$COHORT" \
    --manifest-tasks-path "$TASKS" --verify >/dev/null
  [[ "$STAGED_DB_PATH" == /home/dengyan/* ]] \
    || die 'STAGED_DB_PATH is not bound to the preregistered missing NewGNN database'
  require_sha "$STAGED_DB_PATH" "$EXPECTED_STAGED_DB_SHA256" staged_missing_database
  "$PY" - "$TASKS" "$COHORT_MANIFEST" "$STAGED_DB_PATH" <<'PY'
import json,sys
from pathlib import Path
tasks,manifest,staged=map(Path,sys.argv[1:])
rows=[json.loads(line) for line in tasks.read_text().splitlines() if line]
cohort=json.loads(manifest.read_text()); assert len(rows)==32
assert cohort['schema_version']=='qwen3-v26-earlystop-missing32-behavior-cohort-v1'
assert cohort['status']=='frozen_reward_blind_ordered_complement'
assert cohort['evaluation_contract']['arms']==['sft1','checkpoint-4']
assert cohort['evaluation_contract']['seed']==20260816
ids=[row.get('example_id') or row.get('instance_id') for row in rows]
assert cohort['selection']['ordered_task_ids']==ids
paths={Path(row['db_path']).resolve() for row in rows}
assert staged.resolve() in paths, (staged,paths)
for path in paths: assert path.is_file() and not path.is_symlink(),path
PY
}

preflight() {
  [[ "$GPU_ID" == 6 ]] || die "NewGNN checkpoint-4 gate is restricted to GPU6, got=$GPU_ID"
  [[ -x "$PY" ]] || die "missing Python: $PY"
  require_sha "$PREPARER" "$EXPECTED_PREPARER_SHA256" cohort_preparer
  require_sha "$AUDITOR" "$EXPECTED_AUDITOR_SHA256" paired_auditor
  require_sha "$GENERATOR" "$EXPECTED_GENERATOR_SHA256" fixed_pool_generator
  verify_runtime_tree
  verify_model
  require_sha "$SFT1/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
  require_sha "$SFT1/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256" sft1_adapter_config
  verify_cp4
  verify_tasks_and_databases
  # Import from an unrelated cwd without allocating CUDA.  This catches a
  # staged-runtime PYTHONPATH that works only from a repository checkout.
  (
    cd /tmp
    env CUDA_VISIBLE_DEVICES='' TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
      PYTHONPATH="$RUNTIME" "$PY" "$GENERATOR" --help >/dev/null
  )
}

require_idle_gpu6() {
  local used apps
  used=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  apps=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$used" =~ ^[0-9]+$ && "$used" -le "$GPU_FREE_THRESHOLD_MIB" && -z "$apps" ]] \
    || die "GPU6 is occupied used_mib=$used pids=${apps:-none}"
  sleep "$GPU_STABLE_SECONDS"
  used=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  apps=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$used" =~ ^[0-9]+$ && "$used" -le "$GPU_FREE_THRESHOLD_MIB" && -z "$apps" ]] \
    || die "GPU6 did not remain idle used_mib=$used pids=${apps:-none}"
}

owned_pgid=''
stop_owned() {
  [[ -n "$owned_pgid" ]] || return 0
  if kill -0 -- "-$owned_pgid" 2>/dev/null; then
    kill -TERM -- "-$owned_pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$owned_pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$owned_pgid" 2>/dev/null && kill -KILL -- "-$owned_pgid" 2>/dev/null || true
  fi
  wait "$owned_pgid" 2>/dev/null || true
  owned_pgid=''
}
trap stop_owned EXIT
trap 'stop_owned; exit 130' INT
trap 'stop_owned; exit 143' TERM

generate_arm() {
  local arm=$1 adapter output log
  if [[ "$arm" == sft1 ]]; then
    adapter=$SFT1; output=$SFT1_OUT; log=$RUN_ROOT/sft1.log
  else
    adapter=$CP4; output=$CP4_OUT; log=$RUN_ROOT/checkpoint4.log
  fi
  [[ ! -e "$output" ]] || die "refusing existing $arm output: $output"
  require_idle_gpu6
  mkdir -p "$RUN_ROOT"
  setsid env CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" PYTHONPATH="$RUNTIME" \
    "$PY" "$GENERATOR" --model-path "$MODEL" --adapter-path "$adapter" \
    --tasks "$TASKS" --output-dir "$output" --group-size 8 \
    --temperature 0.8 --top-p 1 --max-steps 30 --max-new-tokens 2048 \
    --max-context-tokens 16384 --history-turns 4 --enable-thinking \
    --seed 20260816 --gpu-memory-utilization 0.82 \
    --scheduler dynamic --question-window 4 >>"$log" 2>&1 &
  owned_pgid=$!
  local code=0
  if wait "$owned_pgid"; then code=0; else code=$?; fi
  owned_pgid=''
  [[ "$code" == 0 ]] || die "$arm generation failed exit=$code log=$log"
}

audit_pair() {
  [[ -f "$SFT1_OUT/manifest.pending.json" && -f "$CP4_OUT/manifest.pending.json" ]] \
    || die 'both checkpoint-4 diagnostic arms must be complete'
  [[ ! -e "$AUDIT_OUT" ]] || die "refusing existing audit: $AUDIT_OUT"
  env PYTHONPATH="$RUNTIME" "$PY" "$AUDITOR" \
    --checkpoint-arm checkpoint-4 --cohort-manifest "$COHORT_MANIFEST" \
    --tasks "$TASKS" --sft1-dir "$SFT1_OUT" --checkpoint-dir "$CP4_OUT" \
    --expected-checkpoint-adapter-sha256 "$EXPECTED_CP4_SHA256" --output "$AUDIT_OUT"
}

if [[ "$MODE" == dry-run ]]; then dry_run; exit 0; fi
preflight
if [[ "$MODE" == preflight ]]; then printf 'checkpoint-4 NewGNN preflight passed\n'; exit 0; fi
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/checkpoint4_gate.lock"
flock -n 9 || die 'another checkpoint-4 gate invocation owns this run'
case "$MODE" in
  generate-sft1) generate_arm sft1 ;;
  generate-checkpoint4) generate_arm checkpoint4 ;;
  audit) audit_pair ;;
  run)
    generate_arm sft1
    generate_arm checkpoint4
    audit_pair
    ;;
esac
