#!/usr/bin/env bash
# Host-neutral, hash-pinned continuation worker for the frozen S1 K8 screen.
# It never chooses tasks, merges groups, finalizes a pool, or signals a process.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|worker]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|worker) ;;
  *) printf 'usage: %s [dry-run|worker]\n' "$0" >&2; exit 2 ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_boundary_screen_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
TASKS_JSONL=${TASKS_JSONL:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}

RUN_ROOT=${RUN_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
ASSIGNMENT=${ASSIGNMENT:-}
ASSIGNMENT_SHA256=${ASSIGNMENT_SHA256:-}
SCREEN_GPU=${SCREEN_GPU:-}

EXPECTED_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e
EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191
EXPECTED_ROLLOUT_SHA256=87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f
EXPECTED_SCORING_SHA256=a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5
EXPECTED_TASK_LOADER_SHA256=d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b
EXPECTED_TOOL_ENV_V26_SHA256=c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

if [[ "$MODE" == dry-run ]]; then
  printf '%s\n' \
    'boundary S1 continuation dry-run: no files written, no GPU inspected' \
    'worker requires RUN_ROOT, OUTPUT_DIR, ASSIGNMENT, ASSIGNMENT_SHA256, SCREEN_GPU' \
    'contract=initial-SFT1,K8,T0.8,top_p1,seed20260812,2048/16384,thinking=true,no-finalize'
  exit 0
fi

[[ -n "$RUN_ROOT" && -n "$OUTPUT_DIR" && -n "$ASSIGNMENT" ]] \
  || die 'RUN_ROOT, OUTPUT_DIR, and ASSIGNMENT are required'
[[ "$ASSIGNMENT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die 'ASSIGNMENT_SHA256 must be explicit lowercase SHA-256'
[[ "$SCREEN_GPU" =~ ^[0-9]+$ ]] || die 'SCREEN_GPU must be an explicit index'
case "$RUN_ROOT" in "$OUTPUT_ROOT"/*) ;; *) die "RUN_ROOT is outside OUTPUT_ROOT: $RUN_ROOT" ;; esac
case "$OUTPUT_DIR" in "$RUN_ROOT"/*) ;; *) die "OUTPUT_DIR is outside RUN_ROOT: $OUTPUT_DIR" ;; esac
[[ ! -L "$RUN_ROOT" && ! -L "$OUTPUT_DIR" ]] || die 'run/output roots may not be symlinks'
[[ -x "$PYTHON_BIN" ]] || die "missing Python: $PYTHON_BIN"
[[ -d "$MODEL_PATH" && -d "$SFT1_ADAPTER" ]] || die 'missing model/SFT1 adapter'
[[ -d "$PROTOCOL_RUNTIME/src/eval" && -d "$PROTOCOL_RUNTIME/src/sft" && -d "$PROTOCOL_RUNTIME/src/harness" ]] \
  || die "incomplete frozen protocol runtime: $PROTOCOL_RUNTIME"

require_sha "$TASKS_JSONL" "$EXPECTED_TASKS_SHA256" train600_tasks
require_sha "$ASSIGNMENT" "$ASSIGNMENT_SHA256" continuation_assignment
require_sha "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" "$EXPECTED_GENERATOR_SHA256" generator
require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" "$EXPECTED_ROLLOUT_SHA256" rollout
require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" "$EXPECTED_SCORING_SHA256" scoring
require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" "$EXPECTED_TASK_LOADER_SHA256" task_loader
require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" "$EXPECTED_TOOL_ENV_V26_SHA256" tool_environment_v26

env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$ASSIGNMENT" \
  "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
tasks, assignment, runtime = map(Path, sys.argv[1:4])
expected_tree = sys.argv[4]
import json
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
selected = [line.strip() for line in assignment.read_text().splitlines() if line.strip()]
assert len(ids) == len(set(ids)) == 600
assert selected and len(selected) == len(set(selected))
assert set(selected) <= set(ids[32:])
digest = hashlib.sha256(); files = []
for relative in ("src/eval", "src/sft", "src/harness"):
    files.extend(
        path for path in (runtime / relative).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
for path in sorted(files, key=lambda value: value.relative_to(runtime).as_posix()):
    relative = path.relative_to(runtime).as_posix()
    digest.update(relative.encode()); digest.update(b"\0")
    digest.update(path.read_bytes()); digest.update(b"\0")
assert len(files) == 101 and digest.hexdigest() == expected_tree
PY

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  | sed -n "$(( SCREEN_GPU + 1 ))p" | tr -d '[:space:]')
[[ "$used" =~ ^[0-9]+$ ]] || die "cannot inspect GPU$SCREEN_GPU"
(( used <= 512 )) || die "GPU$SCREEN_GPU is busy (${used} MiB); no process will be stopped"

mkdir -p "$OUTPUT_DIR" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/$(basename "$OUTPUT_DIR").lock"
flock -n 9 || die "continuation output already has an owner: $OUTPUT_DIR"

exec env \
  CUDA_VISIBLE_DEVICES="$SCREEN_GPU" \
  HF_HUB_OFFLINE=1 \
  TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
  PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
  "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$SFT1_ADAPTER" \
    --tasks "$TASKS_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --task-id-file "$ASSIGNMENT" \
    --no-finalize \
    --group-size 8 \
    --temperature 0.8 \
    --top-p 1 \
    --max-steps 30 \
    --max-new-tokens 2048 \
    --max-context-tokens 16384 \
    --history-turns 4 \
    --enable-thinking \
    --seed 20260812 \
    --gpu-memory-utilization 0.82 \
    --scheduler dynamic \
    --question-window 4
