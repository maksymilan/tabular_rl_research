#!/usr/bin/env bash
# Durable chain: matched-budget SAAM mixed180 training with gradient diagnostics disabled
# -> strict artifact gate
# -> fresh final-only BIRD-dev1534 evaluation -> exact paired SFT1 summary.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---plan}
if [[ $# -gt 1 || ( "$MODE" != --plan && "$MODE" != --run ) ]]; then
  printf 'usage: %s [--plan|--run]\n' "$0" >&2
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_optimized_20260831}
EVAL_RUNTIME=${EVAL_RUNTIME:-$OUTPUT_ROOT/eval_runtime_qwen3_8b_v26_saam_asymmetric_mixed180_reason_toolgrad_memfix_20260831}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_asymmetric_mixed180_optimized_20260831}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train180_two_pass_seed20260812}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
TRAIN_LAUNCHER=$TRAIN_RUNTIME/src/rl/experiments/run_qwen3_8b_atomic_v26_saam_asymmetric_mixed180_table_rl.sh
EVAL_ENTRY=$EVAL_RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_saam_candidate_only_eval.py
EVAL_CONTRACT=$EVAL_RUNTIME/src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json
PAIR_ENTRY=$EVAL_RUNTIME/src/rl/diagnostics/compare_saam_candidate_to_sft1.py
SFT1_ALL=${SFT1_ALL:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_formal/formal_sft1_vs_final12_20260816_portbarrier_retry3/sft1/result/all.jsonl}
EVAL_ROOT=${EVAL_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_saam_asymmetric_mixed180_optimized_20260831}
EVAL_RUN=${EVAL_RUN:-$EVAL_ROOT/final_dev1534}
PAIR_OUTPUT=$EVAL_RUN/paired_vs_sft1.json
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
EVAL_GPU=${EVAL_GPU:-0}
EVAL_PORT=${EVAL_PORT:-8087}
MAXIMUM_USED_MIB=${MAXIMUM_USED_MIB:-512}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-900}
POLL_SECONDS=${POLL_SECONDS:-10}

EXPECTED_TRAIN_LAUNCHER_SHA256=e3cd850cb312ae7fe3322a7bab64405ef55bd222361d5d766c14e5a801712596
EXPECTED_EVAL_ENTRY_SHA256=d1bd7b523fd7dfa2d14c441647efb51fe4707780a3a5dc52bb7ede1506aaaff0
EXPECTED_PAIR_ENTRY_SHA256=1eff3e32b3bb8cd159136264a61627dd9d218b80a9c1f11faa9ca62d888fcfa3
EXPECTED_FORMAL_SHA256=54ea0f435f52bd7145901349241fa7481dae840eabfcd2671cb1b3c555ae7fd8
EXPECTED_WRAPPER_SHA256=b55dc000504a3f3633db02d9cb3afc6a80b6b86d79c7cfc5a3a7f8a7985df1fd
EXPECTED_CONTRACT_SHA256=b0f15dff66e2e19f4baeb8df358b3a3eeeb9cb523c7871a4ecb23202754eb6e3
EXPECTED_SFT1_ALL_SHA256=b2bd1b834633f595a604a446578c101f2747d43330f4a7ad98e89f3a147c9e26

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing/non-regular %s: %s\n' "$label" "$path" >&2
    exit 3
  }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s SHA mismatch expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    exit 3
  }
}
atomic_status() {
  local state=$1 detail=$2
  STATUS_STATE="$state" STATUS_DETAIL="$detail" STATUS_PATH="$RUN_ROOT/chain_status.json" \
    STATUS_TRAIN="$TRAIN_OUT" STATUS_EVAL="$EVAL_RUN" STATUS_PID="$$" \
    env PYTHONPATH= "$PYTHON" - <<'PY'
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_PATH"])
record = {
    "schema_version": "qwen3-v26-saam-mixed180-train-eval-chain-v1",
    "state": os.environ["STATUS_STATE"],
    "detail": os.environ["STATUS_DETAIL"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "supervisor_pid": int(os.environ["STATUS_PID"]),
    "training_run": os.environ["STATUS_TRAIN"],
    "evaluation_run": os.environ["STATUS_EVAL"],
    "training_policy": "fresh SFT1; exact historical train180; K8; two passes; 12 updates; gradient diagnostics disabled",
    "evaluation_policy": "final only; BIRD-dev1534 greedy; paired against frozen fresh SFT1",
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(record, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY
}
gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le "$MAXIMUM_USED_MIB" ]]
}
training_complete() {
  [[ -f "$TRAIN_OUT/run_manifest.json" \
    && -f "$TRAIN_OUT/implementation_lock.json" \
    && -f "$TRAIN_OUT/training_precision.json" \
    && -f "$TRAIN_OUT/rollouts.jsonl" \
    && -f "$TRAIN_OUT/checkpoint-12/trainer_state.json" \
    && -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]
}
evaluation_complete() {
  [[ -f "$EVAL_RUN/status.json" && ! -L "$EVAL_RUN/status.json" ]] || return 1
  env PYTHONPATH= "$PYTHON" - "$EVAL_RUN/status.json" "$EVAL_RUN" <<'PY'
import json, sys
from pathlib import Path
status_path, run_dir = map(Path, sys.argv[1:])
status = json.loads(status_path.read_text(encoding="utf-8"))
if not (
    status.get("schema_version") == "qwen3-v26-formal-saam-candidate-only-eval-status-v2"
    and status.get("state") == "completed"
    and status.get("success") is True
    and status.get("run_dir") == str(run_dir.resolve())
    and (status.get("final") or {}).get("records") == 1534
):
    raise SystemExit(1)
PY
}

[[ -x "$PYTHON" && -d "$TRAIN_RUNTIME" && -d "$EVAL_RUNTIME" ]] || {
  printf 'missing Python or isolated runtime\n' >&2
  exit 3
}
[[ "$EVAL_GPU" =~ ^[0-9]+$ && "$EVAL_PORT" =~ ^[0-9]+$ ]] || {
  printf 'evaluation GPU and port must be non-negative integers\n' >&2
  exit 2
}
require_sha "$TRAIN_LAUNCHER" "$EXPECTED_TRAIN_LAUNCHER_SHA256" training_launcher
require_sha "$EVAL_ENTRY" "$EXPECTED_EVAL_ENTRY_SHA256" candidate_evaluator
require_sha "$PAIR_ENTRY" "$EXPECTED_PAIR_ENTRY_SHA256" paired_auditor
require_sha "$EVAL_RUNTIME/src/rl/evaluation/formal_v26_matched_eval.py" "$EXPECTED_FORMAL_SHA256" formal_evaluator
require_sha "$EVAL_RUNTIME/src/rl/evaluation/formal_v26_rollout_passk.py" "$EXPECTED_WRAPPER_SHA256" rollout_wrapper
require_sha "$EVAL_CONTRACT" "$EXPECTED_CONTRACT_SHA256" evaluation_contract
require_sha "$SFT1_ALL" "$EXPECTED_SFT1_ALL_SHA256" frozen_sft1_result
env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" -m py_compile \
  "$EVAL_ENTRY" "$PAIR_ENTRY" \
  "$EVAL_RUNTIME/src/rl/evaluation/formal_v26_matched_eval.py" \
  "$EVAL_RUNTIME/src/rl/evaluation/formal_v26_rollout_passk.py"
bash -n "$TRAIN_LAUNCHER"

if [[ "$MODE" == --plan ]]; then
  env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" "$EVAL_RUN" "$SFT1_ALL" <<'PY'
import json, sys
print(json.dumps({
    "schema_version": "qwen3-v26-saam-mixed180-train-eval-plan-v1",
    "status": "plan_only_no_training_or_evaluation_started",
    "stages": [
        "fresh SFT1 -> exact historical train180 -> 12 SAAM updates / 2880 rollouts",
        "strict manifest/checkpoint/rollout audit with gradient diagnostics disabled",
        "fresh final-only BIRD-dev1534 greedy evaluation",
        "exact example-index paired comparison against frozen fresh SFT1",
    ],
    "training_run": sys.argv[1],
    "evaluation_run": sys.argv[2],
    "sft1_result": sys.argv[3],
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/train_eval_chain.lock"
flock -n 9 || { printf 'SAAM mixed180 train/eval chain already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/train_eval_chain.log" 2>&1

owned_pgid=""
stop_owned() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_owned "$owned_pgid"
  [[ "$code" -eq 0 ]] || atomic_status failed "chain_exit=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

if training_complete; then
  atomic_status training_already_complete "reusing strictly complete run=$TRAIN_OUT"
elif [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  atomic_status training "resuming matched-budget SAAM mixed180 from $RESUME_FROM_CHECKPOINT"
  setsid env RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" bash "$TRAIN_LAUNCHER" &
  owned_pgid=$!
  wait "$owned_pgid"
  owned_pgid=""
elif [[ -e "$TRAIN_OUT" || -L "$TRAIN_OUT" ]]; then
  atomic_status blocked "partial training output exists; no automatic resume: $TRAIN_OUT"
  exit 3
else
  atomic_status training "launching matched-budget SAAM mixed180"
  setsid bash "$TRAIN_LAUNCHER" &
  owned_pgid=$!
  wait "$owned_pgid"
  owned_pgid=""
fi
training_complete || { atomic_status blocked "training artifacts incomplete"; exit 3; }

if evaluation_complete; then
  atomic_status evaluation_already_complete "reusing complete run=$EVAL_RUN"
elif [[ -e "$EVAL_RUN" || -L "$EVAL_RUN" ]]; then
  atomic_status blocked "partial evaluation exists; no automatic resume: $EVAL_RUN"
  exit 3
else
  atomic_status waiting_for_eval_gpu "gpu=$EVAL_GPU"
  started=$(date +%s)
  while ! gpu_idle "$EVAL_GPU"; do
    now=$(date +%s)
    (( now - started < GPU_WAIT_SECONDS )) || {
      atomic_status blocked "evaluation GPU wait timeout"
      exit 75
    }
    sleep "$POLL_SECONDS"
  done
  atomic_status evaluating "fresh final-only BIRD-dev1534 gpu=$EVAL_GPU"
  mkdir -p "$EVAL_ROOT"
  setsid env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" "$EVAL_ENTRY" \
    --contract "$EVAL_CONTRACT" --training-run "$TRAIN_OUT" \
    --run-dir "$EVAL_RUN" --expected-global-step 12 \
    --gpu-id "$EVAL_GPU" --port "$EVAL_PORT" \
    --maximum-used-mib "$MAXIMUM_USED_MIB" &
  owned_pgid=$!
  wait "$owned_pgid"
  owned_pgid=""
fi
evaluation_complete || { atomic_status blocked "evaluation artifacts incomplete"; exit 3; }

if [[ -e "$PAIR_OUTPUT" || -L "$PAIR_OUTPUT" ]]; then
  atomic_status blocked "paired output already exists; refusing overwrite: $PAIR_OUTPUT"
  exit 3
fi
atomic_status pairing "exact example-index candidate-vs-SFT1 comparison"
env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" "$PAIR_ENTRY" \
  --candidate "$EVAL_RUN/final/result/all.jsonl" --sft1 "$SFT1_ALL" \
  --output "$PAIR_OUTPUT" --expected-records 1534 \
  >"$EVAL_RUN/paired_vs_sft1.stdout.json"

atomic_status complete "training=12_updates evaluation=1534 paired=$PAIR_OUTPUT"
trap - EXIT INT TERM
