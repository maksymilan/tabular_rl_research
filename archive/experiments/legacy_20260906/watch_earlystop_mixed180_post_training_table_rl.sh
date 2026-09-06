#!/usr/bin/env bash
# Durable fail-closed handoff: completed training -> CPU audits -> fresh matched eval.
# This watcher never starts another training arm and never signals an unowned process.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---plan}
if [[ $# -gt 1 || ( "$MODE" != --plan && "$MODE" != --run ) ]]; then
  printf 'usage: %s [--plan|--run]\n' "$0" >&2
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_earlystop_mixed180_grpo_20260813}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_grpo_20260813}
TRAIN_OUT=$RUN_ROOT/train180_two_pass_seed20260812
TASKS=$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/train180.jsonl
COHORT_MANIFEST=$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/earlystop_mixed180_manifest.json
QUEUE_ROOT=$RUN_ROOT/post_training_queue
STATUS=$QUEUE_ROOT/status.json
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
AUDITOR=$RUNTIME/src/rl/diagnostics/audit_earlystop_mixed180_training.py
PAIRING=$RUNTIME/src/rl/diagnostics/summarize_earlystop_pass_pairing.py
EVAL_LAUNCHER=$RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_earlystop_mixed180_formal_matched_eval.sh
TRAIN_AUDIT=$QUEUE_ROOT/training_completion_audit.json
PAIRING_REPORT=$QUEUE_ROOT/pass1_vs_pass2_diagnostic.json
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
EVAL_PORT=${EVAL_PORT:-8087}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-172800}
STARTED_EPOCH_FILE=$QUEUE_ROOT/started_epoch
RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_formal/formal_sft1_vs_final12_20260813}

EXPECTED_AUDITOR_SHA256=2fea91f592f332f17cd825fd301fb30a045522b22cf54c6e6018354125c3e23e
EXPECTED_PAIRING_SHA256=1634e2d8e34a90ae81605a3f17b0b40e22a441b207ce081434d827b748d484f8
EXPECTED_EVAL_LAUNCHER_SHA256=f4ac9aae1305f25dcafe38223d319560b3f202ce948dfc5b5d5b34a93cf2a5ed

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
  STATUS_STATE="$state" STATUS_DETAIL="$detail" WATCHER_PID="$$" STATUS_EVAL_GPU="$EVAL_GPU_ID" \
    env PYTHONPATH= "$PYTHON" - "$STATUS" "$RUN_DIR" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path, run_dir = map(Path, sys.argv[1:])
record = {
    "schema_version": "earlystop-mixed180-post-training-queue-status-v1",
    "state": os.environ["STATUS_STATE"],
    "detail": os.environ["STATUS_DETAIL"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "watcher_pid": int(os.environ["WATCHER_PID"]),
    "evaluation_gpu": int(os.environ["STATUS_EVAL_GPU"]),
    "formal_run_dir": str(run_dir),
    "pid_policy": "watcher sends no signals; formal evaluator signals only process groups it created",
    "resume_policy": "completed run is idempotent; any partial formal run is rejected and never resumed",
}
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary_name, path)
finally:
    try: os.unlink(temporary_name)
    except FileNotFoundError: pass
PY
}
gpu_idle() {
  local gpu=$1 used pids
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}
training_artifacts_present() {
  [[ -f "$TRAIN_OUT/run_manifest.json" \
    && -f "$TRAIN_OUT/implementation_lock.json" \
    && -f "$TRAIN_OUT/training_precision.json" \
    && -f "$TRAIN_OUT/rollouts.jsonl" \
    && -f "$TRAIN_OUT/checkpoint-12/trainer_state.json" \
    && -f "$TRAIN_OUT/checkpoint-12/adapter_model.safetensors" \
    && -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]
}
completed_formal_run_decision() {
  local status=$RUN_DIR/status.json
  [[ -d "$RUN_DIR" && ! -L "$RUN_DIR" && -f "$status" && ! -L "$status" ]] || return 1
  env PYTHONPATH= "$PYTHON" - "$status" "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

status_path = Path(sys.argv[1])
expected_run_dir = str(Path(sys.argv[2]).resolve())
try:
    status = json.loads(status_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(status, dict):
    raise SystemExit(1)
valid = (
    status.get("schema_version") == "qwen3-v26-formal-matched-eval-status-v1"
    and status.get("state") == "completed"
    and status.get("success") is True
    and status.get("run_dir") == expected_run_dir
    and type(status.get("physical_gpu_id")) is int
    and status.get("physical_gpu_id") == 0
    and status.get("decision") in {"promote_final", "do_not_promote_final"}
)
if not valid:
    raise SystemExit(1)
print(status["decision"])
PY
}

[[ "$EVAL_GPU_ID" == 0 ]] || { printf 'this frozen queue permits only physical GPU0\n' >&2; exit 2; }
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { printf 'POLL_SECONDS must be positive\n' >&2; exit 2; }
[[ "$MAX_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { printf 'MAX_WAIT_SECONDS must be positive\n' >&2; exit 2; }
if [[ "$MODE" == --plan ]]; then
  PLAN_PYTHON=$PYTHON
  if [[ ! -x "$PLAN_PYTHON" ]]; then
    PLAN_PYTHON=$(command -v python3)
  fi
  env PYTHONPATH= "$PLAN_PYTHON" - "$TRAIN_OUT" "$TRAIN_AUDIT" "$PAIRING_REPORT" "$RUN_DIR" <<'PY'
import json, sys
train, audit, pairing, run = sys.argv[1:]
print(json.dumps({
    "schema_version": "earlystop-mixed180-post-training-queue-plan-v1",
    "status": "plan_only_no_mutation",
    "stages": [
        "wait for complete training artifacts",
        "strict CPU training audit",
        "CPU pass1-vs-pass2 diagnostic (never a branch gate)",
        "wait for physical GPU0 idle",
        "fresh SFT1 then final12 full-dev1534 matched evaluation",
    ],
    "training_run": train,
    "training_audit": audit,
    "pass_pairing": pairing,
    "formal_run": run,
    "evaluation_gpu": 0,
    "touches_gpu1": False,
    "starts_followup_training": False,
    "resume_policy": "completed run idempotent; partial formal output rejected",
}, indent=2))
PY
  exit 0
fi
mkdir -p "$QUEUE_ROOT"
exec 9>"$QUEUE_ROOT/watcher.lock"
flock -n 9 || { printf 'post-training watcher already running\n' >&2; exit 75; }
exec >>"$QUEUE_ROOT/watcher.log" 2>&1

require_sha "$AUDITOR" "$EXPECTED_AUDITOR_SHA256" training_auditor
require_sha "$PAIRING" "$EXPECTED_PAIRING_SHA256" pass_pairing
require_sha "$EVAL_LAUNCHER" "$EXPECTED_EVAL_LAUNCHER_SHA256" eval_launcher

if [[ -f "$STARTED_EPOCH_FILE" ]]; then
  read -r started <"$STARTED_EPOCH_FILE"
  [[ "$started" =~ ^[0-9]+$ ]] || {
    atomic_status blocked "invalid_persistent_started_epoch"
    exit 3
  }
else
  started=$(date +%s)
  temporary=$STARTED_EPOCH_FILE.next
  printf '%s\n' "$started" >"$temporary"
  mv "$temporary" "$STARTED_EPOCH_FILE"
fi
while ! training_artifacts_present; do
  now=$(date +%s)
  elapsed=$((now - started))
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    atomic_status blocked "training_completion_timeout elapsed=$elapsed"
    exit 75
  fi
  atomic_status waiting_for_training "elapsed=$elapsed"
  sleep "$POLL_SECONDS"
done

atomic_status auditing_training "run=$TRAIN_OUT"
if ! env PYTHONPATH="$RUNTIME" "$PYTHON" "$AUDITOR" \
  --run-dir "$TRAIN_OUT" --tasks "$TASKS" \
  --cohort-manifest "$COHORT_MANIFEST" --output "$TRAIN_AUDIT"
then
  atomic_status blocked "strict_training_audit_failed audit=$TRAIN_AUDIT"
  exit 3
fi

# This is deliberately not a branch condition.  It measures training-cohort
# fitting only and is always followed by the same fresh full-dev comparison.
atomic_status summarizing_pass_pairing "diagnostic_only=1"
env PYTHONPATH="$RUNTIME" "$PYTHON" "$PAIRING" \
  --rollouts "$TRAIN_OUT/rollouts.jsonl" --output "$PAIRING_REPORT"

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  if decision=$(completed_formal_run_decision); then
    atomic_status complete "evaluation_already_complete decision=$decision run=$RUN_DIR pairing=$PAIRING_REPORT"
    exit 0
  fi
  atomic_status blocked "formal_run_dir_exists_no_resume run=$RUN_DIR"
  exit 3
fi

while ! gpu_idle "$EVAL_GPU_ID"; do
  now=$(date +%s)
  elapsed=$((now - started))
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    atomic_status blocked "eval_gpu_wait_timeout gpu=$EVAL_GPU_ID elapsed=$elapsed"
    exit 75
  fi
  atomic_status waiting_for_eval_gpu "gpu=$EVAL_GPU_ID elapsed=$elapsed"
  sleep "$POLL_SECONDS"
done

atomic_status evaluating "fresh_sft1_then_final12 gpu=$EVAL_GPU_ID run=$RUN_DIR"
if ! EVAL_GPU_ID="$EVAL_GPU_ID" EVAL_PORT="$EVAL_PORT" RUN_DIR="$RUN_DIR" \
  PYTHON_BIN="$PYTHON" "$EVAL_LAUNCHER" --run
then
  atomic_status blocked "formal_matched_eval_failed_or_interrupted run=$RUN_DIR no_resume=1"
  exit 3
fi

if ! decision=$(completed_formal_run_decision); then
  atomic_status blocked "formal_matched_eval_returned_without_strict_completion run=$RUN_DIR"
  exit 3
fi
atomic_status complete "decision=$decision run=$RUN_DIR pairing=$PAIRING_REPORT"
