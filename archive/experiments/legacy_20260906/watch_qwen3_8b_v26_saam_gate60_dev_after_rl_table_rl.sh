#!/usr/bin/env bash
# Durable handoff: completed SAAM Gate60 RL -> strict artifact checks ->
# fresh full BIRD-dev pass@1 evaluation of the SAAM adapter only.
# SFT1 is the already-evaluated v26 anchor and is deliberately not rerun.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---plan}
if [[ $# -gt 1 || ( "$MODE" != --plan && "$MODE" != --run ) ]]; then
  printf 'usage: %s [--plan|--run]\n' "$0" >&2
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_saam_asymmetric_gate60_20260829}
TRAIN_OUT=${TRAIN_OUT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_asymmetric_gate60_20260829/run3}
EVAL_RUNTIME=${EVAL_RUNTIME:-$OUTPUT_ROOT/eval_runtime_qwen3_8b_v26_saam_gate60_20260829}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}

# These are captured from the currently running Gate60 launcher.  They may be
# overridden when the watcher is reused for a relaunch, but the command-line
# identity is checked before the watcher considers training finished.
TRAIN_LAUNCHER_PID=${TRAIN_LAUNCHER_PID:-2590847}
TRAINER_PID=${TRAINER_PID:-2591342}

EVAL_ROOT=${EVAL_ROOT:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_saam_asymmetric_gate60_20260829}
RUN_DIR=${RUN_DIR:-$EVAL_ROOT/saam_final_dev_20260829}
QUEUE_ROOT=${QUEUE_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_asymmetric_gate60_20260829/post_dev_eval}
STATUS=$QUEUE_ROOT/status.json
STARTED_EPOCH_FILE=$QUEUE_ROOT/started_epoch
WATCHER_LOG=$QUEUE_ROOT/watcher.log
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
EVAL_PORT=${EVAL_PORT:-8087}
MAXIMUM_USED_MIB=${MAXIMUM_USED_MIB:-512}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-172800}

EVAL_ENTRY=$EVAL_RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_saam_gate60_candidate_only_eval.py
EVAL_FORMAL=$EVAL_RUNTIME/src/rl/evaluation/formal_v26_matched_eval.py
EVAL_WRAPPER=$EVAL_RUNTIME/src/rl/evaluation/formal_v26_rollout_passk.py
EVAL_CONTRACT=$EVAL_RUNTIME/src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json
EVAL_ANALYZER=$EVAL_RUNTIME/src/rl/diagnostics/analyze_evaluation_results.py
EVAL_GATE=$EVAL_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_matched_gate.py

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

atomic_status() {
  local state=$1 detail=$2
  STATUS_STATE="$state" STATUS_DETAIL="$detail" WATCHER_PID="$$" \
    STATUS_GPU="$EVAL_GPU_ID" STATUS_RUN_DIR="$RUN_DIR" STATUS_TRAIN_OUT="$TRAIN_OUT" \
    env PYTHONPATH= "$PYTHON" - "$STATUS" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": "qwen3-v26-saam-post-training-dev-watch-v1",
    "state": os.environ["STATUS_STATE"],
    "detail": os.environ["STATUS_DETAIL"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "watcher_pid": int(os.environ["WATCHER_PID"]),
    "training_run": os.environ["STATUS_TRAIN_OUT"],
    "formal_run_dir": os.environ["STATUS_RUN_DIR"],
    "evaluation_gpu": int(os.environ["STATUS_GPU"]),
    "evaluation_scope": "SAAM final adapter only; frozen SFT1 dev anchor is not rerun",
    "pid_policy": "watcher never signals training; evaluator owns and cleans only its process groups",
    "resume_policy": "completed evaluation is idempotent; partial evaluation is rejected and never resumed",
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(record, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

process_matches() {
  local pid=$1 marker=$2 args
  is_uint "$pid" || return 1
  args=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ -n "$args" && "$args" == *"$marker"* ]]
}

training_live() {
  process_matches "$TRAIN_LAUNCHER_PID" "run_qwen3_8b_atomic_v26_saam_asymmetric_gate60_table_rl.sh" || \
    process_matches "$TRAINER_PID" "run_transition_grpo.py"
}

gpu_idle() {
  local gpu=$1 used pids
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le "$MAXIMUM_USED_MIB" ]]
}

training_artifacts_ready() {
  [[ -f "$TRAIN_OUT/run_manifest.json" \
    && -f "$TRAIN_OUT/implementation_lock.json" \
    && -f "$TRAIN_OUT/training_precision.json" \
    && -f "$TRAIN_OUT/rollouts.jsonl" \
    && -f "$TRAIN_OUT/checkpoint-4/trainer_state.json" \
    && -f "$TRAIN_OUT/checkpoint-4/adapter_model.safetensors" \
    && -f "$TRAIN_OUT/checkpoint-4/adapter_config.json" \
    && -f "$TRAIN_OUT/final/adapter_model.safetensors" \
    && -f "$TRAIN_OUT/final/adapter_config.json" ]]
}

validate_training_manifest() {
  env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
expected = {
    "schema_version": "table-agent-trl-transition-grpo-v2",
    "protocol_version": "version26",
    "protocol_hash": "4da19387399bd3a5",
    "student_prompt_sha256": "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
    "credit_assignment": "saam-asymmetric-error",
    "error_penalty": 1.0,
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean",
    "records": 60,
    "expected_records": 60,
    "group_size": 8,
    "prompts_per_update": 30,
    "optimizer_steps": 4,
    "transition_micro_batch_size": 1,
    "kl_beta": 0.0,
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise SystemExit(f"training manifest {field}: expected {value!r}, got {manifest.get(field)!r}")
if (manifest.get("gradient_conflict_logging") or {}).get("enabled") is not True:
    raise SystemExit("gradient conflict logging was not enabled")
state = json.loads((root / "checkpoint-4/trainer_state.json").read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) != 4:
    raise SystemExit(f"checkpoint-4 global_step is {state.get('global_step')!r}")
print(json.dumps({
    "status": "training_manifest_ok",
    "experiment_config_sha256": manifest.get("experiment_config_sha256"),
    "rollout_sha256": manifest.get("examples_json_sha256"),
}, ensure_ascii=False))
PY
}

static_eval_check() {
  [[ -x "$PYTHON" && -d "$EVAL_RUNTIME" && -f "$EVAL_ENTRY" && -f "$EVAL_FORMAL" \
    && -f "$EVAL_WRAPPER" && -f "$EVAL_CONTRACT" && -f "$EVAL_ANALYZER" && -f "$EVAL_GATE" ]]
  env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" -m py_compile \
    "$EVAL_ENTRY" "$EVAL_FORMAL" "$EVAL_WRAPPER" "$EVAL_ANALYZER" "$EVAL_GATE"
  env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" - <<'PY'
from rl.scenarios.evaluation import formal_v26_matched_eval as formal
formal.load_object(formal.PROJECT_ROOT / "src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json")
print("evaluation package import and contract load: ok")
PY
}

completed_eval() {
  [[ -f "$RUN_DIR/status.json" && ! -L "$RUN_DIR/status.json" ]] || return 1
  env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" - "$RUN_DIR/status.json" "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
status_path, run_dir = map(Path, sys.argv[1:])
try:
    status = json.loads(status_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
if not (
    status.get("schema_version") == "qwen3-v26-formal-candidate-only-eval-status-v1"
    and status.get("state") == "completed"
    and status.get("success") is True
    and status.get("run_dir") == str(run_dir.resolve())
):
    raise SystemExit(1)
final = status.get("final") or {}
print(f"correct={final.get('correct')} total={final.get('records')} legal={final.get('legal')}")
PY
}

for value in "$TRAIN_LAUNCHER_PID" "$TRAINER_PID" "$EVAL_GPU_ID" "$EVAL_PORT" "$MAXIMUM_USED_MIB"; do
  is_uint "$value" || { printf 'non-negative integer required: %s\n' "$value" >&2; exit 2; }
done
is_posint "$POLL_SECONDS" || { printf 'POLL_SECONDS must be positive\n' >&2; exit 2; }
is_posint "$MAX_WAIT_SECONDS" || { printf 'MAX_WAIT_SECONDS must be positive\n' >&2; exit 2; }

if [[ "$MODE" == --plan ]]; then
  env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" "$RUN_DIR" "$EVAL_GPU_ID" <<'PY'
import json, sys
train, run, gpu = sys.argv[1:]
print(json.dumps({
    "schema_version": "qwen3-v26-saam-post-training-dev-watch-plan-v1",
    "status": "plan_only_no_mutation",
    "stages": [
        "wait for the exact Gate60 launcher and trainer PIDs to exit",
        "validate the SAAM run manifest/checkpoint/precision/gradient-log artifacts",
        "wait for the selected GPU to be idle",
        "run fresh greedy BIRD-dev 1534 on the SAAM final adapter only",
    ],
    "training_run": train,
    "formal_run": run,
    "evaluation_gpu": int(gpu),
    "sft1_rerun": False,
    "resume_policy": "partial evaluation is rejected; no resume",
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

mkdir -p "$QUEUE_ROOT"
exec 9>"$QUEUE_ROOT/watcher.lock"
flock -n 9 || { printf 'post-training dev watcher already running\n' >&2; exit 75; }
exec >>"$WATCHER_LOG" 2>&1

if [[ -f "$STARTED_EPOCH_FILE" ]]; then
  read -r started <"$STARTED_EPOCH_FILE"
  is_uint "$started" || { atomic_status blocked "invalid_started_epoch"; exit 3; }
else
  started=$(date +%s)
  printf '%s\n' "$started" >"$STARTED_EPOCH_FILE"
fi

atomic_status waiting_for_training "launcher_pid=$TRAIN_LAUNCHER_PID trainer_pid=$TRAINER_PID"
while training_live; do
  now=$(date +%s)
  elapsed=$((now - started))
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    atomic_status blocked "training_wait_timeout elapsed=$elapsed"
    exit 75
  fi
  atomic_status waiting_for_training "elapsed=$elapsed launcher_pid=$TRAIN_LAUNCHER_PID trainer_pid=$TRAINER_PID"
  sleep "$POLL_SECONDS"
done

if ! training_artifacts_ready; then
  atomic_status blocked "training_processes_exited_but_artifacts_incomplete run=$TRAIN_OUT"
  exit 3
fi
atomic_status validating_training "run=$TRAIN_OUT"
if ! validate_training_manifest; then
  atomic_status blocked "training_manifest_validation_failed run=$TRAIN_OUT"
  exit 3
fi

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  if result=$(completed_eval); then
    atomic_status complete "evaluation_already_complete $result run=$RUN_DIR"
    exit 0
  fi
  atomic_status blocked "partial_eval_dir_exists_no_resume run=$RUN_DIR"
  exit 3
fi

atomic_status static_checking "evaluation_runtime=$EVAL_RUNTIME"
if ! static_eval_check; then
  atomic_status blocked "evaluation_static_check_failed runtime=$EVAL_RUNTIME"
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

atomic_status evaluating "fresh_saam_final_only_dev1534 gpu=$EVAL_GPU_ID run=$RUN_DIR"
if ! env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" "$EVAL_ENTRY" \
  --contract "$EVAL_CONTRACT" --training-run "$TRAIN_OUT" --run-dir "$RUN_DIR" \
  --gpu-id "$EVAL_GPU_ID" --port "$EVAL_PORT" --maximum-used-mib "$MAXIMUM_USED_MIB" \
  >"$QUEUE_ROOT/candidate_eval.stdout.log" 2>&1; then
  atomic_status blocked "candidate_dev_eval_failed_or_interrupted run=$RUN_DIR no_resume=1"
  exit 3
fi

if ! result=$(completed_eval); then
  atomic_status blocked "candidate_dev_eval_returned_without_completion run=$RUN_DIR"
  exit 3
fi
atomic_status complete "${result} run=$RUN_DIR"
