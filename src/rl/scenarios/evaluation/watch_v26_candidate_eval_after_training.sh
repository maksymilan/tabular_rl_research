#!/usr/bin/env bash
# Durable, fail-closed handoff from an already-running v26 training process to
# a final-only BIRD-dev1534 evaluation and an optional paired baseline report.
# The watcher never owns or signals the training process.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---plan}
if [[ $# -gt 1 || ( "$MODE" != --plan && "$MODE" != --run ) ]]; then
  printf 'usage: %s [--plan|--run]\n' "$0" >&2
  exit 2
fi

PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
EVAL_RUNTIME=${EVAL_RUNTIME:?set EVAL_RUNTIME to the immutable evaluator source root}
TRAIN_OUT=${TRAIN_OUT:?set TRAIN_OUT to the complete training-run directory}
TRAIN_STATUS_FILE=${TRAIN_STATUS_FILE:?set TRAIN_STATUS_FILE to the launcher status file}
EVAL_ROOT=${EVAL_ROOT:?set EVAL_ROOT to a new evaluation output root}
EVAL_RUN=${EVAL_RUN:-$EVAL_ROOT/final_dev1534}
QUEUE_ROOT=${QUEUE_ROOT:-$EVAL_ROOT/handoff}
STATUS_PATH=$QUEUE_ROOT/status.json
WATCHER_LOG=$QUEUE_ROOT/watcher.log
EVAL_STDOUT=$QUEUE_ROOT/evaluation.stdout.log

EVAL_ENTRY=${EVAL_ENTRY:-$EVAL_RUNTIME/src/rl/evaluation/runners/run_qwen3_8b_v26_saam_candidate_only_eval.py}
EVAL_CONTRACT=${EVAL_CONTRACT:-$EVAL_RUNTIME/archive/config/evaluation_contracts/20260906/qwen3_8b_v26_vanilla_formal_matched_contract.json}
ANALYZER=${ANALYZER:-$EVAL_RUNTIME/src/rl/scenarios/diagnostics/analyze_evaluation_results.py}

TRAIN_LAUNCHER_PID=${TRAIN_LAUNCHER_PID:-0}
TRAINER_PID=${TRAINER_PID:-0}
TRAIN_LAUNCHER_MARKER=${TRAIN_LAUNCHER_MARKER:-run_qwen3_8b_atomic_v26}
TRAINER_MARKER=${TRAINER_MARKER:-run_transition_grpo.py}
EXPECTED_GLOBAL_STEP=${EXPECTED_GLOBAL_STEP:?set EXPECTED_GLOBAL_STEP}
EXPECTED_RECORDS=${EXPECTED_RECORDS:?set EXPECTED_RECORDS}
EXPECTED_ROLLOUTS=${EXPECTED_ROLLOUTS:?set EXPECTED_ROLLOUTS}
EXPECTED_CREDIT_ASSIGNMENT=${EXPECTED_CREDIT_ASSIGNMENT:?set EXPECTED_CREDIT_ASSIGNMENT}
EXPECTED_SPAN_BALANCE_ALPHA=${EXPECTED_SPAN_BALANCE_ALPHA:-}
EXPECTED_TRAINER_SHA256=${EXPECTED_TRAINER_SHA256:-}
REQUIRE_NONZERO_GRADIENT_STEPS=${REQUIRE_NONZERO_GRADIENT_STEPS:-false}

EVAL_GPU_ID=${EVAL_GPU_ID:-0}
EVAL_PORT=${EVAL_PORT:-8087}
MAXIMUM_USED_MIB=${MAXIMUM_USED_MIB:-512}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-172800}

BASELINE_ALL=${BASELINE_ALL:-}
BASELINE_IDENTITY=${BASELINE_IDENTITY:-}
BASELINE_ALL_SHA256=${BASELINE_ALL_SHA256:-}
BASELINE_IDENTITY_SHA256=${BASELINE_IDENTITY_SHA256:-}
CANDIDATE_LABEL=${CANDIDATE_LABEL:-candidate}
BASELINE_LABEL=${BASELINE_LABEL:-baseline}
ANALYSIS_OUTPUT=${ANALYSIS_OUTPUT:-$EVAL_RUN/paired_analysis.json}

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

die() {
  printf 'candidate eval handoff: %s\n' "$*" >&2
  exit 3
}

atomic_status() {
  local state=$1 detail=$2
  STATUS_STATE="$state" STATUS_DETAIL="$detail" WATCHER_PID="$$" \
    STATUS_TRAIN_OUT="$TRAIN_OUT" STATUS_EVAL_RUN="$EVAL_RUN" \
    STATUS_EVAL_GPU="$EVAL_GPU_ID" \
    env PYTHONPATH= "$PYTHON" - "$STATUS_PATH" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": "v26-candidate-post-training-eval-handoff-v1",
    "state": os.environ["STATUS_STATE"],
    "detail": os.environ["STATUS_DETAIL"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "watcher_pid": int(os.environ["WATCHER_PID"]),
    "training_run": os.environ["STATUS_TRAIN_OUT"],
    "evaluation_run": os.environ["STATUS_EVAL_RUN"],
    "evaluation_gpu": int(os.environ["STATUS_EVAL_GPU"]),
    "training_ownership": "watcher observes but never signals training",
    "evaluation_ownership": "watcher cleans only the evaluator process group it starts",
    "resume_policy": "completed evaluation is idempotent; partial evaluation is fail-closed",
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

process_matches() {
  local pid=$1 marker=$2 args
  is_uint "$pid" || return 1
  [[ "$pid" != 0 ]] || return 1
  args=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ -n "$args" && "$args" == *"$marker"* ]]
}

training_live() {
  process_matches "$TRAIN_LAUNCHER_PID" "$TRAIN_LAUNCHER_MARKER" || \
    process_matches "$TRAINER_PID" "$TRAINER_MARKER"
}

launcher_completed() {
  [[ -f "$TRAIN_STATUS_FILE" && ! -L "$TRAIN_STATUS_FILE" ]] || return 1
  local timestamp state detail
  IFS=$'\t' read -r timestamp state detail <"$TRAIN_STATUS_FILE"
  [[ -n "$timestamp" && "$state" == complete ]]
}

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le "$MAXIMUM_USED_MIB" ]]
}

training_artifacts_ready() {
  local checkpoint=$TRAIN_OUT/checkpoint-$EXPECTED_GLOBAL_STEP
  [[ -f "$TRAIN_OUT/run_manifest.json" \
    && -f "$TRAIN_OUT/implementation_lock.json" \
    && -f "$TRAIN_OUT/training_precision.json" \
    && -f "$TRAIN_OUT/rollouts.jsonl" \
    && -d "$TRAIN_OUT/implementation_source_snapshot" \
    && -f "$checkpoint/trainer_state.json" \
    && -f "$checkpoint/adapter_model.safetensors" \
    && -f "$checkpoint/adapter_config.json" \
    && -f "$TRAIN_OUT/final/adapter_model.safetensors" \
    && -f "$TRAIN_OUT/final/adapter_config.json" ]]
}

validate_training() {
  EXPECTED_GLOBAL_STEP="$EXPECTED_GLOBAL_STEP" EXPECTED_RECORDS="$EXPECTED_RECORDS" \
    EXPECTED_ROLLOUTS="$EXPECTED_ROLLOUTS" \
    EXPECTED_CREDIT_ASSIGNMENT="$EXPECTED_CREDIT_ASSIGNMENT" \
    EXPECTED_SPAN_BALANCE_ALPHA="$EXPECTED_SPAN_BALANCE_ALPHA" \
    EXPECTED_TRAINER_SHA256="$EXPECTED_TRAINER_SHA256" \
    REQUIRE_NONZERO_GRADIENT_STEPS="$REQUIRE_NONZERO_GRADIENT_STEPS" \
    env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" <<'PY'
import hashlib, json, math, os, sys
from pathlib import Path

root = Path(sys.argv[1])
step = int(os.environ["EXPECTED_GLOBAL_STEP"])
records = int(os.environ["EXPECTED_RECORDS"])
rollouts = int(os.environ["EXPECTED_ROLLOUTS"])
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
expected = {
    "schema_version": "table-agent-trl-transition-grpo-v2",
    "protocol_version": "version26",
    "protocol_hash": "4da19387399bd3a5",
    "student_prompt_sha256": "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean",
    "records": records,
    "expected_records": records,
    "optimizer_steps": step,
    "group_size": 8,
    "prompts_per_update": 30,
    "credit_assignment": os.environ["EXPECTED_CREDIT_ASSIGNMENT"],
    "kl_beta": 0.0,
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise SystemExit(f"training manifest {field}: expected {value!r}, got {manifest.get(field)!r}")
span = os.environ.get("EXPECTED_SPAN_BALANCE_ALPHA", "")
if span:
    observed = manifest.get("span_balance_alpha")
    if observed is None or not math.isclose(float(observed), float(span), rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"span_balance_alpha: expected {span}, got {observed!r}")
if (manifest.get("gradient_conflict_logging") or {}).get("enabled") is not False:
    raise SystemExit("gradient conflict logging must be disabled")

checkpoint = root / f"checkpoint-{step}"
state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) != step:
    raise SystemExit(f"checkpoint global_step must be {step}, got {state.get('global_step')!r}")
observed_rollouts = sum(1 for line in (root / "rollouts.jsonl").open(encoding="utf-8") if line.strip())
if observed_rollouts != rollouts:
    raise SystemExit(f"rollout count: expected {rollouts}, got {observed_rollouts}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint_adapter = checkpoint / "adapter_model.safetensors"
final_adapter = root / "final/adapter_model.safetensors"
if sha256(checkpoint_adapter) != sha256(final_adapter):
    raise SystemExit("final adapter does not match the declared final checkpoint")

trainer_relative = "src/rl/frameworks/trl/transition_grpo.py"
trainer_expected = os.environ.get("EXPECTED_TRAINER_SHA256", "")
if trainer_expected:
    recorded = (manifest.get("implementation_source_sha256") or {}).get(trainer_relative)
    snapshot = root / "implementation_source_snapshot" / trainer_relative
    if recorded != trainer_expected or sha256(snapshot) != trainer_expected:
        raise SystemExit(
            f"trainer implementation mismatch recorded={recorded!r} expected={trainer_expected}"
        )

if os.environ.get("REQUIRE_NONZERO_GRADIENT_STEPS", "false").lower() == "true":
    gradients = {}
    for item in state.get("log_history") or []:
        item_step = item.get("step")
        if isinstance(item_step, int) and "grad_norm" in item:
            gradients[item_step] = float(item["grad_norm"])
    missing = [index for index in range(1, step + 1) if index not in gradients]
    dead = [index for index, value in gradients.items() if index <= step and (not math.isfinite(value) or value <= 0.0)]
    if missing or dead:
        raise SystemExit(f"nonzero-gradient gate failed missing={missing} dead={dead}")

print(json.dumps({
    "status": "training_validated",
    "global_step": step,
    "rollouts": observed_rollouts,
    "adapter_sha256": sha256(final_adapter),
}, sort_keys=True))
PY
}

static_eval_check() {
  [[ -x "$PYTHON" && -d "$EVAL_RUNTIME" && -f "$EVAL_ENTRY" \
    && -f "$EVAL_CONTRACT" && -f "$ANALYZER" ]] || return 1
  env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" - "$EVAL_ENTRY" "$EVAL_CONTRACT" "$ANALYZER" <<'PY'
import sys
from pathlib import Path

entry, contract, analyzer = map(Path, sys.argv[1:])
for path in (entry, analyzer):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
from rl.evaluation.runners import formal_v26_matched_eval as formal
payload = formal.load_object(contract)
runtime = Path(payload["host_paths"]["runtime"])
formal.verify_runtime(payload, runtime)
print(runtime)
PY
}

completed_eval() {
  [[ -f "$EVAL_RUN/status.json" && ! -L "$EVAL_RUN/status.json" ]] || return 1
  env PYTHONPATH= "$PYTHON" - "$EVAL_RUN/status.json" "$EVAL_RUN" <<'PY'
import json, sys
from pathlib import Path

status_path, run_dir = map(Path, sys.argv[1:])
status = json.loads(status_path.read_text(encoding="utf-8"))
final = status.get("final") or {}
valid = (
    status.get("schema_version") == "qwen3-v26-formal-saam-candidate-only-eval-status-v2"
    and status.get("state") == "completed"
    and status.get("success") is True
    and status.get("run_dir") == str(run_dir.resolve())
    and final.get("records") == 1534
    and (run_dir / "final/result/all.jsonl").is_file()
    and (run_dir / "final/result/summary.json").is_file()
    and (run_dir / "final/evaluation_identity.json").is_file()
)
if not valid:
    raise SystemExit(1)
print(f"correct={final.get('correct')} total={final.get('records')} legal={final.get('legal')}")
PY
}

validate_baseline() {
  [[ -n "$BASELINE_ALL" ]] || return 0
  [[ -f "$BASELINE_ALL" && -f "$BASELINE_IDENTITY" ]] || return 1
  if [[ -n "$BASELINE_ALL_SHA256" && "$(sha256_file "$BASELINE_ALL")" != "$BASELINE_ALL_SHA256" ]]; then
    return 1
  fi
  if [[ -n "$BASELINE_IDENTITY_SHA256" && "$(sha256_file "$BASELINE_IDENTITY")" != "$BASELINE_IDENTITY_SHA256" ]]; then
    return 1
  fi
}

run_analysis() {
  [[ -n "$BASELINE_ALL" ]] || return 0
  [[ ! -e "$ANALYSIS_OUTPUT" && ! -L "$ANALYSIS_OUTPUT" ]] || return 1
  local examples
  examples=$(env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" - "$EVAL_CONTRACT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["host_paths"]["source_input"])
PY
)
  local match_fields=(
    base_model base_model_revision base_model_identity_sha256
    dataset input_sha256 task_identity_sha256_without_db_path
    protocol_version protocol_hash runtime runtime_sha256
    serving concurrency decode agent tool_execution_timeout_seconds
  )
  local args=(
    --examples "$examples"
    --arm "$CANDIDATE_LABEL=$EVAL_RUN/final/result/all.jsonl"
    --arm "$BASELINE_LABEL=$BASELINE_ALL"
    --compare "$CANDIDATE_LABEL:$BASELINE_LABEL"
    --expected-count 1534
    --protocol-version version26 --protocol-hash 4da19387399bd3a5
    --temperature 0 --top-p 1 --denotation-comparison bird-set
    --identity "$CANDIDATE_LABEL=$EVAL_RUN/final/evaluation_identity.json"
    --identity "$BASELINE_LABEL=$BASELINE_IDENTITY"
    --require-identities --require-distinct-adapters
    --output "$ANALYSIS_OUTPUT"
  )
  local field
  for field in "${match_fields[@]}"; do args+=(--match-identity-field "$field"); done
  env PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" "$ANALYZER" "${args[@]}"
}

for value in "$TRAIN_LAUNCHER_PID" "$TRAINER_PID" "$EXPECTED_GLOBAL_STEP" \
  "$EXPECTED_RECORDS" "$EXPECTED_ROLLOUTS" "$EVAL_GPU_ID" "$EVAL_PORT" \
  "$MAXIMUM_USED_MIB"; do
  is_uint "$value" || die "non-negative integer required: $value"
done
is_posint "$POLL_SECONDS" || die "POLL_SECONDS must be positive"
is_posint "$MAX_WAIT_SECONDS" || die "MAX_WAIT_SECONDS must be positive"
[[ "$CANDIDATE_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid candidate label"
[[ "$BASELINE_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid baseline label"

if [[ "$MODE" == --plan ]]; then
  static_eval_check >/dev/null || die "evaluation static check failed"
  validate_baseline || die "baseline identity/hash check failed"
  env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" "$EVAL_RUN" "$EVAL_GPU_ID" "$ANALYSIS_OUTPUT" <<'PY'
import json, sys
train, evaluation, gpu, analysis = sys.argv[1:]
print(json.dumps({
    "schema_version": "v26-candidate-post-training-eval-handoff-plan-v1",
    "status": "plan_only_no_process_started",
    "stages": [
        "wait for exact training launcher and trainer PIDs to exit",
        "require launcher complete status and strict checkpoint/final/rollout identity",
        "require every optimizer step to have nonzero gradient when configured",
        "wait for the selected GPU and port to be free",
        "verify frozen source hashes while ignoring generated interpreter/tool caches",
        "run final-only greedy BIRD-dev1534 evaluation",
        "validate complete status/result/identity and generate the paired baseline report",
    ],
    "training_run": train,
    "evaluation_run": evaluation,
    "evaluation_gpu": int(gpu),
    "analysis_output": analysis,
    "training_ownership": "observe only; never signal",
    "partial_evaluation_policy": "fail closed; never reuse partial output",
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

mkdir -p "$QUEUE_ROOT"
exec 9>"$QUEUE_ROOT/watcher.lock"
flock -n 9 || die "another watcher already owns this handoff"
exec >>"$WATCHER_LOG" 2>&1

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
  if [[ "$code" -ne 0 ]]; then atomic_status failed "handoff_exit=$code"; fi
  exit "$code"
}
trap cleanup EXIT INT TERM

started=$(date +%s)
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

if ! launcher_completed; then
  atomic_status blocked "training launcher did not record complete status: $TRAIN_STATUS_FILE"
  exit 3
fi
if ! training_artifacts_ready; then
  atomic_status blocked "training complete status exists but final artifacts are incomplete"
  exit 3
fi
atomic_status validating_training "run=$TRAIN_OUT"
if ! validation=$(validate_training); then
  atomic_status blocked "strict training validation failed"
  exit 3
fi

if [[ -e "$EVAL_RUN" || -L "$EVAL_RUN" ]]; then
  if result=$(completed_eval); then
    atomic_status evaluation_already_complete "$result run=$EVAL_RUN"
  else
    atomic_status blocked "partial evaluation directory exists; no resume: $EVAL_RUN"
    exit 3
  fi
else
  atomic_status static_checking "evaluation_runtime=$EVAL_RUNTIME"
  if ! static_eval_check >/dev/null; then
    atomic_status blocked "evaluation static check failed"
    exit 3
  fi
  if ! validate_baseline; then
    atomic_status blocked "baseline identity/hash check failed"
    exit 3
  fi
  while ! gpu_idle "$EVAL_GPU_ID"; do
    now=$(date +%s)
    elapsed=$((now - started))
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      atomic_status blocked "evaluation GPU wait timeout gpu=$EVAL_GPU_ID elapsed=$elapsed"
      exit 75
    fi
    atomic_status waiting_for_eval_gpu "gpu=$EVAL_GPU_ID elapsed=$elapsed"
    sleep "$POLL_SECONDS"
  done

  mkdir -p "$EVAL_ROOT"
  atomic_status evaluating "final-only BIRD-dev1534 gpu=$EVAL_GPU_ID port=$EVAL_PORT $validation"
  setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$EVAL_RUNTIME" \
    "$PYTHON" "$EVAL_ENTRY" \
      --contract "$EVAL_CONTRACT" --training-run "$TRAIN_OUT" \
      --run-dir "$EVAL_RUN" --expected-global-step "$EXPECTED_GLOBAL_STEP" \
      --gpu-id "$EVAL_GPU_ID" --port "$EVAL_PORT" \
      --maximum-used-mib "$MAXIMUM_USED_MIB" \
      >"$EVAL_STDOUT" 2>&1 &
  owned_pgid=$!
  if ! wait "$owned_pgid"; then
    owned_pgid=""
    atomic_status blocked "candidate evaluation failed; see $EVAL_STDOUT"
    exit 3
  fi
  owned_pgid=""
  if ! result=$(completed_eval); then
    atomic_status blocked "evaluator exited without a strictly complete result"
    exit 3
  fi
fi

if [[ -n "$BASELINE_ALL" ]]; then
  atomic_status analyzing "paired $CANDIDATE_LABEL vs $BASELINE_LABEL"
  if ! run_analysis >"$QUEUE_ROOT/analysis.stdout.log" 2>&1; then
    atomic_status blocked "paired analysis failed"
    exit 3
  fi
fi
atomic_status complete "$result evaluation=$EVAL_RUN analysis=$ANALYSIS_OUTPUT"
trap - EXIT INT TERM
