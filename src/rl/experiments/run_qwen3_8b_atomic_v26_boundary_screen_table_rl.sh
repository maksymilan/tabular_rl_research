#!/usr/bin/env bash
# Fail-closed boundary screening for the frozen Qwen3-8B atomic-v26 SFT1 policy.
#
# Nothing runs by default.  The explicit S1 lifecycle is:
#   prepare -> worker (one invocation per shard) -> reuse-first32 -> finalize
#
# Each worker owns only the process group that it starts.  This launcher never
# searches for, reuses, or signals another experiment's process.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|prepare|worker|reuse-first32|finalize]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|prepare|worker|reuse-first32|finalize) ;;
  *) printf 'usage: %s [dry-run|prepare|worker|reuse-first32|finalize]\n' "$0" >&2; exit 2 ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_vanilla_grpo_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
TASKS_JSONL=${TASKS_JSONL:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl}
TASKS_MANIFEST=${TASKS_MANIFEST:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}

SCREEN_STAGE=${SCREEN_STAGE:-S1}
ENABLE_S2=${ENABLE_S2:-0}
SCREEN_SHARDS=${SCREEN_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
SCREEN_GPU0=${SCREEN_GPU0:-0}
SCREEN_GPU1=${SCREEN_GPU1:-1}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_20260812}
POOL_OUT=${POOL_OUT:-$RUN_ROOT/pool_current600_s1}
FIRST32_SOURCE=${FIRST32_SOURCE:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_vanilla_grpo_20260812/probe_first32_k8_seed20260812}

# The selector/auditor are deliberately external and hash-pinned.  An absent or
# changed admission/selection policy can never silently choose a training cohort.
BOUNDARY_AUDITOR=${BOUNDARY_AUDITOR:-$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_boundary_screen.py}
BOUNDARY_SELECTOR=${BOUNDARY_SELECTOR:-$TRAIN_RUNTIME/src/rl/select_policy_boundary_grpo_tasks.py}
BOUNDARY_AUDIT_DIR=${BOUNDARY_AUDIT_DIR:-$RUN_ROOT/boundary_selection}

ASSIGNMENTS_DIR=$RUN_ROOT/assignments
WORKERS_DIR=$RUN_ROOT/workers
STATUS_DIR=$RUN_ROOT/status
LOG_DIR=$RUN_ROOT/logs
LOCK_DIR=$RUN_ROOT/locks
PLAN=$RUN_ROOT/screen_plan.json
REUSE_MANIFEST=$RUN_ROOT/first32_reuse_manifest.json
FINALIZE_STATUS=$STATUS_DIR/finalize.status

EXPECTED_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e
EXPECTED_TASKS_MANIFEST_SHA256=9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6
EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191
EXPECTED_ROLLOUT_SHA256=87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f
EXPECTED_SCORING_SHA256=a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5
EXPECTED_TASK_LOADER_SHA256=d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b
EXPECTED_TOOL_ENV_V26_SHA256=c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316
EXPECTED_BOUNDARY_AUDITOR_SHA256=b983bd5d8083de49328fb25b98fba6105b2ffc773faceb7a5d0c8eea7c9181e6
EXPECTED_BOUNDARY_SELECTOR_SHA256=2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" ]] || die "missing $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}
write_status() {
  local path=$1 state=$2 detail=$3 temporary=${1}.next
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$path"
}

validate_stage() {
  if [[ "$SCREEN_STAGE" == S2 ]]; then
    [[ "$ENABLE_S2" == 1 ]] \
      || die 'S2 is reserved and requires explicit ENABLE_S2=1 plus separately frozen inputs'
    die 'S2 inputs are not frozen; this S1 launcher cannot infer or auto-enable S2'
  fi
  [[ "$SCREEN_STAGE" == S1 ]] || die "unsupported SCREEN_STAGE=$SCREEN_STAGE"
}

validate_shape() {
  [[ "$SCREEN_SHARDS" =~ ^[12]$ ]] || die 'SCREEN_SHARDS must be exactly 1 or 2'
  [[ "$SHARD_INDEX" =~ ^[01]$ ]] || die 'SHARD_INDEX must be 0 or 1'
  (( SHARD_INDEX < SCREEN_SHARDS )) \
    || die "SHARD_INDEX=$SHARD_INDEX is outside SCREEN_SHARDS=$SCREEN_SHARDS"
  [[ "$SCREEN_GPU0" =~ ^[0-9]+$ && "$SCREEN_GPU1" =~ ^[0-9]+$ ]] \
    || die 'SCREEN_GPU0 and SCREEN_GPU1 must be explicit nonnegative indices'
  [[ "$SCREEN_GPU0" != "$SCREEN_GPU1" || "$SCREEN_SHARDS" == 1 ]] \
    || die 'two shards require distinct GPU indices'
  case "$POOL_OUT" in "$RUN_ROOT"/*) ;; *) die "POOL_OUT is outside RUN_ROOT: $POOL_OUT" ;; esac
  case "$BOUNDARY_AUDIT_DIR" in "$RUN_ROOT"/*) ;; *) die "BOUNDARY_AUDIT_DIR is outside RUN_ROOT: $BOUNDARY_AUDIT_DIR" ;; esac
  [[ ! -L "$RUN_ROOT" && ! -L "$POOL_OUT" ]] || die 'run and pool roots may not be symlinks'
  local owned
  for owned in "$ASSIGNMENTS_DIR" "$WORKERS_DIR" "$STATUS_DIR" "$LOG_DIR" \
    "$LOCK_DIR" "$POOL_OUT/groups" "$BOUNDARY_AUDIT_DIR"; do
    [[ ! -L "$owned" ]] || die "owned output path may not be a symlink: $owned"
  done
}

validate_static_inputs() {
  validate_stage
  validate_shape
  [[ -x "$PYTHON_BIN" ]] || die "missing Python runtime: $PYTHON_BIN"
  [[ -d "$MODEL_PATH" ]] || die "missing model: $MODEL_PATH"
  [[ -d "$SFT1_ADAPTER" ]] || die "missing SFT1 adapter: $SFT1_ADAPTER"
  [[ -d "$PROTOCOL_RUNTIME/src/eval" && -d "$PROTOCOL_RUNTIME/src/sft" && -d "$PROTOCOL_RUNTIME/src/harness" ]] \
    || die "incomplete frozen protocol runtime: $PROTOCOL_RUNTIME"
  for forbidden in src/eval src/sft src/harness src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden" ]] \
      || die "training overlay shadows frozen protocol source: $TRAIN_RUNTIME/$forbidden"
  done
  require_sha "$TASKS_JSONL" "$EXPECTED_TASKS_SHA256" current600_tasks
  require_sha "$TASKS_MANIFEST" "$EXPECTED_TASKS_MANIFEST_SHA256" current600_manifest
  require_sha "$MODEL_PATH/model-00001-of-00005.safetensors" 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f model_shard_1
  require_sha "$MODEL_PATH/model-00002-of-00005.safetensors" 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 model_shard_2
  require_sha "$MODEL_PATH/model-00003-of-00005.safetensors" c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 model_shard_3
  require_sha "$MODEL_PATH/model-00004-of-00005.safetensors" b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a model_shard_4
  require_sha "$MODEL_PATH/model-00005-of-00005.safetensors" 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff model_shard_5
  require_sha "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
  require_sha "$SFT1_ADAPTER/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256" sft1_adapter_config
  require_sha "$SFT1_ADAPTER/trainer_state.json" "$EXPECTED_SFT1_STATE_SHA256" sft1_trainer_state
  require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" "$EXPECTED_GENERATOR_SHA256" fixed_pool_generator
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" "$EXPECTED_ROLLOUT_SHA256" rollout_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" "$EXPECTED_SCORING_SHA256" rollout_scoring
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" "$EXPECTED_TASK_LOADER_SHA256" task_loader
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" "$EXPECTED_TOOL_ENV_V26_SHA256" tool_environment_v26

  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$TASKS_MANIFEST" \
    "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import hashlib, importlib, json, sys
from pathlib import Path

tasks_path, manifest_path, runtime = map(lambda value: Path(value).resolve(), sys.argv[1:4])
expected_tree = sys.argv[4]
expected_version, expected_protocol, expected_prompt = sys.argv[5:8]
rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
assert len(rows) == 600
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert "None" not in ids and len(set(ids)) == 600
cohort = json.loads(manifest_path.read_text())
assert cohort["schema_version"] == "bird-train-vanilla-grpo-cohort-v1"
assert cohort["status"] == "frozen_training_cohort"
assert cohort["all_acceptance_gates_passed"] is True
assert cohort["output"]["records"] == 600
assert cohort["output"]["task_ids_in_frozen_order"] == ids
digest = hashlib.sha256()
files = []
for relative in ("src/eval", "src/sft", "src/harness"):
    files.extend(
        path for path in (runtime / relative).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
for path in sorted(files, key=lambda item: item.relative_to(runtime).as_posix()):
    relative = path.relative_to(runtime).as_posix()
    digest.update(relative.encode()); digest.update(b"\0")
    digest.update(path.read_bytes()); digest.update(b"\0")
assert digest.hexdigest() == expected_tree
sys.path[:0] = [str(runtime / "src/sft")]
protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(
    context_mode="rolling-legal-history", compact=False
)
assert protocol.PROTOCOL_VERSION == expected_version
assert protocol.protocol_hash(prompt) == expected_protocol
assert hashlib.sha256(prompt.encode()).hexdigest() == expected_prompt
PY
}

dry_run() {
  validate_stage
  validate_shape
  printf '%s\n' \
    'boundary-screen dry-run (no files written, no GPU inspected)' \
    "stage=$SCREEN_STAGE tasks=current600 first32=reuse remaining=568" \
    "shards=$SCREEN_SHARDS shard0_gpu=$SCREEN_GPU0 shard1_gpu=$SCREEN_GPU1" \
    "run_root=$RUN_ROOT" \
    'lifecycle=prepare -> worker[0..shards-1] -> reuse-first32 -> finalize' \
    'S2=reserved; never automatically enabled'
}

prepare() {
  validate_static_inputs
  mkdir -p "$ASSIGNMENTS_DIR" "$WORKERS_DIR" "$STATUS_DIR" "$LOG_DIR" "$LOCK_DIR" "$POOL_OUT/groups"
  exec 9>"$LOCK_DIR/control.lock"
  flock -n 9 || die "another control operation owns $LOCK_DIR/control.lock"
  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$RUN_ROOT" "$SCREEN_SHARDS" \
    "$EXPECTED_TASKS_SHA256" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

tasks_path, root = map(Path, sys.argv[1:3])
shards = int(sys.argv[3])
tasks_sha, runtime_sha = sys.argv[4:6]
rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert len(ids) == 600 and len(set(ids)) == 600
assignment_dir = root / "assignments"
assignments = []
for shard in range(shards):
    assigned = [task_id for offset, task_id in enumerate(ids[32:]) if offset % shards == shard]
    payload = "".join(f"{task_id}\n" for task_id in assigned).encode()
    path = assignment_dir / f"remaining568.shard-{shard:02d}-of-{shards:02d}.txt"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing changed assignment: {path}")
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_bytes(payload); os.replace(temporary, path)
    assignments.append({
        "shard_index": shard,
        "records": len(assigned),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first_task_id": assigned[0],
        "last_task_id": assigned[-1],
    })
expected_assignment_names = {
    Path(entry["path"]).name for entry in assignments
}
observed_assignment_names = {path.name for path in assignment_dir.iterdir()}
assert observed_assignment_names == expected_assignment_names, (
    observed_assignment_names, expected_assignment_names
)
plan = {
    "schema_version": "vanilla-grpo-boundary-screen-plan-v1",
    "status": "prepared",
    "stage": "S1",
    "tasks_path": str(tasks_path.resolve()),
    "tasks_sha256": tasks_sha,
    "tasks": 600,
    "reused_prefix_tasks": 32,
    "generated_remaining_tasks": 568,
    "group_size": 8,
    "temperature": 0.8,
    "top_p": 1.0,
    "max_steps": 30,
    "max_new_tokens": 2048,
    "max_context_tokens": 16384,
    "history_turns": 4,
    "enable_thinking": True,
    "seed": 20260812,
    "screen_shards": shards,
    "assignments": assignments,
    "protocol_runtime_content_tree_sha256": runtime_sha,
    "s2": {"status": "reserved_not_enabled", "automatic_launch": False},
}
encoded = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
path = root / "screen_plan.json"
if path.exists() and path.read_bytes() != encoded:
    raise RuntimeError(f"refusing changed screen plan: {path}")
temporary = path.with_suffix(".json.next")
temporary.write_bytes(encoded); os.replace(temporary, path)
PY
  write_status "$STATUS_DIR/prepare.status" complete "plan=$PLAN shards=$SCREEN_SHARDS remaining=568"
}

verify_plan_and_assignment() {
  local assignment=$1
  [[ -f "$PLAN" ]] || die "missing prepared plan: $PLAN"
  [[ -f "$assignment" ]] || die "missing shard assignment: $assignment"
  env PYTHONPATH= "$PYTHON_BIN" - "$PLAN" "$assignment" "$SCREEN_SHARDS" "$SHARD_INDEX" \
    "$EXPECTED_TASKS_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
assignment = Path(sys.argv[2]).resolve()
shards, index = map(int, sys.argv[3:5])
assert plan["schema_version"] == "vanilla-grpo-boundary-screen-plan-v1"
assert plan["stage"] == "S1" and plan["tasks"] == 600
assert plan["reused_prefix_tasks"] == 32 and plan["generated_remaining_tasks"] == 568
assert plan["group_size"] == 8 and plan["seed"] == 20260812
assert plan["screen_shards"] == shards
assert plan["tasks_sha256"] == sys.argv[5]
entry = plan["assignments"][index]
assert entry["shard_index"] == index
assert Path(entry["path"]).resolve() == assignment
assert entry["sha256"] == hashlib.sha256(assignment.read_bytes()).hexdigest()
assert entry["records"] == len([line for line in assignment.read_text().splitlines() if line])
PY
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$(( $1 + 1 ))p" | tr -d '[:space:]'
}
wait_for_gpu() {
  local gpu=$1 status=$2 used
  while true; do
    used=$(gpu_used_mib "$gpu")
    [[ "$used" =~ ^[0-9]+$ ]] || die "cannot read GPU$gpu memory usage"
    if [[ "$used" -le 512 ]]; then return 0; fi
    write_status "$status" waiting_gpu "gpu=$gpu used_mib=$used; no process will be stopped"
    sleep 30
  done
}

worker_pgid=''
terminate_owned_worker() {
  [[ -n "$worker_pgid" ]] || return 0
  if kill -0 -- "-$worker_pgid" 2>/dev/null; then
    kill -TERM -- "-$worker_pgid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$worker_pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$worker_pgid" 2>/dev/null && kill -KILL -- "-$worker_pgid" 2>/dev/null || true
  fi
  wait "$worker_pgid" 2>/dev/null || true
  worker_pgid=''
}
on_worker_signal() { terminate_owned_worker; exit "$1"; }

run_worker() {
  validate_static_inputs
  local assignment shard_name worker_out status log gpu complete
  shard_name=$(printf 'shard-%02d-of-%02d' "$SHARD_INDEX" "$SCREEN_SHARDS")
  assignment=$ASSIGNMENTS_DIR/remaining568.$shard_name.txt
  worker_out=$WORKERS_DIR/$shard_name
  status=$STATUS_DIR/worker-$shard_name.status
  log=$LOG_DIR/worker-$shard_name.log
  complete=$worker_out/worker_complete.json
  if [[ "$SHARD_INDEX" -eq 0 ]]; then gpu=$SCREEN_GPU0; else gpu=$SCREEN_GPU1; fi
  verify_plan_and_assignment "$assignment"
  mkdir -p "$worker_out/groups" "$LOCK_DIR"
  exec 8>"$LOCK_DIR/worker-$shard_name.lock"
  flock -n 8 || die "another invocation owns worker $shard_name"
  if [[ -f "$complete" ]]; then
    verify_worker_output "$assignment" "$worker_out" "$complete"
    write_status "$status" complete "existing_verified=$complete"
    return
  fi
  local unexpected
  unexpected=$(find "$worker_out" -mindepth 1 -maxdepth 1 ! -name groups -print -quit)
  [[ -z "$unexpected" ]] || die "unknown worker output: $unexpected"
  if find "$worker_out/groups" -mindepth 1 -maxdepth 1 ! -type f -print -quit | grep -q .; then
    die "worker groups contain a non-file entry: $worker_out/groups"
  fi
  if find "$worker_out/groups" -mindepth 1 -maxdepth 1 -type f ! -name '*.json' -print -quit | grep -q .; then
    die "worker groups contain an unknown or partial file: $worker_out/groups"
  fi
  wait_for_gpu "$gpu" "$status"
  write_status "$status" generating "gpu=$gpu assignment=$assignment resume_groups=true"
  trap 'on_worker_signal 130' INT
  trap 'on_worker_signal 143' TERM
  trap terminate_owned_worker EXIT
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 \
    TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
    PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
    "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
      --model-path "$MODEL_PATH" \
      --adapter-path "$SFT1_ADAPTER" \
      --tasks "$TASKS_JSONL" \
      --output-dir "$worker_out" \
      --task-id-file "$assignment" \
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
      --question-window 4 >>"$log" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_owned_worker
  trap - EXIT INT TERM
  [[ "$code" -eq 0 ]] || { write_status "$status" failed "exit=$code log=$log"; return "$code"; }
  write_worker_completion "$assignment" "$worker_out" "$complete"
  verify_worker_output "$assignment" "$worker_out" "$complete"
  write_status "$status" complete "gpu=$gpu completion=$complete"
}

write_worker_completion() {
  local assignment=$1 worker_out=$2 complete=$3
  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$assignment" "$worker_out" "$complete" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
tasks, assignment, worker, output = map(Path, sys.argv[1:5])
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
order = {task_id: position for position, task_id in enumerate(ids)}
assigned = [line for line in assignment.read_text().splitlines() if line]
assert len(assigned) == len(set(assigned)) and not set(assigned) & set(ids[:32])
groups = worker / "groups"
observed = {path.stem for path in groups.glob("*.json")}
assert observed == set(assigned), (len(observed), len(assigned))
files = []
for task_id in assigned:
    path = groups / f"{task_id}.json"
    payload = path.read_bytes()
    group = json.loads(payload)
    assert len(group) == 8
    for sample_index, row in enumerate(group):
        assert row["environment"]["task_id"] == task_id
        assert int(row["sequence"]) == order[task_id] * 8 + sample_index
        assert int(row["sample"]["audit_record"]["sample_index"]) == sample_index
    files.append({"task_id": task_id, "sha256": hashlib.sha256(payload).hexdigest()})
record = {
    "schema_version": "vanilla-grpo-boundary-screen-worker-v1",
    "status": "complete",
    "assignment_path": str(assignment.resolve()),
    "assignment_sha256": hashlib.sha256(assignment.read_bytes()).hexdigest(),
    "tasks": len(assigned),
    "group_size": 8,
    "group_files": files,
}
encoded = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
temporary = output.with_suffix(".json.next")
temporary.write_bytes(encoded); os.replace(temporary, output)
PY
}

verify_worker_output() {
  local assignment=$1 worker_out=$2 complete=$3
  env PYTHONPATH= "$PYTHON_BIN" - "$assignment" "$worker_out" "$complete" <<'PY'
import hashlib, json, sys
from pathlib import Path
assignment, worker, complete = map(Path, sys.argv[1:4])
record = json.loads(complete.read_text())
assert record["schema_version"] == "vanilla-grpo-boundary-screen-worker-v1"
assert record["status"] == "complete" and record["group_size"] == 8
assert record["assignment_sha256"] == hashlib.sha256(assignment.read_bytes()).hexdigest()
assigned = [line for line in assignment.read_text().splitlines() if line]
assert record["tasks"] == len(assigned)
assert [entry["task_id"] for entry in record["group_files"]] == assigned
group_entries = list((worker / "groups").iterdir())
assert all(path.is_file() and path.name.endswith(".json") for path in group_entries)
assert {path.stem for path in group_entries} == set(assigned)
for entry in record["group_files"]:
    path = worker / "groups" / f'{entry["task_id"]}.json'
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
PY
}

reuse_first32() {
  validate_static_inputs
  [[ -f "$PLAN" ]] || die "missing prepared plan: $PLAN"
  mkdir -p "$STATUS_DIR" "$LOCK_DIR" "$POOL_OUT/groups"
  exec 9>"$LOCK_DIR/control.lock"
  flock -n 9 || die "another control operation owns $LOCK_DIR/control.lock"
  local source_manifest=$FIRST32_SOURCE/manifest.pending.json
  local source_trajectories=$FIRST32_SOURCE/trajectories.jsonl
  [[ -f "$source_manifest" && -f "$source_trajectories" ]] \
    || die "first32 source is not finalized: $FIRST32_SOURCE"
  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$FIRST32_SOURCE" "$POOL_OUT" \
    "$REUSE_MANIFEST" "$EXPECTED_TASKS_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" "$EXPECTED_PROTOCOL_VERSION" \
    "$EXPECTED_PROTOCOL_HASH" "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
tasks, source, pool, output = map(Path, sys.argv[1:5])
expected_tasks, expected_adapter, expected_tree, expected_version, expected_protocol, expected_prompt = sys.argv[5:11]
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
manifest_path, trajectories_path = source / "manifest.pending.json", source / "trajectories.jsonl"
manifest = json.loads(manifest_path.read_text())
assert manifest["tasks_sha256"] == expected_tasks
assert manifest["adapter_sha256"] == expected_adapter
assert manifest["protocol_runtime_content_tree_sha256"] == expected_tree
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_protocol
assert manifest["student_prompt_sha256"] == expected_prompt
for key, value in {
    "tasks": 32, "group_size": 8, "trajectories": 256, "temperature": 0.8,
    "top_p": 1.0, "max_steps": 30, "max_new_tokens": 2048,
    "max_context_tokens": 16384, "history_turns": 4,
    "enable_thinking": True, "seed": 20260812,
    "result_reward_profile": "binary", "reward_mode": "result-only",
}.items():
    assert manifest[key] == value, (key, manifest[key], value)
assert manifest["trajectories_sha256"] == hashlib.sha256(trajectories_path.read_bytes()).hexdigest()
trajectory_rows = [json.loads(line) for line in trajectories_path.read_text().splitlines() if line]
assert len(trajectory_rows) == 256
files = []
source_group_rows = []
destination = pool / "groups"
destination.mkdir(parents=True, exist_ok=True)
for position, task_id in enumerate(ids[:32]):
    path = source / "groups" / f"{task_id}.json"
    payload = path.read_bytes()
    group = json.loads(payload)
    assert len(group) == 8
    for sample_index, row in enumerate(group):
        assert row["environment"]["task_id"] == task_id
        assert int(row["sequence"]) == position * 8 + sample_index
        assert int(row["sample"]["audit_record"]["sample_index"]) == sample_index
    source_group_rows.extend(group)
    target = destination / path.name
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"refusing non-identical reused group: {target}")
    temporary = target.with_suffix(".json.next")
    temporary.write_bytes(payload); os.replace(temporary, target)
    assert target.read_bytes() == payload
    files.append({"task_id": task_id, "sha256": hashlib.sha256(payload).hexdigest()})
source_group_rows.sort(key=lambda row: int(row["sequence"]))
reconstructed = "".join(
    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    for row in source_group_rows
).encode()
assert reconstructed == trajectories_path.read_bytes()
record = {
    "schema_version": "vanilla-grpo-boundary-screen-first32-reuse-v1",
    "status": "byte_identical_reuse_complete",
    "source_manifest_path": str(manifest_path.resolve()),
    "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "source_trajectories_path": str(trajectories_path.resolve()),
    "source_trajectories_sha256": hashlib.sha256(trajectories_path.read_bytes()).hexdigest(),
    "tasks": 32, "group_size": 8, "group_files": files,
}
encoded = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
if output.exists() and output.read_bytes() != encoded:
    raise RuntimeError(f"refusing changed reuse manifest: {output}")
temporary = output.with_suffix(".json.next")
temporary.write_bytes(encoded); os.replace(temporary, output)
PY
  write_status "$STATUS_DIR/reuse-first32.status" complete "source=$FIRST32_SOURCE manifest=$REUSE_MANIFEST"
}

merge_all_groups() {
  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$PLAN" "$REUSE_MANIFEST" \
    "$WORKERS_DIR" "$POOL_OUT" "$SCREEN_SHARDS" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
tasks, plan_path, reuse_path, workers, pool = map(Path, sys.argv[1:6])
shards = int(sys.argv[6])
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert len(ids) == 600 and len(set(ids)) == 600
plan, reuse = json.loads(plan_path.read_text()), json.loads(reuse_path.read_text())
assert plan["screen_shards"] == shards
assert reuse["status"] == "byte_identical_reuse_complete"
sources = {}
for entry in reuse["group_files"]:
    task_id = entry["task_id"]
    path = pool / "groups" / f"{task_id}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    sources[task_id] = path
for shard in range(shards):
    name = f"shard-{shard:02d}-of-{shards:02d}"
    complete = workers / name / "worker_complete.json"
    record = json.loads(complete.read_text())
    assert record["status"] == "complete"
    for entry in record["group_files"]:
        task_id = entry["task_id"]
        path = workers / name / "groups" / f"{task_id}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert task_id not in sources
        sources[task_id] = path
assert set(sources) == set(ids) and len(sources) == 600
destination = pool / "groups"
for task_id in ids:
    source, target = sources[task_id], destination / f"{task_id}.json"
    payload = source.read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"refusing non-identical merged group: {target}")
    temporary = target.with_suffix(".json.next")
    temporary.write_bytes(payload); os.replace(temporary, target)
destination_entries = list(destination.iterdir())
assert all(path.is_file() and path.name.endswith(".json") for path in destination_entries)
assert {path.stem for path in destination_entries} == set(ids)
PY
}

verify_final_pool_identity() {
  env PYTHONPATH= "$PYTHON_BIN" - "$TASKS_JSONL" "$POOL_OUT" "$REUSE_MANIFEST" \
    "$WORKERS_DIR" "$SCREEN_SHARDS" "$EXPECTED_TASKS_SHA256" \
    "$EXPECTED_SFT1_SHA256" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
tasks, pool, reuse_path, workers = map(Path, sys.argv[1:5])
shards = int(sys.argv[5])
expected_tasks, expected_adapter, expected_tree, expected_version, expected_protocol, expected_prompt = sys.argv[6:12]
manifest_path, trajectories_path = pool / "manifest.pending.json", pool / "trajectories.jsonl"
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == "table-agent-fixed-rollout-pool-pending-v1"
assert manifest["tasks_sha256"] == expected_tasks
assert manifest["adapter_sha256"] == expected_adapter
assert manifest["protocol_runtime_content_tree_sha256"] == expected_tree
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_protocol
assert manifest["student_prompt_sha256"] == expected_prompt
for key, value in {
    "tasks": 600, "group_size": 8, "trajectories": 4800,
    "reward_mode": "result-only", "result_reward_profile": "binary",
    "temperature": 0.8, "top_p": 1.0, "max_steps": 30,
    "max_new_tokens": 2048, "max_context_tokens": 16384,
    "history_turns": 4, "enable_thinking": True, "seed": 20260812,
}.items():
    assert manifest[key] == value, (key, manifest[key], value)
assert manifest["trajectories_sha256"] == hashlib.sha256(trajectories_path.read_bytes()).hexdigest()
task_rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in task_rows]
trajectory_rows = [json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()]
assert len(trajectory_rows) == 4800
assert [int(row["sequence"]) for row in trajectory_rows] == list(range(4800))
counts = collections.Counter(str(row["environment"]["task_id"]) for row in trajectory_rows)
assert counts == collections.Counter({task_id: 8 for task_id in ids})
expected_group_hashes = {
    entry["task_id"]: entry["sha256"]
    for entry in json.loads(reuse_path.read_text())["group_files"]
}
for shard in range(shards):
    name = f"shard-{shard:02d}-of-{shards:02d}"
    record = json.loads((workers / name / "worker_complete.json").read_text())
    for entry in record["group_files"]:
        assert entry["task_id"] not in expected_group_hashes
        expected_group_hashes[entry["task_id"]] = entry["sha256"]
assert set(expected_group_hashes) == set(ids)
for task_id, expected in expected_group_hashes.items():
    assert hashlib.sha256((pool / "groups" / f"{task_id}.json").read_bytes()).hexdigest() == expected
PY
}

read_boundary_next_stage() {
  local audit_path=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$audit_path" \
    "$POOL_OUT/manifest.pending.json" "$POOL_OUT/trajectories.jsonl" \
    "$TASKS_JSONL" <<'PY'
import hashlib, json, sys
from pathlib import Path

audit_path, manifest_path, trajectories_path, tasks_path = map(Path, sys.argv[1:5])
audit = json.loads(audit_path.read_text())
assert audit["schema_version"] == "vanilla-grpo-boundary-screen-audit-v1"
assert audit["issues"] == [] and audit["issue_counts"] == {}
status = audit["status"]
assert status["audit_passes"] is True and status["pool_admitted"] is True
inputs = audit["inputs"]
for key, path in (
    ("manifest", manifest_path),
    ("trajectories", trajectories_path),
    ("tasks", tasks_path),
):
    assert Path(inputs[key]).resolve() == path.resolve()
    assert inputs[f"{key}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
next_stage = status["next_stage"]
ready = status["selection_ready_without_s2"]
assert (next_stage, ready) in {("select_s1", True), ("screen_s2", False)}
print(next_stage)
PY
}

verify_boundary_selection() {
  local audit_path=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$audit_path" "$TASKS_JSONL" \
    "$BOUNDARY_AUDIT_DIR" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

audit_path, tasks_path, output_dir = map(Path, sys.argv[1:4])
manifest_path = output_dir / "boundary_cohort_manifest.json"
paths = {
    "boundary": output_dir / "boundary332.jsonl",
    "train": output_dir / "train300.jsonl",
    "validation": output_dir / "validation32.jsonl",
}
assert manifest_path.is_file() and not manifest_path.is_symlink()
assert all(path.is_file() and not path.is_symlink() for path in paths.values())
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
assert manifest["schema_version"] == "policy-boundary-grpo-cohort-v1"
assert manifest["status"] == "frozen_boundary_training_cohort"
assert len(manifest["screen_pools"]) == 1
pool = manifest["screen_pools"][0]
assert Path(pool["audit_path"]).resolve() == audit_path.resolve()
assert pool["audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
assert Path(pool["tasks_path"]).resolve() == tasks_path.resolve()
assert pool["tasks_sha256"] == hashlib.sha256(tasks_path.read_bytes()).hexdigest()
contract = manifest["contract"]
assert contract["boundary_records"] == 332
assert contract["train_records"] == 300
assert contract["validation_records"] == 32
formal = contract["formal_training"]
assert formal == {
    "optimizer_updates": 20,
    "prompts_per_update": 30,
    "group_size": 8,
    "train_passes": 2,
    "prompt_appearances": 600,
    "fresh_online_trajectories": 4800,
    "sampler": "trl-0.29-repeat-sampler-v1",
    "shuffle_dataset": True,
    "data_seed": 20260812,
    "task_order": "two deterministic data-seed shuffled passes",
    "per_pass_coverage": "each train300 identity exactly once",
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
}
assert contract["validation"] == {
    "policy": "fresh initial-SFT1",
    "records": 32,
    "group_size": 8,
    "seed": 20260813,
    "screen_seed_must_differ": 20260812,
    "generation_seed_scheme": "sha256-task-sample-turn-v1",
    "gate": "same >=20/32 mixed probe gate",
    "runtime_contamination_allowed": False,
    "screen_trajectories_reused": False,
}
assert contract["primary_checkpoint"] == "final-step20-only"
assert manifest["selection"]["screen_stage"] == "S1"
assert all(manifest["selection"]["acceptance_gates"].values())

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

source_rows = read_jsonl(tasks_path)
source_by_id = {
    str(row.get("example_id") or row.get("instance_id")): row for row in source_rows
}
assert len(source_rows) == len(source_by_id) == 600
expected_counts = {"boundary": 332, "train": 300, "validation": 32}
observed_ids = {}
verified_outputs = {}
for name, path in paths.items():
    data = path.read_bytes()
    rows = read_jsonl(path)
    ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
    assert len(rows) == expected_counts[name] == len(set(ids))
    assert all(source_by_id[task_id] == row for task_id, row in zip(ids, rows, strict=True))
    declared = manifest["outputs"][name]
    actual_sha = hashlib.sha256(data).hexdigest()
    assert Path(declared["path"]).resolve() == path.resolve()
    assert declared["records"] == expected_counts[name]
    assert declared["sha256"] == actual_sha
    assert manifest["task_ids"][name] == ids
    observed_ids[name] = ids
    verified_outputs[name] = {
        "path": str(path.resolve()), "records": len(rows), "sha256": actual_sha
    }
assert set(observed_ids["train"]).isdisjoint(observed_ids["validation"])
assert set(observed_ids["train"]) | set(observed_ids["validation"]) == set(observed_ids["boundary"])
verification = {
    "schema_version": "policy-boundary-grpo-cohort-verification-v1",
    "status": "verified",
    "manifest": {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    },
    "outputs": verified_outputs,
}
encoded = (json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
verification_path = output_dir / "boundary_selection_verification.json"
temporary = verification_path.with_name(verification_path.name + ".next")
temporary.write_bytes(encoded)
os.replace(temporary, verification_path)
print(verification["manifest"]["sha256"])
PY
}

finalize() {
  validate_static_inputs
  [[ -f "$PLAN" && -f "$REUSE_MANIFEST" ]] || die 'prepare and reuse-first32 must complete first'
  mkdir -p "$STATUS_DIR" "$LOCK_DIR" "$BOUNDARY_AUDIT_DIR"
  exec 9>"$LOCK_DIR/control.lock"
  flock -n 9 || die "another control operation owns $LOCK_DIR/control.lock"
  require_sha "$BOUNDARY_AUDITOR" "$EXPECTED_BOUNDARY_AUDITOR_SHA256" boundary_auditor
  write_status "$FINALIZE_STATUS" merging 'strict byte merge of first32 plus all worker shards'
  merge_all_groups
  write_status "$FINALIZE_STATUS" finalizing 'generator --finalize-only; no model is loaded'
  env \
    HF_HUB_OFFLINE=1 \
    TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
    PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
    "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
      --model-path "$MODEL_PATH" \
      --adapter-path "$SFT1_ADAPTER" \
      --tasks "$TASKS_JSONL" \
      --output-dir "$POOL_OUT" \
      --group-size 8 \
      --temperature 0.8 \
      --top-p 1 \
      --max-steps 30 \
      --max-new-tokens 2048 \
      --max-context-tokens 16384 \
      --history-turns 4 \
      --enable-thinking \
      --seed 20260812 \
      --finalize-only >>"$LOG_DIR/finalize.log" 2>&1
  verify_final_pool_identity
  write_status "$FINALIZE_STATUS" auditing "auditor=$BOUNDARY_AUDITOR"
  local audit_path=$BOUNDARY_AUDIT_DIR/boundary_screen_audit.json
  env PYTHONPATH= "$PYTHON_BIN" "$BOUNDARY_AUDITOR" \
    --manifest "$POOL_OUT/manifest.pending.json" \
    --trajectories "$POOL_OUT/trajectories.jsonl" \
    --tasks "$TASKS_JSONL" \
    --output-dir "$BOUNDARY_AUDIT_DIR" \
    --overwrite >>"$LOG_DIR/boundary_audit.log" 2>&1
  [[ -f "$audit_path" && ! -L "$audit_path" ]] \
    || die "external auditor did not write a regular boundary_screen_audit.json: $BOUNDARY_AUDIT_DIR"
  local next_stage
  if ! next_stage=$(read_boundary_next_stage "$audit_path"); then
    die "boundary audit status or input binding is invalid: $audit_path"
  fi
  if [[ "$next_stage" == screen_s2 ]]; then
    write_status "$FINALIZE_STATUS" requires_s2 \
      "audit=$audit_path selection_ready_without_s2=false; S2 is not automatically started"
    printf 'requires_s2: S1 boundary thresholds were not met; selector was not called; audit=%s\n' \
      "$audit_path" >&2
    return 4
  fi
  [[ "$next_stage" == select_s1 ]] \
    || die "unsupported boundary audit next_stage=$next_stage"
  require_sha "$BOUNDARY_SELECTOR" "$EXPECTED_BOUNDARY_SELECTOR_SHA256" boundary_selector

  local selection_paths=(
    "$BOUNDARY_AUDIT_DIR/boundary332.jsonl"
    "$BOUNDARY_AUDIT_DIR/train300.jsonl"
    "$BOUNDARY_AUDIT_DIR/validation32.jsonl"
    "$BOUNDARY_AUDIT_DIR/boundary_cohort_manifest.json"
  )
  local existing_selection=0 selection_path
  for selection_path in "${selection_paths[@]}"; do
    [[ ! -L "$selection_path" ]] || die "selection output may not be a symlink: $selection_path"
    if [[ -e "$selection_path" ]]; then
      existing_selection=$((existing_selection + 1))
    fi
  done
  if [[ "$existing_selection" -eq 0 ]]; then
    write_status "$FINALIZE_STATUS" selecting \
      "next_stage=select_s1 selector=$BOUNDARY_SELECTOR"
    env PYTHONPATH= "$PYTHON_BIN" "$BOUNDARY_SELECTOR" \
      --screen-audit "$audit_path" \
      --tasks "$TASKS_JSONL" \
      --output-dir "$BOUNDARY_AUDIT_DIR" >>"$LOG_DIR/boundary_selection.log" 2>&1
  elif [[ "$existing_selection" -ne 4 ]]; then
    die "partial boundary selection exists ($existing_selection/4); refusing overwrite"
  fi
  local selection_manifest_sha
  if ! selection_manifest_sha=$(verify_boundary_selection "$audit_path"); then
    die "boundary332/train300/validation32 manifest or SHA verification failed"
  fi
  write_status "$FINALIZE_STATUS" complete \
    "pool=$POOL_OUT audit=$audit_path selection=select_s1 manifest_sha256=$selection_manifest_sha"
}

case "$MODE" in
  dry-run) dry_run ;;
  prepare) prepare ;;
  worker) run_worker ;;
  reuse-first32) reuse_first32 ;;
  finalize) finalize ;;
esac
