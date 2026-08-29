#!/usr/bin/env bash
# Fail-closed S2 policy-boundary screening for the frozen Qwen3-8B v26 SFT1.
#
# S2 is activated only by one explicitly hash-frozen S1 requires_s2 audit. It
# screens a disjoint public-field-selected extra600 from scratch; no S1 rollout
# is reused. Nothing runs by default and this launcher never searches for or
# signals another experiment's processes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|prepare|worker|finalize]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|prepare|worker|finalize) ;;
  *) printf 'usage: %s [dry-run|prepare|worker|finalize]\n' "$0" >&2; exit 2 ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
PROJECT_ROOT=${PROJECT_ROOT:-/home/dengyan/tabular_rl_project}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_vanilla_grpo_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
REMOTE_DB_ROOT=${REMOTE_DB_ROOT:-$PROJECT_ROOT/data/bird/train/train_databases}

S1_SCREEN_ROOT=${S1_SCREEN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_2gpu_20260812}
S1_TASKS=${S1_TASKS:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl}
S1_COHORT_MANIFEST=${S1_COHORT_MANIFEST:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json}
S1_AUDIT=${S1_AUDIT:-$S1_SCREEN_ROOT/boundary_selection/boundary_screen_audit.json}
FROZEN_S1_AUDIT_SHA256=${FROZEN_S1_AUDIT_SHA256:-}

REFERENCE_TASKS=${REFERENCE_TASKS:-$PROJECT_ROOT/data/eval_inputs/bird_train_filtered.jsonl}
ELIGIBLE_TASKS=${ELIGIBLE_TASKS:-$PROJECT_ROOT/data/eval_inputs/bird_train_tool_compatible_nonempty_v1.jsonl}
BASELINE_EVAL300=${BASELINE_EVAL300:-$PROJECT_ROOT/data/eval_inputs/bird_train_baseline300_v1.jsonl}
# These two artifacts are intentionally not guessed. They must be copied into a
# dedicated persistent input directory and passed explicitly on active modes.
SFT1_INDEX=${SFT1_INDEX:-}
OLD_MIXED60=${OLD_MIXED60:-}

RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s2_2gpu_20260812}
S2_INPUT_DIR=${S2_INPUT_DIR:-$RUN_ROOT/inputs}
S2_TASKS=${S2_TASKS:-$S2_INPUT_DIR/s2_extra600.jsonl}
S2_COHORT_MANIFEST=${S2_COHORT_MANIFEST:-$S2_INPUT_DIR/s2_extra600.manifest.json}
POOL_OUT=${POOL_OUT:-$RUN_ROOT/pool_extra600_s2}
S2_AUDIT_DIR=${S2_AUDIT_DIR:-$RUN_ROOT/s2_audit}
BOUNDARY_SELECTION_DIR=${BOUNDARY_SELECTION_DIR:-$RUN_ROOT/boundary_selection}

S2_PREPARER=${S2_PREPARER:-$TRAIN_RUNTIME/src/rl/prepare_policy_boundary_s2_tasks.py}
S2_PREPARATION_RECOVERY_HELPER=${S2_PREPARATION_RECOVERY_HELPER:-$TRAIN_RUNTIME/src/rl/diagnostics/recover_policy_boundary_s2_preparation.py}
GROUP_RESUME_HELPER=${GROUP_RESUME_HELPER:-$TRAIN_RUNTIME/src/rl/diagnostics/prepare_boundary_screen_group_resume.py}
SELECTION_RECOVERY_HELPER=${SELECTION_RECOVERY_HELPER:-$TRAIN_RUNTIME/src/rl/diagnostics/recover_policy_boundary_selection.py}
BOUNDARY_AUDITOR=${BOUNDARY_AUDITOR:-$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_boundary_screen.py}
BOUNDARY_SELECTOR=${BOUNDARY_SELECTOR:-$TRAIN_RUNTIME/src/rl/select_policy_boundary_grpo_tasks.py}
SCREEN_SHARDS=2
SHARD_INDEX=${SHARD_INDEX:-0}
SCREEN_GPU0=${SCREEN_GPU0:-0}
SCREEN_GPU1=${SCREEN_GPU1:-1}

ASSIGNMENTS_DIR=$RUN_ROOT/assignments
WORKERS_DIR=$RUN_ROOT/workers
STATUS_DIR=$RUN_ROOT/status
LOG_DIR=$RUN_ROOT/logs
LOCK_DIR=$RUN_ROOT/locks
PLAN=$RUN_ROOT/s2_screen_plan.json
FINALIZE_STATUS=$STATUS_DIR/finalize.status
QUARANTINE_DIR=$RUN_ROOT/resume_quarantine

EXPECTED_PREPARER_SHA256=069f17a32c39aa28777432a664ead17a1d90eab68c62ff3811f4fc9bc19ccc18
EXPECTED_S2_PREPARATION_RECOVERY_HELPER_SHA256=dbd331d650a376ff6065601ce5c781d0f77f70b5e23ecaa5ec886002b9472092
EXPECTED_GROUP_RESUME_HELPER_SHA256=4ef3642cf6737c35a231ba0419a4d319f2052a4a2bcae1cbbd6088e5186576fe
EXPECTED_SELECTION_RECOVERY_HELPER_SHA256=bcd827433481fd5dae491d9ba210aee5313eaad798445b931f9b1e206a8d18a7
EXPECTED_BOUNDARY_AUDITOR_SHA256=b983bd5d8083de49328fb25b98fba6105b2ffc773faceb7a5d0c8eea7c9181e6
EXPECTED_BOUNDARY_SELECTOR_SHA256=2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8
EXPECTED_REFERENCE_SHA256=24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1
EXPECTED_ELIGIBLE_SHA256=c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545
EXPECTED_S1_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e
EXPECTED_S1_COHORT_MANIFEST_SHA256=9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6
EXPECTED_BASELINE_EVAL300_SHA256=87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03
EXPECTED_SFT1_INDEX_SHA256=aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5
EXPECTED_OLD_MIXED60_SHA256=ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae
EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191
EXPECTED_ROLLOUT_SHA256=87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f
EXPECTED_SCORING_SHA256=a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5
EXPECTED_TASK_LOADER_SHA256=d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b
EXPECTED_TOOL_ENV_V26_SHA256=c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c
EXPECTED_FIXED_POOL_RUNTIME_SHA256=e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d
EXPECTED_TRANSITION_BATCH_SHA256=c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8
EXPECTED_TERMINAL_REWARD_SHA256=2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b
EXPECTED_COHORT_SELECTOR_SHA256=3a37ab408d6e0e76f9b2cc12b84dbcf93677650ebb03a56ff5a130bfee8c2b94
EXPECTED_PUBLIC_SELECTOR_SHA256=afaf6b97cb80371054fdd96b21b5f44269ec690e6760cce460e22bc19a461b1c
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}
write_status() {
  local path=$1 state=$2 detail=$3 temporary=${1}.next
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$path"
}

validate_shape() {
  [[ "$SHARD_INDEX" =~ ^[01]$ ]] || die 'SHARD_INDEX must be exactly 0 or 1'
  [[ "$SCREEN_GPU0" =~ ^[0-9]+$ && "$SCREEN_GPU1" =~ ^[0-9]+$ ]] \
    || die 'SCREEN_GPU0 and SCREEN_GPU1 must be explicit nonnegative indices'
  [[ "$SCREEN_GPU0" != "$SCREEN_GPU1" ]] || die 'S2 two-shard screen requires distinct GPUs'
  local path
  for path in "$RUN_ROOT" "$S2_INPUT_DIR" "$S2_TASKS" "$S2_COHORT_MANIFEST" \
    "$POOL_OUT" "$S2_AUDIT_DIR" "$BOUNDARY_SELECTION_DIR"; do
    [[ "$path" == /* && "$path" != *'/../'* && "$path" != */.. ]] \
      || die "S2 owned paths must be absolute and traversal-free: $path"
  done
  case "$S2_INPUT_DIR" in "$RUN_ROOT"/*) ;; *) die 'S2_INPUT_DIR must be under RUN_ROOT' ;; esac
  case "$S2_TASKS" in "$S2_INPUT_DIR"/*) ;; *) die 'S2_TASKS must be under S2_INPUT_DIR' ;; esac
  case "$S2_COHORT_MANIFEST" in "$S2_INPUT_DIR"/*) ;; *) die 'S2_COHORT_MANIFEST must be under S2_INPUT_DIR' ;; esac
  case "$POOL_OUT" in "$RUN_ROOT"/*) ;; *) die 'POOL_OUT must be under RUN_ROOT' ;; esac
  case "$S2_AUDIT_DIR" in "$RUN_ROOT"/*) ;; *) die 'S2_AUDIT_DIR must be under RUN_ROOT' ;; esac
  case "$BOUNDARY_SELECTION_DIR" in "$RUN_ROOT"/*) ;; *) die 'BOUNDARY_SELECTION_DIR must be under RUN_ROOT' ;; esac
  [[ "$S2_INPUT_DIR" != "$POOL_OUT" && "$S2_INPUT_DIR" != "$S2_AUDIT_DIR" \
    && "$S2_INPUT_DIR" != "$BOUNDARY_SELECTION_DIR" && "$POOL_OUT" != "$S2_AUDIT_DIR" \
    && "$POOL_OUT" != "$BOUNDARY_SELECTION_DIR" && "$S2_AUDIT_DIR" != "$BOUNDARY_SELECTION_DIR" ]] \
    || die 'S2 input, pool, audit, and combined selection directories must differ'
  [[ "$RUN_ROOT" != "$S1_SCREEN_ROOT" && "$RUN_ROOT" != "$S1_SCREEN_ROOT"/* \
    && "$S1_SCREEN_ROOT" != "$RUN_ROOT"/* ]] \
    || die 'S2 RUN_ROOT must be isolated from the S1 screen root'
  for path in "$RUN_ROOT" "$S2_INPUT_DIR" "$POOL_OUT" "$S2_AUDIT_DIR" \
    "$BOUNDARY_SELECTION_DIR" "$ASSIGNMENTS_DIR" "$WORKERS_DIR" "$STATUS_DIR" \
    "$LOG_DIR" "$LOCK_DIR" "$QUARANTINE_DIR"; do
    [[ ! -L "$path" ]] || die "owned S2 output path may not be a symlink: $path"
  done
}

validate_resolved_owned_paths() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$S1_SCREEN_ROOT" "$S2_INPUT_DIR" "$S2_TASKS" \
    "$S2_COHORT_MANIFEST" "$POOL_OUT" "$S2_AUDIT_DIR" \
    "$BOUNDARY_SELECTION_DIR" "$ASSIGNMENTS_DIR" "$WORKERS_DIR" \
    "$STATUS_DIR" "$LOG_DIR" "$LOCK_DIR" "$QUARANTINE_DIR" <<'PY'
import sys
from pathlib import Path

run_root, s1_root, *owned = map(Path, sys.argv[1:])

resolved_root = run_root.resolve()
for path in owned:
    if not path.resolve().is_relative_to(resolved_root):
        raise SystemExit(f"resolved owned S2 path escapes RUN_ROOT: {path}")

# System ancestors such as macOS /home may legitimately be symlinks. Reject
# only user-controlled components at or below the resolved RUN_ROOT boundary.
for path in (run_root, *owned):
    current = path
    while current != run_root.parent and current.parent != current:
        if current.exists() and current.is_symlink():
            raise SystemExit(f"owned S2 path crosses symlink below RUN_ROOT: {current}")
        current = current.parent
resolved_s1 = s1_root.resolve()
if (
    resolved_root == resolved_s1
    or resolved_root.is_relative_to(resolved_s1)
    or resolved_s1.is_relative_to(resolved_root)
):
    raise SystemExit("resolved S2 RUN_ROOT overlaps the S1 screen root")
PY
}

verify_screen_audit() {
  local audit=$1 tasks=$2 expected_sha=$3 stage=$4 expected_manifest=$5 expected_trajectories=$6
  env PYTHONPATH= "$PYTHON_BIN" - "$audit" "$tasks" "$expected_sha" "$stage" \
    "$expected_manifest" "$expected_trajectories" <<'PY'
import collections, hashlib, json, re, sys
from pathlib import Path

audit_path, tasks_path = map(Path, sys.argv[1:3])
expected_sha, stage, expected_manifest, expected_trajectories = sys.argv[3:7]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for path in (audit_path, tasks_path):
    assert path.is_file() and not path.is_symlink(), path
actual_audit_sha = sha(audit_path)
if expected_sha != "-":
    assert re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    assert actual_audit_sha == expected_sha
audit = json.loads(audit_path.read_text())
assert audit["schema_version"] == "vanilla-grpo-boundary-screen-audit-v1"
assert audit["issues"] == [] and audit["issue_counts"] == {}
status = audit["status"]
assert status["audit_passes"] is True and status["pool_admitted"] is True
if stage == "S1":
    assert status == {
        "audit_passes": True, "pool_admitted": True,
        "selection_ready_without_s2": False, "next_stage": "screen_s2",
    }
else:
    assert stage == "S2"
    assert (status["selection_ready_without_s2"], status["next_stage"]) in {
        (True, "select_s1"), (False, "screen_s2")
    }
assert audit["contract"] == {
    "tasks": 600, "group_size": 8, "optimizer_updates": 0,
    "protocol_version": "version26", "protocol_hash": "4da19387399bd3a5",
    "initial_adapter_sha256": "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5",
    "reward_mode": "result-only", "result_reward_profile": "binary",
    "screen_seed": 20260812,
    "generation_seed_scheme": "sha256-task-sample-turn-v1",
    "minimum_mixed_groups": 360, "minimum_core_groups": 240,
}
inputs = audit["inputs"]
for key, path in (("tasks", tasks_path),):
    assert Path(inputs[key]).resolve() == path.resolve()
    assert inputs[f"{key}_sha256"] == sha(path)
for key in ("manifest", "trajectories"):
    path = Path(inputs[key])
    assert path.is_file() and not path.is_symlink()
    assert inputs[f"{key}_sha256"] == sha(path)
    expected = expected_manifest if key == "manifest" else expected_trajectories
    if expected != "-":
        assert path.resolve() == Path(expected).resolve()
task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
task_index = {str(row.get("example_id") or row.get("instance_id")): row for row in task_rows}
assert len(task_rows) == len(task_index) == 600
groups = audit["groups"]
assert len(groups) == 600
seen, mixed, core = set(), 0, 0
for group in groups:
    task_id = group["task_id"]
    correct = group["correct_count"]
    assert task_id in task_index and task_id not in seen
    assert group["example_index"] == task_index[task_id]["example_index"]
    assert group["trajectories"] == 8 and type(correct) is int and 0 <= correct <= 8
    assert group["uncertainty"] == (correct * (8 - correct) if 1 <= correct <= 7 else 0)
    assert group["mixed_boundary"] == (group["usable"] and 1 <= correct <= 7)
    assert group["core_boundary"] == (group["usable"] and 2 <= correct <= 6)
    assert bool(group["contamination"]) != bool(group["usable"])
    seen.add(task_id); mixed += int(group["mixed_boundary"]); core += int(group["core_boundary"])
assert seen == set(task_index)
observed = audit["observed"]
assert observed["tasks"] == 600 and observed["trajectories"] == 4800
assert observed["mixed_boundary_groups"] == mixed
assert observed["core_boundary_groups"] == core
print(actual_audit_sha)
PY
}

verify_s2_inputs() {
  env PYTHONPATH= "$PYTHON_BIN" - "$S2_TASKS" "$S2_COHORT_MANIFEST" \
    "$S1_AUDIT" "$FROZEN_S1_AUDIT_SHA256" "$S1_TASKS" \
    "$EXPECTED_REFERENCE_SHA256" "$EXPECTED_ELIGIBLE_SHA256" \
    "$EXPECTED_S1_COHORT_MANIFEST_SHA256" "$EXPECTED_BASELINE_EVAL300_SHA256" \
    "$EXPECTED_SFT1_INDEX_SHA256" "$EXPECTED_OLD_MIXED60_SHA256" \
    "$REMOTE_DB_ROOT" "$REFERENCE_TASKS" "$ELIGIBLE_TASKS" \
    "$S1_COHORT_MANIFEST" "$BASELINE_EVAL300" "$SFT1_INDEX" "$OLD_MIXED60" <<'PY'
import hashlib, json, sys
from pathlib import Path

tasks_path, manifest_path, s1_audit, s1_tasks = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5]))
expected_s1 = sys.argv[4]
expected_reference, expected_eligible, expected_s1_manifest = sys.argv[6:9]
expected_baseline, expected_sft, expected_mixed = sys.argv[9:12]
remote_root = Path(sys.argv[12]).resolve()
reference_path, eligible_path, s1_manifest_path, baseline_path, sft_path, mixed_path = map(Path, sys.argv[13:19])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for path in (tasks_path, manifest_path, s1_audit, s1_tasks, reference_path, eligible_path, s1_manifest_path, baseline_path, sft_path, mixed_path):
    assert path.is_file() and not path.is_symlink(), path
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == "bird-train-policy-boundary-s2-extra600-v1"
assert manifest["status"] == "frozen_s2_screening_cohort"
assert manifest["all_acceptance_gates_passed"] is True
assert all(manifest["acceptance_gates"].values())
activation = manifest["activation_gate"]
assert activation["decision"] == "requires_s2"
assert Path(activation["path"]).resolve() == s1_audit.resolve()
assert activation["sha256"] == expected_s1 == sha(s1_audit)
assert activation["status"] == {
    "audit_passes": True, "pool_admitted": True,
    "selection_ready_without_s2": False, "next_stage": "screen_s2",
}
assert Path(activation["bound_inputs"]["tasks"]["path"]).resolve() == s1_tasks.resolve()
assert activation["bound_inputs"]["tasks"]["sha256"] == sha(s1_tasks)
assert Path(manifest["reference_population"]["path"]).resolve() == reference_path.resolve()
assert manifest["reference_population"]["sha256"] == expected_reference == sha(reference_path)
assert manifest["reference_population"]["records"] == 6601
assert Path(manifest["eligibility_population"]["path"]).resolve() == eligible_path.resolve()
assert manifest["eligibility_population"]["sha256"] == expected_eligible == sha(eligible_path)
assert manifest["eligibility_population"]["records"] == 5915
assert Path(manifest["current_s1_cohort_manifest"]["path"]).resolve() == s1_manifest_path.resolve()
assert manifest["current_s1_cohort_manifest"]["sha256"] == expected_s1_manifest == sha(s1_manifest_path)
exclusions = {entry["role"]: entry for entry in manifest["exclusions"]}
assert set(exclusions) == {"s1_tasks", "baseline_eval300", "sft1_index", "old_mixed60"}
for role, path, expected, counts in (
    ("s1_tasks", s1_tasks, sha(s1_tasks), (600, 600)),
    ("baseline_eval300", baseline_path, expected_baseline, (300, 300)),
    ("sft1_index", sft_path, expected_sft, (4471, 678)),
    ("old_mixed60", mixed_path, expected_mixed, (60, 60)),
):
    entry = exclusions[role]
    assert Path(entry["path"]).resolve() == path.resolve()
    assert entry["sha256"] == expected == sha(path)
    assert (entry["records"], entry["unique_task_ids"]) == counts
selection = manifest["selection"]
assert selection["count"] == 600
assert selection["seed"] == "qwen3-v26-policy-boundary-s2-extra600-v1-20260812"
assert selection["gold_used_for_selection_or_order"] is False
assert set(manifest["acceptance_gates"]) == {
    "s1_requires_s2", "record_count_exact", "task_ids_unique",
    "task_example_identity_bound", "same_eligible_universe",
    "overlap_with_all_exclusions_zero", "overlap_with_current_s1_zero",
    "overlap_with_baseline_eval300_zero", "overlap_with_sft1_train678_zero",
    "overlap_with_old_mixed60_zero", "all_reference_databases_covered",
    "database_tv_at_most_0_04", "length_quintile_tv_at_most_0_03",
    "knowledge_rate_delta_at_most_0_02", "joint_knowledge_length_tv_at_most_0_04",
    "question_length_mean_delta_at_most_0_05", "remote_paths_bound",
}
output = manifest["output"]
assert Path(output["path"]).resolve() == tasks_path.resolve()
assert output["sha256"] == sha(tasks_path) and output["records"] == 600
rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert len(rows) == len(set(ids)) == 600
assert output["task_ids_in_frozen_order"] == ids
s1_ids = {
    str(row.get("example_id") or row.get("instance_id"))
    for row in (json.loads(line) for line in s1_tasks.read_text().splitlines() if line.strip())
}
assert set(ids).isdisjoint(s1_ids)
assert output["remote_db_root"] == str(remote_root)
assert output["task_example_identity_in_frozen_order"] == [
    {"task_id": task_id, "example_index": int(row["example_index"]), "db_id": str(row["db_id"])}
    for task_id, row in zip(ids, rows, strict=True)
]
for row, task_id in zip(rows, ids, strict=True):
    assert task_id == row["instance_id"] == f'bird_train_{int(row["example_index"]):05d}'
    assert row["metadata"]["rl_training_cohort"] == "bird-train-policy-boundary-s2-extra600-v1"
    assert row["metadata"]["policy_boundary_screen_stage"] == "S2"
    assert Path(row["db_path"]).resolve().is_relative_to(remote_root)
assert manifest["downstream_contract"] == {
    "screen_policy": "fresh initial-SFT1", "screen_group_size": 8,
    "screen_seed": 20260812,
    "merge_rule": "S1+S2 audits only; no S3 and no threshold relaxation",
    "training_admission": "none_until_existing_boundary332_selection_and_validation32_gates_pass",
}
PY
}

validate_static_inputs() {
  validate_shape
  [[ "$FROZEN_S1_AUDIT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || die 'FROZEN_S1_AUDIT_SHA256 must be explicitly supplied as 64 lowercase hex'
  [[ -n "$SFT1_INDEX" && -n "$OLD_MIXED60" ]] \
    || die 'SFT1_INDEX and OLD_MIXED60 must be explicit persistent input paths'
  [[ -x "$PYTHON_BIN" ]] || die "missing Python runtime: $PYTHON_BIN"
  validate_resolved_owned_paths || die 'resolved S2 path ownership/isolation check failed'
  [[ -d "$MODEL_PATH" && -d "$SFT1_ADAPTER" ]] || die 'missing model or SFT1 adapter'
  [[ -d "$PROTOCOL_RUNTIME/src/eval" && -d "$PROTOCOL_RUNTIME/src/sft" && -d "$PROTOCOL_RUNTIME/src/harness" ]] \
    || die "incomplete frozen protocol runtime: $PROTOCOL_RUNTIME"
  for forbidden in src/eval src/sft src/harness src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden" ]] \
      || die "training overlay shadows frozen protocol source: $TRAIN_RUNTIME/$forbidden"
  done
  require_sha "$S2_PREPARER" "$EXPECTED_PREPARER_SHA256" s2_preparer
  require_sha "$S2_PREPARATION_RECOVERY_HELPER" "$EXPECTED_S2_PREPARATION_RECOVERY_HELPER_SHA256" s2_preparation_recovery_helper
  require_sha "$GROUP_RESUME_HELPER" "$EXPECTED_GROUP_RESUME_HELPER_SHA256" group_resume_helper
  require_sha "$SELECTION_RECOVERY_HELPER" "$EXPECTED_SELECTION_RECOVERY_HELPER_SHA256" selection_recovery_helper
  require_sha "$BOUNDARY_AUDITOR" "$EXPECTED_BOUNDARY_AUDITOR_SHA256" boundary_auditor
  require_sha "$BOUNDARY_SELECTOR" "$EXPECTED_BOUNDARY_SELECTOR_SHA256" boundary_selector
  require_sha "$REFERENCE_TASKS" "$EXPECTED_REFERENCE_SHA256" reference6601
  require_sha "$ELIGIBLE_TASKS" "$EXPECTED_ELIGIBLE_SHA256" eligible5915
  require_sha "$S1_TASKS" "$EXPECTED_S1_TASKS_SHA256" s1_tasks
  require_sha "$S1_COHORT_MANIFEST" "$EXPECTED_S1_COHORT_MANIFEST_SHA256" s1_cohort_manifest
  require_sha "$BASELINE_EVAL300" "$EXPECTED_BASELINE_EVAL300_SHA256" baseline_eval300
  require_sha "$SFT1_INDEX" "$EXPECTED_SFT1_INDEX_SHA256" sft1_index
  require_sha "$OLD_MIXED60" "$EXPECTED_OLD_MIXED60_SHA256" old_mixed60
  require_sha "$S1_AUDIT" "$FROZEN_S1_AUDIT_SHA256" frozen_s1_requires_s2_audit
  require_sha "$TRAIN_RUNTIME/src/rl/select_vanilla_grpo_tasks.py" "$EXPECTED_COHORT_SELECTOR_SHA256" cohort_selector_dependency
  require_sha "$PROJECT_ROOT/src/eval/select_bird_train_baseline.py" "$EXPECTED_PUBLIC_SELECTOR_SHA256" public_selector_dependency
  require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" "$EXPECTED_GENERATOR_SHA256" fixed_pool_generator
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" "$EXPECTED_ROLLOUT_SHA256" rollout_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" "$EXPECTED_SCORING_SHA256" rollout_scoring
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" "$EXPECTED_TASK_LOADER_SHA256" task_loader
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" "$EXPECTED_TOOL_ENV_V26_SHA256" tool_environment_v26
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" "$EXPECTED_FIXED_POOL_RUNTIME_SHA256" fixed_rollout_pool_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" "$EXPECTED_TRANSITION_BATCH_SHA256" transition_batch_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/terminal_reward.py" "$EXPECTED_TERMINAL_REWARD_SHA256" terminal_reward
  require_sha "$MODEL_PATH/model-00001-of-00005.safetensors" 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f model_shard_1
  require_sha "$MODEL_PATH/model-00002-of-00005.safetensors" 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 model_shard_2
  require_sha "$MODEL_PATH/model-00003-of-00005.safetensors" c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 model_shard_3
  require_sha "$MODEL_PATH/model-00004-of-00005.safetensors" b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a model_shard_4
  require_sha "$MODEL_PATH/model-00005-of-00005.safetensors" 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff model_shard_5
  require_sha "$MODEL_PATH/config.json" f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 model_config
  require_sha "$MODEL_PATH/model.safetensors.index.json" f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc model_index
  require_sha "$MODEL_PATH/tokenizer_config.json" d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 tokenizer_config
  require_sha "$MODEL_PATH/tokenizer.json" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 tokenizer
  require_sha "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
  require_sha "$SFT1_ADAPTER/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256" sft1_adapter_config
  require_sha "$SFT1_ADAPTER/trainer_state.json" "$EXPECTED_SFT1_STATE_SHA256" sft1_trainer_state
  verify_screen_audit "$S1_AUDIT" "$S1_TASKS" "$FROZEN_S1_AUDIT_SHA256" S1 - - >/dev/null \
    || die 'frozen S1 audit is not an exact admitted requires_s2 decision'
  env PYTHONPATH= "$PYTHON_BIN" - "$PROTOCOL_RUNTIME" \
    "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" "$EXPECTED_PROTOCOL_VERSION" \
    "$EXPECTED_PROTOCOL_HASH" "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import hashlib, importlib, sys
from pathlib import Path
runtime = Path(sys.argv[1]).resolve()
digest, files = hashlib.sha256(), []
for relative in ("src/eval", "src/sft", "src/harness"):
    files.extend(path for path in (runtime / relative).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
for path in sorted(files, key=lambda item: item.relative_to(runtime).as_posix()):
    digest.update(path.relative_to(runtime).as_posix().encode()); digest.update(b"\0")
    digest.update(path.read_bytes()); digest.update(b"\0")
assert digest.hexdigest() == sys.argv[2]
sys.path.insert(0, str(runtime / "src/sft"))
protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(context_mode="rolling-legal-history", compact=False)
assert protocol.PROTOCOL_VERSION == sys.argv[3]
assert protocol.protocol_hash(prompt) == sys.argv[4]
assert hashlib.sha256(prompt.encode()).hexdigest() == sys.argv[5]
PY
}

dry_run() {
  validate_shape
  printf '%s\n' \
    'boundary-screen S2 dry-run (no files written, no GPU inspected)' \
    'activation=explicit frozen S1 requires_s2 audit SHA only' \
    'screen=extra600 fresh initial-SFT1 K8 seed20260812; no S1 reuse' \
    "shards=2 shard0_gpu=$SCREEN_GPU0 shard1_gpu=$SCREEN_GPU1" \
    "run_root=$RUN_ROOT" \
    'lifecycle=prepare -> worker[0] + worker[1] -> finalize' \
    'finalize=S2 audit -> selector(S1,S2) -> canonical boundary_selection; no S3' \
    'active modes additionally require FROZEN_S1_AUDIT_SHA256, SFT1_INDEX, OLD_MIXED60'
}

write_plan() {
  env PYTHONPATH= "$PYTHON_BIN" - "$S2_TASKS" "$S2_COHORT_MANIFEST" \
    "$RUN_ROOT" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$FROZEN_S1_AUDIT_SHA256" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
tasks, cohort, root = map(Path, sys.argv[1:4])
runtime_sha, adapter_sha, s1_audit_sha = sys.argv[4:7]
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert len(ids) == len(set(ids)) == 600
assignment_dir = root / "assignments"
assignments = []
for shard in range(2):
    assigned = [task_id for position, task_id in enumerate(ids) if position % 2 == shard]
    payload = "".join(f"{task_id}\n" for task_id in assigned).encode()
    path = assignment_dir / f"extra600.shard-{shard:02d}-of-02.txt"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing changed assignment: {path}")
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_bytes(payload); os.replace(temporary, path)
    assignments.append({"shard_index": shard, "records": 300, "path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
expected_names = {Path(entry["path"]).name for entry in assignments}
assert {path.name for path in assignment_dir.iterdir()} == expected_names
plan = {
    "schema_version": "vanilla-grpo-boundary-screen-s2-plan-v1", "status": "prepared", "stage": "S2",
    "tasks_path": str(tasks.resolve()), "tasks_sha256": hashlib.sha256(tasks.read_bytes()).hexdigest(), "tasks": 600,
    "cohort_manifest_path": str(cohort.resolve()), "cohort_manifest_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
    "s1_requires_s2_audit_sha256": s1_audit_sha, "reused_tasks": 0, "generated_tasks": 600,
    "group_size": 8, "temperature": 0.8, "top_p": 1.0, "max_steps": 30,
    "max_new_tokens": 2048, "max_context_tokens": 16384, "history_turns": 4,
    "enable_thinking": True, "seed": 20260812, "screen_shards": 2,
    "scheduler": "static", "task_batch_size": 1, "question_window": None,
    "protocol_runtime_content_tree_sha256": runtime_sha, "initial_adapter_sha256": adapter_sha,
    "assignments": assignments, "next_stage": "S1+S2 selector only; no S3",
}
encoded = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
path = root / "s2_screen_plan.json"
if path.exists() and path.read_bytes() != encoded:
    raise RuntimeError(f"refusing changed S2 screen plan: {path}")
temporary = path.with_suffix(".json.next"); temporary.write_bytes(encoded); os.replace(temporary, path)
PY
}

prepare() {
  validate_static_inputs
  mkdir -p "$S2_INPUT_DIR" "$ASSIGNMENTS_DIR" "$WORKERS_DIR" "$STATUS_DIR" "$LOG_DIR" "$LOCK_DIR" "$POOL_OUT/groups" "$S2_AUDIT_DIR" "$BOUNDARY_SELECTION_DIR"
  exec 9>"$LOCK_DIR/control.lock"
  flock -n 9 || die "another S2 control operation owns $LOCK_DIR/control.lock"
  write_status "$STATUS_DIR/prepare.status" preparing "recoverable S1_audit_sha256=$FROZEN_S1_AUDIT_SHA256"
  env PYTHONPATH="$TRAIN_RUNTIME/src/rl:$PROJECT_ROOT/src/eval" \
    "$PYTHON_BIN" "$S2_PREPARATION_RECOVERY_HELPER" \
      --preparer "$S2_PREPARER" \
      --quarantine-dir "$QUARANTINE_DIR/s2-preparation" \
      --s1-audit "$S1_AUDIT" \
      --expected-s1-audit-sha256 "$FROZEN_S1_AUDIT_SHA256" \
      --reference "$REFERENCE_TASKS" \
      --eligible "$ELIGIBLE_TASKS" \
      --s1-tasks "$S1_TASKS" \
      --s1-cohort-manifest "$S1_COHORT_MANIFEST" \
      --baseline-eval300 "$BASELINE_EVAL300" \
      --sft1-index "$SFT1_INDEX" \
      --old-mixed60 "$OLD_MIXED60" \
      --output "$S2_TASKS" \
      --manifest "$S2_COHORT_MANIFEST" \
      --remote-db-root "$REMOTE_DB_ROOT" >>"$LOG_DIR/prepare.log" 2>&1 \
    || die "frozen S2 preparer/recovery rejected activation or inputs; see $LOG_DIR/prepare.log"
  verify_s2_inputs || die 'S2 extra600 tasks/manifest identity verification failed'
  write_plan || die 'S2 assignment/plan preparation failed'
  write_status "$STATUS_DIR/prepare.status" complete "tasks=$S2_TASKS shards=2 records=600"
}

verify_plan_assignment() {
  local assignment=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$PLAN" "$assignment" "$SHARD_INDEX" \
    "$S2_TASKS" "$S2_COHORT_MANIFEST" "$FROZEN_S1_AUDIT_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path
plan_path, assignment, tasks, cohort = map(Path, (sys.argv[1], sys.argv[2], sys.argv[4], sys.argv[5]))
index, expected_s1 = int(sys.argv[3]), sys.argv[6]
plan = json.loads(plan_path.read_text())
assert plan["schema_version"] == "vanilla-grpo-boundary-screen-s2-plan-v1"
assert plan["status"] == "prepared" and plan["stage"] == "S2"
assert plan["tasks"] == 600 and plan["generated_tasks"] == 600 and plan["reused_tasks"] == 0
assert plan["group_size"] == 8 and plan["seed"] == 20260812 and plan["screen_shards"] == 2
for key, expected in {
    "temperature": 0.8, "top_p": 1.0, "max_steps": 30,
    "max_new_tokens": 2048, "max_context_tokens": 16384,
    "history_turns": 4, "enable_thinking": True,
    "scheduler": "static", "task_batch_size": 1, "question_window": None,
}.items():
    assert plan[key] == expected, (key, plan[key], expected)
assert Path(plan["tasks_path"]).resolve() == tasks.resolve()
assert plan["tasks_sha256"] == hashlib.sha256(tasks.read_bytes()).hexdigest()
assert Path(plan["cohort_manifest_path"]).resolve() == cohort.resolve()
assert plan["cohort_manifest_sha256"] == hashlib.sha256(cohort.read_bytes()).hexdigest()
assert plan["s1_requires_s2_audit_sha256"] == expected_s1
task_rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
task_ids = [str(row.get("example_id") or row.get("instance_id")) for row in task_rows]
for shard, recorded in enumerate(plan["assignments"]):
    recorded_path = Path(recorded["path"])
    expected_ids = task_ids[shard::2]
    assert recorded["shard_index"] == shard and recorded["records"] == 300
    assert [line for line in recorded_path.read_text().splitlines() if line] == expected_ids
    assert recorded["sha256"] == hashlib.sha256(recorded_path.read_bytes()).hexdigest()
entry = plan["assignments"][index]
assert entry["shard_index"] == index and entry["records"] == 300
assert Path(entry["path"]).resolve() == assignment.resolve()
assert entry["sha256"] == hashlib.sha256(assignment.read_bytes()).hexdigest()
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

verify_worker_output() {
  local assignment=$1 worker_out=$2 complete=$3
  env PYTHONPATH= "$PYTHON_BIN" - "$S2_TASKS" "$assignment" "$worker_out" "$complete" <<'PY'
import hashlib, json, sys
from pathlib import Path
tasks, assignment, worker, complete = map(Path, sys.argv[1:5])
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
order = {task_id: position for position, task_id in enumerate(ids)}
assigned = [line for line in assignment.read_text().splitlines() if line]
record = json.loads(complete.read_text())
assert record["schema_version"] == "vanilla-grpo-boundary-screen-s2-worker-v1"
assert record["status"] == "complete" and record["group_size"] == 8 and record["tasks"] == 300
assert Path(record["assignment_path"]).resolve() == assignment.resolve()
assert record["assignment_sha256"] == hashlib.sha256(assignment.read_bytes()).hexdigest()
assert [entry["task_id"] for entry in record["group_files"]] == assigned
root_entries = list(worker.iterdir())
assert {path.name for path in root_entries} == {"groups", "worker_complete.json"}
assert all(not path.is_symlink() for path in root_entries)
entries = list((worker / "groups").iterdir())
assert all(path.is_file() and not path.is_symlink() and path.name.endswith(".json") for path in entries)
assert {path.stem for path in entries} == set(assigned)
for entry in record["group_files"]:
    path = worker / "groups" / f'{entry["task_id"]}.json'
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    group = json.loads(path.read_text())
    assert len(group) == 8
    for sample_index, row in enumerate(group):
        assert row["environment"]["task_id"] == entry["task_id"]
        assert int(row["sequence"]) == order[entry["task_id"]] * 8 + sample_index
        assert int(row["sample"]["audit_record"]["sample_index"]) == sample_index
PY
}

write_worker_completion() {
  local assignment=$1 worker_out=$2 complete=$3 marker_quarantine=$4
  env PYTHONPATH= "$PYTHON_BIN" - "$assignment" "$worker_out" "$complete" "$marker_quarantine" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
assignment, worker, output, quarantine = map(Path, sys.argv[1:5])
assigned = [line for line in assignment.read_text().splitlines() if line]
assert len(assigned) == len(set(assigned)) == 300
groups = worker / "groups"
assert {path.stem for path in groups.glob("*.json")} == set(assigned)
files = [{"task_id": task_id, "sha256": hashlib.sha256((groups / f"{task_id}.json").read_bytes()).hexdigest()} for task_id in assigned]
record = {
    "schema_version": "vanilla-grpo-boundary-screen-s2-worker-v1", "status": "complete",
    "assignment_path": str(assignment.resolve()), "assignment_sha256": hashlib.sha256(assignment.read_bytes()).hexdigest(),
    "tasks": 300, "group_size": 8, "group_files": files,
}
encoded = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
quarantine.mkdir(parents=True, exist_ok=True)
temporary = quarantine / "worker_complete.json.next"
suffix = 1
while temporary.exists():
    temporary = quarantine / f"worker_complete.json.next.{suffix}"
    suffix += 1
temporary.write_bytes(encoded); os.replace(temporary, output)
PY
}

run_worker() {
  validate_static_inputs
  [[ -f "$S2_TASKS" && -f "$S2_COHORT_MANIFEST" && -f "$PLAN" ]] \
    || die 'S2 prepare must complete before workers'
  verify_s2_inputs || die 'prepared S2 input identity changed'
  local shard_name assignment worker_out complete status log gpu unexpected
  shard_name=$(printf 'shard-%02d-of-02' "$SHARD_INDEX")
  assignment=$ASSIGNMENTS_DIR/extra600.$shard_name.txt
  worker_out=$WORKERS_DIR/$shard_name
  complete=$worker_out/worker_complete.json
  status=$STATUS_DIR/worker-$shard_name.status
  log=$LOG_DIR/worker-$shard_name.log
  if [[ "$SHARD_INDEX" -eq 0 ]]; then gpu=$SCREEN_GPU0; else gpu=$SCREEN_GPU1; fi
  verify_plan_assignment "$assignment" || die "S2 assignment contract failed: $assignment"
  mkdir -p "$worker_out/groups" "$LOCK_DIR"
  exec 8>"$LOCK_DIR/worker-$shard_name.lock"
  flock -n 8 || die "another invocation owns S2 worker $shard_name"
  "$PYTHON_BIN" "$GROUP_RESUME_HELPER" \
    --tasks "$S2_TASKS" \
    --task-id-file "$assignment" \
    --groups-dir "$worker_out/groups" \
    --quarantine-dir "$QUARANTINE_DIR/$shard_name/groups" >>"$log" 2>&1 \
    || die "S2 worker partial-group resume preparation failed: $worker_out/groups"
  if [[ -f "$complete" ]]; then
    verify_worker_output "$assignment" "$worker_out" "$complete" || die 'existing worker completion failed verification'
    write_status "$status" complete "existing_verified=$complete"
    return
  fi
  unexpected=$(find "$worker_out" -mindepth 1 -maxdepth 1 ! -name groups -print -quit)
  [[ -z "$unexpected" ]] || die "unknown S2 worker output: $unexpected"
  if find "$worker_out/groups" -mindepth 1 -maxdepth 1 ! -type f -print -quit | grep -q .; then
    die "S2 worker groups contain non-file entries: $worker_out/groups"
  fi
  if find "$worker_out/groups" -mindepth 1 -maxdepth 1 -type f ! -name '*.json' -print -quit | grep -q .; then
    die "S2 worker groups contain unknown/partial files: $worker_out/groups"
  fi
  wait_for_gpu "$gpu" "$status"
  write_status "$status" generating "gpu=$gpu tasks=300 K8 resume_groups=true"
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
      --tasks "$S2_TASKS" \
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
      --scheduler static \
      --task-batch-size 1 >>"$log" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_owned_worker
  trap - EXIT INT TERM
  [[ "$code" -eq 0 ]] || { write_status "$status" failed "exit=$code log=$log"; return "$code"; }
  write_worker_completion "$assignment" "$worker_out" "$complete" "$QUARANTINE_DIR/$shard_name/markers"
  verify_worker_output "$assignment" "$worker_out" "$complete" || die 'new worker output failed verification'
  write_status "$status" complete "gpu=$gpu completion=$complete"
}

merge_worker_groups() {
  env PYTHONPATH= "$PYTHON_BIN" - "$S2_TASKS" "$WORKERS_DIR" "$POOL_OUT" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
tasks, workers, pool = map(Path, sys.argv[1:4])
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
sources = {}
for shard in range(2):
    name = f"shard-{shard:02d}-of-02"
    worker = workers / name
    record = json.loads((worker / "worker_complete.json").read_text())
    assert record["status"] == "complete" and record["tasks"] == 300
    for entry in record["group_files"]:
        source = worker / "groups" / f'{entry["task_id"]}.json'
        assert source.is_file() and not source.is_symlink()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["task_id"] not in sources
        sources[entry["task_id"]] = source
assert set(sources) == set(ids) and len(sources) == 600
destination = pool / "groups"
for task_id in ids:
    payload = sources[task_id].read_bytes()
    target = destination / f"{task_id}.json"
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"refusing non-identical S2 merged group: {target}")
    temporary = target.with_suffix(".json.next"); temporary.write_bytes(payload); os.replace(temporary, target)
entries = list(destination.iterdir())
assert all(path.is_file() and not path.is_symlink() and path.name.endswith(".json") for path in entries)
assert {path.stem for path in entries} == set(ids)
PY
}

verify_final_pool() {
  env PYTHONPATH= "$PYTHON_BIN" - "$S2_TASKS" "$POOL_OUT" "$WORKERS_DIR" \
    "$EXPECTED_SFT1_SHA256" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
tasks, pool, workers = map(Path, sys.argv[1:4])
expected_adapter, expected_tree, expected_version, expected_protocol, expected_prompt = sys.argv[4:9]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path, trajectories_path = pool / "manifest.pending.json", pool / "trajectories.jsonl"
for path in (tasks, manifest_path, trajectories_path):
    assert path.is_file() and not path.is_symlink(), path
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == "table-agent-fixed-rollout-pool-pending-v1"
assert manifest["tasks_sha256"] == sha(tasks)
assert manifest["adapter_sha256"] == expected_adapter
assert manifest["protocol_runtime_content_tree_sha256"] == expected_tree
assert manifest["protocol_version"] == expected_version and manifest["protocol_hash"] == expected_protocol
assert manifest["student_prompt_sha256"] == expected_prompt
for key, value in {
    "tasks": 600, "group_size": 8, "trajectories": 4800,
    "reward_mode": "result-only", "result_reward_profile": "binary",
    "temperature": 0.8, "top_p": 1.0, "max_steps": 30,
    "max_new_tokens": 2048, "max_context_tokens": 16384,
    "history_turns": 4, "enable_thinking": True, "seed": 20260812,
}.items(): assert manifest[key] == value, (key, manifest[key], value)
assert manifest["generation_scheduler"] == "static"
assert manifest["generation_task_batch_size"] == 1
assert manifest["generation_question_window"] is None
assert manifest["trajectories_sha256"] == sha(trajectories_path)
task_rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in task_rows]
rows = [json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()]
assert len(rows) == 4800 and [int(row["sequence"]) for row in rows] == list(range(4800))
assert collections.Counter(str(row["environment"]["task_id"]) for row in rows) == collections.Counter({task_id: 8 for task_id in ids})
expected_hashes = {}
for shard in range(2):
    record = json.loads((workers / f"shard-{shard:02d}-of-02/worker_complete.json").read_text())
    for entry in record["group_files"]:
        assert entry["task_id"] not in expected_hashes
        expected_hashes[entry["task_id"]] = entry["sha256"]
assert set(expected_hashes) == set(ids)
for task_id, expected in expected_hashes.items():
    assert sha(pool / "groups" / f"{task_id}.json") == expected
PY
}

verify_combined_selection() {
  local s2_audit=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$S1_AUDIT" "$S1_TASKS" "$s2_audit" "$S2_TASKS" \
    "$BOUNDARY_SELECTION_DIR" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
s1_audit_path, s1_tasks_path, s2_audit_path, s2_tasks_path, output = map(Path, sys.argv[1:6])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path = output / "boundary_cohort_manifest.json"
paths = {name: output / filename for name, filename in {
    "boundary": "boundary332.jsonl", "train": "train300.jsonl", "validation": "validation32.jsonl",
}.items()}
for path in (manifest_path, *paths.values()): assert path.is_file() and not path.is_symlink(), path
manifest_bytes = manifest_path.read_bytes(); manifest = json.loads(manifest_bytes)
assert manifest["schema_version"] == "policy-boundary-grpo-cohort-v1"
assert manifest["status"] == "frozen_boundary_training_cohort"
pools = manifest["screen_pools"]
assert len(pools) == 2
for pool, audit, tasks in zip(pools, (s1_audit_path, s2_audit_path), (s1_tasks_path, s2_tasks_path), strict=True):
    assert Path(pool["audit_path"]).resolve() == audit.resolve() and pool["audit_sha256"] == sha(audit)
    assert Path(pool["tasks_path"]).resolve() == tasks.resolve() and pool["tasks_sha256"] == sha(tasks)
selection = manifest["selection"]
assert selection["screen_stage"] == "S1+S2"
assert selection["seed"] == "qwen3-v26-boundary332-v1-20260812"
assert set(selection["acceptance_gates"]) == {
    "exact_boundary332", "mixed_at_least_360", "core_at_least_240",
    "unique_databases_at_least_50", "database_tv_at_most_0_15",
    "length_tv_at_most_0_10", "knowledge_tv_at_most_0_05",
}
assert all(selection["acceptance_gates"].values())
audits = [json.loads(path.read_text()) for path in (s1_audit_path, s2_audit_path)]
pool_counts = [{"mixed": audit["observed"]["mixed_boundary_groups"], "core": audit["observed"]["core_boundary_groups"]} for audit in audits]
assert selection["pool_eligible_counts"] == pool_counts
assert selection["eligible_mixed_groups"] == sum(row["mixed"] for row in pool_counts)
assert selection["eligible_core_groups"] == sum(row["core"] for row in pool_counts)
assert manifest["contract"] == {
    "boundary_records": 332, "train_records": 300, "validation_records": 32,
    "formal_training": {
        "optimizer_updates": 20, "prompts_per_update": 30, "group_size": 8,
        "train_passes": 2, "prompt_appearances": 600,
        "fresh_online_trajectories": 4800,
        "sampler": "trl-0.29-repeat-sampler-v1", "shuffle_dataset": True,
        "data_seed": 20260812,
        "task_order": "two deterministic data-seed shuffled passes",
        "per_pass_coverage": "each train300 identity exactly once",
        "reward_mode": "result-only", "result_reward_profile": "binary",
    },
    "validation": {
        "policy": "fresh initial-SFT1", "records": 32, "group_size": 8,
        "seed": 20260813, "screen_seed_must_differ": 20260812,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "gate": "same >=20/32 mixed probe gate",
        "runtime_contamination_allowed": False,
        "screen_trajectories_reused": False,
    },
    "primary_checkpoint": "final-step20-only",
}
def read_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def task_id(row): return str(row.get("example_id") or row.get("instance_id"))
source = {}
for path in (s1_tasks_path, s2_tasks_path):
    for row in read_jsonl(path):
        identifier = task_id(row); assert identifier not in source; source[identifier] = row
observed = {}
for name, count in (("boundary", 332), ("train", 300), ("validation", 32)):
    rows = read_jsonl(paths[name]); ids = [task_id(row) for row in rows]
    assert len(rows) == len(set(ids)) == count and all(source[identifier] == row for identifier, row in zip(ids, rows, strict=True))
    assert manifest["outputs"][name] == {"path": str(paths[name]), "sha256": sha(paths[name]), "records": count}
    assert manifest["task_ids"][name] == ids
    observed[name] = ids
assert set(observed["train"]).isdisjoint(observed["validation"])
assert set(observed["train"]) | set(observed["validation"]) == set(observed["boundary"])
group_maps = [{row["task_id"]: row for row in audit["groups"]} for audit in audits]
metadata = selection["selected_metadata"]
assert [row["task_id"] for row in metadata] == observed["boundary"]
for row in metadata:
    assert row["screen_pool"] in {"S1", "S2"}
    pool_index = 0 if row["screen_pool"] == "S1" else 1
    group = group_maps[pool_index][row["task_id"]]
    assert row["correct_count"] == group["correct_count"]
    assert row["uncertainty"] == group["uncertainty"]
    assert row["core_boundary"] == group["core_boundary"]
verification = {
    "schema_version": "policy-boundary-grpo-cohort-verification-v1", "status": "verified",
    "manifest": {"path": str(manifest_path.resolve()), "sha256": sha(manifest_path)},
    "outputs": {name: {"path": str(path.resolve()), "records": len(observed[name]), "sha256": sha(path)} for name, path in paths.items()},
}
encoded = (json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
verification_path = output / "boundary_selection_verification.json"
if verification_path.exists():
    assert verification_path.is_file() and not verification_path.is_symlink()
    assert verification_path.read_bytes() == encoded
else:
    temporary = verification_path.with_name(verification_path.name + ".next")
    temporary.write_bytes(encoded); os.replace(temporary, verification_path)
print(sha(manifest_path))
PY
}

finalize() {
  validate_static_inputs
  [[ -f "$PLAN" && -f "$S2_TASKS" && -f "$S2_COHORT_MANIFEST" ]] \
    || die 'S2 prepare must complete before finalize'
  verify_s2_inputs || die 'S2 inputs changed before finalize'
  mkdir -p "$STATUS_DIR" "$LOG_DIR" "$LOCK_DIR" "$POOL_OUT/groups" "$S2_AUDIT_DIR" "$BOUNDARY_SELECTION_DIR"
  exec 9>"$LOCK_DIR/control.lock"
  flock -n 9 || die "another S2 control operation owns $LOCK_DIR/control.lock"
  for shard in 0 1; do
    local name assignment complete worker
    name=$(printf 'shard-%02d-of-02' "$shard")
    assignment=$ASSIGNMENTS_DIR/extra600.$name.txt
    worker=$WORKERS_DIR/$name
    complete=$worker/worker_complete.json
    [[ -f "$complete" && ! -L "$complete" ]] || die "missing complete S2 worker: $name"
    verify_worker_output "$assignment" "$worker" "$complete" || die "S2 worker failed final verification: $name"
  done
  write_status "$FINALIZE_STATUS" merging 'strict hash merge of two fresh S2 worker shards'
  "$PYTHON_BIN" "$GROUP_RESUME_HELPER" \
    --tasks "$S2_TASKS" \
    --groups-dir "$POOL_OUT/groups" \
    --quarantine-dir "$QUARANTINE_DIR/final-pool/groups" >>"$LOG_DIR/finalize.log" 2>&1 \
    || die 'S2 final-pool partial-group resume preparation failed'
  merge_worker_groups || die 'strict S2 group merge failed'
  write_status "$FINALIZE_STATUS" finalizing 'generator --finalize-only; no model loaded'
  env \
    HF_HUB_OFFLINE=1 \
    TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
    PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
    "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
      --model-path "$MODEL_PATH" \
      --adapter-path "$SFT1_ADAPTER" \
      --tasks "$S2_TASKS" \
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
      --scheduler static \
      --task-batch-size 1 \
      --finalize-only >>"$LOG_DIR/finalize.log" 2>&1 \
    || die "generator --finalize-only failed; see $LOG_DIR/finalize.log"
  verify_final_pool || die 'final S2 pool identity/coverage verification failed'

  local s2_audit=$S2_AUDIT_DIR/boundary_screen_audit.json s2_audit_sha
  if [[ -e "$s2_audit" ]]; then
    [[ -f "$s2_audit" && ! -L "$s2_audit" ]] || die 'existing S2 audit is not a regular file'
  else
    write_status "$FINALIZE_STATUS" auditing 'running frozen boundary auditor on S2 pool'
    env PYTHONPATH="$TRAIN_RUNTIME/src/rl" "$PYTHON_BIN" "$BOUNDARY_AUDITOR" \
      --manifest "$POOL_OUT/manifest.pending.json" \
      --trajectories "$POOL_OUT/trajectories.jsonl" \
      --tasks "$S2_TASKS" \
      --output-dir "$S2_AUDIT_DIR" >>"$LOG_DIR/s2_audit.log" 2>&1 \
      || die "S2 boundary audit failed; see $LOG_DIR/s2_audit.log"
  fi
  s2_audit_sha=$(verify_screen_audit "$s2_audit" "$S2_TASKS" - S2 \
    "$POOL_OUT/manifest.pending.json" "$POOL_OUT/trajectories.jsonl") \
    || die 'S2 boundary audit/input hash binding failed'
  verify_screen_audit "$S1_AUDIT" "$S1_TASKS" "$FROZEN_S1_AUDIT_SHA256" S1 - - >/dev/null \
    || die 'S1 audit/input bytes changed before combined selection'

  write_status "$FINALIZE_STATUS" selecting "recoverable S1+S2 publication s2_audit_sha256=$s2_audit_sha no_S3=true"
  if ! env PYTHONPATH="$TRAIN_RUNTIME/src/rl" "$PYTHON_BIN" "$SELECTION_RECOVERY_HELPER" \
    --selector "$BOUNDARY_SELECTOR" \
    --screen-audit "$S1_AUDIT" \
    --screen-audit "$s2_audit" \
    --tasks "$S1_TASKS" \
    --tasks "$S2_TASKS" \
    --output-dir "$BOUNDARY_SELECTION_DIR" \
    --quarantine-dir "$QUARANTINE_DIR/boundary-selection" \
    --seed qwen3-v26-boundary332-v1-20260812 >>"$LOG_DIR/boundary_selection.log" 2>&1; then
    write_status "$FINALIZE_STATUS" selection_blocked 'S1+S2 frozen selector failed; no S3 and no threshold relaxation'
    printf 'blocked: S1+S2 selector did not admit boundary332; no S3 will start; see %s\n' "$LOG_DIR/boundary_selection.log" >&2
    return 4
  fi
  local manifest_sha
  manifest_sha=$(verify_combined_selection "$s2_audit") \
    || die 'combined boundary332/train300/validation32 manifest verification failed'
  write_status "$FINALIZE_STATUS" complete \
    "selection=$BOUNDARY_SELECTION_DIR manifest_sha256=$manifest_sha s2_audit_sha256=$s2_audit_sha"
  printf 'complete: export BOUNDARY_SELECTION_DIR=%s FROZEN_BOUNDARY_MANIFEST_SHA256=%s\n' \
    "$BOUNDARY_SELECTION_DIR" "$manifest_sha"
}

case "$MODE" in
  dry-run) dry_run ;;
  prepare) prepare ;;
  worker) run_worker ;;
  finalize) finalize ;;
esac
