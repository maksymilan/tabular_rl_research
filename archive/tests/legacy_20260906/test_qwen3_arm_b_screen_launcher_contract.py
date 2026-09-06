from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "src/rl/experiments/"
    "run_qwen3_8b_atomic_v26_vanilla_grpo_arm_b_screen_table_rl.sh"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _launcher_pin(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}=([0-9a-f]{{64}})$",
        LAUNCHER.read_text(),
        flags=re.MULTILINE,
    )
    assert match is not None, f"missing literal launcher pin: {name}"
    return match.group(1)


def _run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        text=True,
        capture_output=True,
        check=False,
        env=merged,
    )


def test_default_is_read_only_and_all_stages_need_explicit_modes(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    run_root = tmp_path / "must-not-exist"
    result = _run(env={"RUN_ROOT": str(run_root)})
    assert result.returncode == 0, result.stderr
    assert "Arm B staged screen dry-run" in result.stdout
    assert "no files written, no GPU inspected" in result.stdout
    assert not run_root.exists()

    invalid = _run("run-all")
    assert invalid.returncode == 2
    text = LAUNCHER.read_text()
    for mode in (
        "prepare-f1",
        "worker-f1",
        "finalize-f1",
        "prepare-f2",
        "worker-f2",
        "finalize-f2",
        "prepare-f3",
        "worker-f3",
        "finalize-f3",
    ):
        assert mode in text
    assert "explicit prepare-f2" in text
    assert "explicit prepare-f3" in text


def test_wide_preparer_uses_an_isolated_pinned_public_selector_support_root() -> None:
    text = LAUNCHER.read_text()
    assert (
        "PUBLIC_SELECTOR_ROOT=${PUBLIC_SELECTOR_ROOT:-$TRAIN_RUNTIME/support/public-selector-v1}"
        in text
    )
    assert (
        'require_sha "$PUBLIC_SELECTOR_ROOT/src/eval/select_bird_train_baseline.py" '
        '"$EXPECTED_PUBLIC_SELECTOR_SHA256"'
    ) in text
    assert (
        'PYTHONPATH="$TRAIN_RUNTIME:$PUBLIC_SELECTOR_ROOT" "$PYTHON_BIN" '
        '"$WIDE_PREPARER"'
    ) in text
    assert 'require_sha "$PROJECT_ROOT/src/eval/select_bird_train_baseline.py"' not in text
    assert "public selector support root must contain exactly" in text
    assert '--bind public-selector "$PUBLIC_SELECTOR_ROOT/src/eval/select_bird_train_baseline.py"' in text
    assert '--bind wide-preparer "$WIDE_PREPARER"' in text


def test_preregistered_three_stage_counts_seeds_and_gates_are_exact() -> None:
    text = LAUNCHER.read_text()
    assert "STAGE_RECORDS=3000; STAGE_GROUP_SIZE=8; STAGE_SEED=20260814" in text
    assert "STAGE_RECORDS=640; STAGE_GROUP_SIZE=16; STAGE_SEED=20260815" in text
    assert "STAGE_RECORDS=64; STAGE_GROUP_SIZE=16; STAGE_SEED=20260816" in text
    for literal in (
        "F1=wide3000 fresh initial-SFT1 K8 seed20260814; require >=640 whole-group-clean mixed; no replacement",
        "F2=confirmation640 fresh initial-SFT1 K16 seed20260815; require >=384 whole-group-clean c=2..14; freeze 384 -> train320 + validation64",
        "F3=validation64 fresh initial-SFT1 K16 seed20260816; require 1024/1024 clean, mixed>=56, core>=48; no replacement",
        '"screen_trajectories_reused":False',
        '"primary_checkpoint"]=="final-step32-only"',
        '"fresh_online_trajectories"]==10240',
    ):
        assert literal in text
    assert "no_resample=true" in text
    assert "no_replacement=true" in text


def test_arm_b_activation_is_recomputed_from_actual_arm_a_artifacts() -> None:
    text = LAUNCHER.read_text()
    assert "ARM_A_TRIGGER_MODE must be explicitly readiness or validation" in text
    assert "readiness fallback requires exactly the actual S1 and S2 screen pools" in text
    for option in (
        "--screen-audit",
        "--expected-screen-audit-sha256",
        "--screen-tasks",
        "--expected-screen-tasks-sha256",
        "--arm-a-selection-manifest",
        "--expected-arm-a-selection-manifest-sha256",
        "--arm-a-validation-manifest",
        "--expected-arm-a-validation-manifest-sha256",
        "--arm-a-validation-tasks",
        "--expected-arm-a-validation-tasks-sha256",
        "--arm-a-validation-trajectories",
        "--expected-arm-a-validation-trajectories-sha256",
    ):
        assert option in text
    assert "validate_arm_a_not_started" in text
    assert '[[ ! -e "$ARM_A_FORMAL_TRAIN_DIR" ]]' in text
    assert '[[ ! -e "$ARM_A_FULL_DEV_DIR" ]]' in text
    assert "pgrep" not in text and "pkill" not in text and "killall" not in text
    assert re.search(r"\bssh\b", text) is None


def test_gpu_mapping_is_explicit_parameterized_and_owned_without_global_kills() -> None:
    text = LAUNCHER.read_text()
    assert "F1_GPU_MAP=${F1_GPU_MAP:-}" in text
    assert "F2_GPU_MAP=${F2_GPU_MAP:-}" in text
    assert "F3_GPU_MAP=${F3_GPU_MAP:-}" in text
    assert "GPU map must be explicitly supplied" in text
    assert "GPU map contains duplicate" in text
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in text
    assert 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"' in text
    assert '"$used" -le "$GPU_IDLE_MAX_MIB"' in text
    assert '"$RUN_ROOT/locks/physical-gpu-$gpu.lock"' in text
    assert "setsid env" in text
    assert 'kill -TERM -- "-$worker_pgid"' in text
    assert 'kill -KILL -- "-$worker_pgid"' in text
    # The source must not permanently ban one device; this run simply starts none.
    assert "GPU5 is forbidden" not in text


def test_plan_sha_and_stage_specific_contract_are_reverified_every_active_step() -> None:
    text = LAUNCHER.read_text()
    assert "verify_stage_plan" in text
    assert '"$STAGE_MANAGER" verify-plan "${STAGE_PLAN_ARGS[@]}"' in text
    assert 'require_sha "$STAGE_PLAN" "$FROZEN_STAGE_PLAN_SHA"' in text
    for command in (
        "complete-worker",
        "verify-worker",
        "prepare-pool-transaction",
        "verify-pool",
        "publish-pool",
    ):
        assert command in text
    assert text.count('--expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA"') >= 6
    for variable in (
        "STAGE_RECORDS",
        "STAGE_GROUP_SIZE",
        "STAGE_SEED",
        "STAGE_SHARDS",
        "STAGE_GPU_MAP",
        "STAGE_TASKS",
        "STAGE_TASK_MANIFEST",
        "STAGE_ACTIVATION_SHA",
    ):
        assert variable in text


def test_workers_publish_only_atomic_groups_before_transactional_finalize() -> None:
    text = LAUNCHER.read_text()
    worker = text.split("run_worker() {", 1)[1].split("\n}\n\nassemble_and_publish_pool", 1)[0]
    finalize = text.split("assemble_and_publish_pool() {", 1)[1].split("\n}\n\nrun_f1_audit", 1)[0]
    assert "--no-finalize" in worker
    assert "--finalize-only" not in worker
    assert "prepare_boundary_screen_group_resume.py" in text
    assert "complete-worker" in worker
    assert "prepare-pool-transaction" in finalize
    assert 'STAGE_TRANSACTION=$STAGE_ROOT/transactions/pool.next' in text
    assert 'STAGE_POOL=$STAGE_ROOT/pool' in text
    assert "--finalize-only" in finalize
    assert finalize.index("verify-pool") < finalize.index("publish-pool")
    assert "canonical pool absent until transaction verifies" in text


def test_f1_f2_failure_artifacts_stop_without_next_stage_outputs() -> None:
    text = LAUNCHER.read_text()
    f1 = text.split("run_f1_audit() {", 1)[1].split("\n}\n\nrun_f2_selector", 1)[0]
    f2 = text.split("run_f2_selector() {", 1)[1].split("\n}\n\nrun_f3_audit", 1)[0]
    assert '[[ "$code" -eq 0 || "$code" -eq 1 ]]' in f1
    assert 'write_status "$STAGE_STATUS/finalize.status" stop_arm_b' in f1
    assert "no F2 was started" in f1
    assert '[[ "$code" -eq 0 || "$code" -eq 1 ]]' in f2
    assert "core_boundary_groups\"]<384" in f2
    assert "F2 failed gate left forbidden selection/training artifacts" in f2
    assert "no F3/training was started" in f2


def test_f3_is_hash_pinned_matches_validator_and_publishes_canonical_trigger() -> None:
    text = LAUNCHER.read_text()
    assert (
        "EXPECTED_K16_VALIDATION_AUDITOR_SHA256="
        "1b3e52770519d663cb6eb24f10fccba0fae974dde6247e20c5e8063fa66e90e0"
    ) in text
    assert "F3 auditor interface has not been hash-frozen" not in text
    assert '(observed["tasks"],observed["group_size"],observed["trajectories"])==(64,16,1024)' in text
    assert 'observed["usable_groups"]==64' in text
    assert 'observed["contaminated_groups"]==0' in text
    assert '"eligible_trajectories"]==1024' in text
    assert '"mixed_boundary_groups"]>=56' in text
    assert '"core_boundary_groups"]>=48' in text
    assert 'all(audit["checks"].values())' in text
    assert (
        "EXPECTED_CONFIRMATORY_TRIGGER_PREPARER_SHA256="
        "554fe25b39fe7c012a73b8d4045762476347bfd6fbcc70355f98f4cb8bdd7b98"
    ) in text
    assert (
        "EXPECTED_ARM_B_TRAINING_INPUT_VALIDATOR_SHA256="
        "a416b9490e8c6287c7f05fff5c2c314c528b8524dddbd782c5e25ed6eb7125e0"
    ) in text
    assert '--bind arm-b-training-input-validator "$ARM_B_TRAINING_INPUT_VALIDATOR"' in text
    assert 'CONFIRMATORY_TRIGGER=${CONFIRMATORY_TRIGGER:-$F3_AUDIT_DIR/confirmatory_arm_trigger.json}' in text
    assert '"$PYTHON_BIN" "$CONFIRMATORY_TRIGGER_PREPARER"' in text
    assert '--source-cohort "$F1_COHORT_MANIFEST"' in text
    assert '--selection-manifest "$F2_SELECTION_MANIFEST"' in text
    assert '--train320 "$train320"' in text
    assert '--validation64-audit "$F3_AUDIT"' in text
    assert "verify_confirmatory_trigger" in text
    assert 'trigger["admitted_arm"]=="arm_b"' in text
    assert 'trigger["confirmatory_arm_count"]==1' in text
    assert 'trigger["full_dev_policy"]=="exactly-one-admitted-arm"' in text
    assert "failed F3 gate left a forbidden confirmatory trigger artifact" in text
    assert "no training was started" in text


def test_local_stage_sources_match_every_frozen_launcher_pin() -> None:
    sources = {
        "EXPECTED_WIDE_PREPARER_SHA256": ROOT
        / "src/rl/prepare_vanilla_grpo_arm_b_wide3000.py",
        "EXPECTED_WIDE_AUDITOR_SHA256": ROOT
        / "src/rl/diagnostics/audit_vanilla_grpo_arm_b_wide_k8_screen.py",
        "EXPECTED_K16_SELECTOR_SHA256": ROOT
        / "src/rl/select_vanilla_grpo_arm_b_k16.py",
        "EXPECTED_K16_VALIDATION_AUDITOR_SHA256": ROOT
        / "src/rl/diagnostics/audit_vanilla_grpo_arm_b_k16_validation.py",
        "EXPECTED_CONFIRMATORY_TRIGGER_PREPARER_SHA256": ROOT
        / "src/rl/diagnostics/prepare_vanilla_grpo_confirmatory_arm_trigger.py",
        "EXPECTED_ARM_B_TRAINING_INPUT_VALIDATOR_SHA256": ROOT
        / "src/rl/diagnostics/validate_arm_b_training_inputs.py",
        "EXPECTED_STAGE_MANAGER_SHA256": ROOT
        / "src/rl/diagnostics/manage_vanilla_grpo_arm_b_screen_stage.py",
        "EXPECTED_GROUP_RESUME_HELPER_SHA256": ROOT
        / "src/rl/diagnostics/prepare_boundary_screen_group_resume.py",
        "EXPECTED_ARM_B_LIBRARY_SHA256": ROOT / "src/rl/vanilla_grpo_arm_b.py",
    }
    for pin, path in sources.items():
        assert _launcher_pin(pin) == _sha256(path), f"stale {pin} for {path}"


def test_embedded_python_is_syntactically_valid() -> None:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=re.DOTALL)
    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER}:heredoc-{index}", "exec")
