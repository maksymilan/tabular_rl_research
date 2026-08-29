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
    / "src/rl/experiments/run_qwen3_8b_atomic_v26_boundary300_vanilla_grpo_table_rl.sh"
)
CONFIG = (
    ROOT
    / "src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_boundary300.yaml"
)
AUDITOR = ROOT / "src/rl/diagnostics/audit_vanilla_grpo_boundary_validation.py"
SELECTOR = ROOT / "src/rl/select_policy_boundary_grpo_tasks.py"
SCREEN_LAUNCHER = (
    ROOT / "src/rl/experiments/run_qwen3_8b_atomic_v26_boundary_screen_table_rl.sh"
)


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        text=True,
        capture_output=True,
        env=merged,
        check=False,
    )


def test_default_is_read_only_dry_run_and_active_modes_need_frozen_digest(
    tmp_path: Path,
) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    run_root = tmp_path / "must-not-exist"
    result = _run(env={"RUN_ROOT": str(run_root)})
    assert result.returncode == 0, result.stderr
    assert "no files written, no GPU inspected" in result.stdout
    assert "validate -> train" in result.stdout
    assert not run_root.exists()

    active_root = tmp_path / "active"
    result = _run("validate", env={"RUN_ROOT": str(active_root)})
    assert result.returncode == 3
    assert "FROZEN_BOUNDARY_MANIFEST_SHA256" in result.stderr


def test_boundary_artifact_defaults_and_code_sources_are_dedicated() -> None:
    text = LAUNCHER.read_text()
    assert "MODE=${1:-dry-run}" in text
    assert "dry-run|validate|train" in text
    assert "rl_runtime_qwen3_8b_v26_boundary300_grpo_20260812" in text
    assert "qwen3_8b_atomic_v26_boundary_screen_s1_2gpu_20260812/boundary_selection" in text
    assert "boundary_cohort_manifest.json" in text
    assert "boundary_selection_verification.json" in text
    assert "boundary332.jsonl" in text
    assert "train300.jsonl" in text
    assert "validation32.jsonl" in text
    assert "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl" not in text
    assert "probe_first32" not in text
    assert 'audit["status"]["probe_admitted"]' not in text


def test_validation_is_fresh_independent_and_zero_contamination() -> None:
    text = LAUNCHER.read_text()
    assert "VALIDATION_SEED=20260813" in text
    assert '[[ "$VALIDATION_SEED" != 20260812 ]]' in text
    assert "audit_vanilla_grpo_boundary_validation.py" in text
    assert '--selection-manifest "$BOUNDARY_MANIFEST"' in text
    assert '--expected-selection-manifest-sha256 "$FROZEN_BOUNDARY_MANIFEST_SHA256"' in text
    assert '--tasks "$VALIDATION_TASKS"' in text
    assert '--group-size 8' in text
    assert '--seed "$VALIDATION_SEED"' in text
    assert 'audit["observed"]["eligible_trajectories"] == 256' in text
    assert 'audit["observed"]["mixed_outcome_groups"] >= 20' in text
    assert 'audit["observed"]["tokenization_warnings"] == 0' in text
    assert 'audit["checks"]["no_runtime_contamination"] is True' in text
    train = text.split("run_train() {", 1)[1]
    assert train.lstrip().startswith("verify_validation_gate")


def test_training_is_binary_boundary300_two_pass_step20_only() -> None:
    text = LAUNCHER.read_text()
    assert "qwen3_8b_atomic_v26_vanilla_grpo_boundary300.yaml" in text
    assert '--optimizer-steps 20' in text
    assert '--prompts-per-update 30' in text
    assert '--group-size 8' in text
    assert '--kl-beta 0' in text
    assert '--save-steps 2' in text
    assert '--save-total-limit 1' in text
    assert re.search(r"--save-steps 20(?:\s|$)", text) is None
    assert 'checkpoints == ["checkpoint-20"]' in text
    assert 'len(rollouts) == 4800' in text
    assert 'set(block_counts.values()) == {8}' in text
    assert 'counts == collections.Counter({index: 8 for index in indices})' in text
    assert 'manifest["records"] == manifest["expected_records"] == 300' in text
    assert 'manifest["optimizer_steps"] == 20' in text
    assert 'manifest["save_steps"] == 2' in text
    assert 'manifest["save_total_limit"] == 1' in text
    assert 'manifest["policy_reduction"] == "trajectory_token_mean"' in text
    assert 'manifest["result_reward_profile"] == "binary"' in text


def test_pinned_trl_repeat_sampler_is_proved_as_two_complete_seeded_passes() -> None:
    text = LAUNCHER.read_text()
    for snippet in (
        'importlib.metadata.version("trl").startswith("0.29.")',
        "from trl.trainer.utils import RepeatSampler",
        'inspect.signature(GRPOConfig).parameters["shuffle_dataset"].default',
        "mini_repeat_count=8",
        "batch_size=30",
        "repeat_count=1",
        "shuffle=True",
        "seed=20260812",
        "first_run == two_passes()",
        "len(sampled) == 2400",
        "set(collapsed) == set(range(300))",
    ):
        assert snippet in text
    assert "train300 JSONL order repeated exactly twice" not in text
    assert "two deterministic data-seed shuffled passes" in text
    assert "each train300 identity exactly once" in text


def test_source_hashes_are_literal_current_and_no_placeholder_can_pass() -> None:
    text = LAUNCHER.read_text()
    assert re.search(r"__[A-Z0-9_]+__", text) is None
    for path, variable in (
        (CONFIG, "EXPECTED_CONFIG_SHA256"),
        (AUDITOR, "EXPECTED_VALIDATION_AUDITOR_SHA256"),
        (SELECTOR, "EXPECTED_BOUNDARY_SELECTOR_SHA256"),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"{variable}={digest}" in text
    assert '[[ "$FROZEN_BOUNDARY_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]' in text
    assert 'sha(manifest_path) == expected_manifest_sha' in text


def test_selector_hash_change_is_propagated_to_screen_launcher_pin() -> None:
    digest = hashlib.sha256(SELECTOR.read_bytes()).hexdigest()
    screen = SCREEN_LAUNCHER.read_text()
    assert f"EXPECTED_BOUNDARY_SELECTOR_SHA256={digest}" in screen
    assert "two deterministic data-seed shuffled passes" in screen
    assert '"seed": 20260813' in screen
    assert '"runtime_contamination_allowed": False' in screen


def test_launcher_owns_only_its_spawned_process_groups() -> None:
    text = LAUNCHER.read_text()
    assert "setsid env" in text
    assert 'kill -TERM -- "-$pgid"' in text
    assert 'kill -KILL -- "-$pgid"' in text
    assert "wait_for_gpu" in text
    assert "no process will be stopped" in text
    assert "pgrep" not in text
    assert "pkill" not in text
    assert re.search(r"\bssh\b", text) is None


def test_periodic_checkpoint_resume_is_exact_and_rollout_prefix_safe() -> None:
    text = LAUNCHER.read_text()
    runner = ROOT / "src/rl/frameworks/trl/run_transition_grpo.py"
    preparer = ROOT / "src/rl/diagnostics/prepare_vanilla_grpo_resume.py"
    assert f"EXPECTED_RUN_TRANSITION_GRPO_SHA256={hashlib.sha256(runner.read_bytes()).hexdigest()}" in text
    assert f"EXPECTED_RESUME_PREPARER_SHA256={hashlib.sha256(preparer.read_bytes()).hexdigest()}" in text
    assert 'prepare_vanilla_grpo_resume.py' in text
    assert '--expected-records 300' in text
    assert '--optimizer-steps 20' in text
    assert '--prompts-per-update 30' in text
    assert '--group-size 8' in text
    assert '--save-steps 2' in text
    assert '0 < step < 20 and step % 2 == 0' in text
    assert 'record["rollouts"]["retained_rows"] == step * 240' in text
    assert 'resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")' in text
