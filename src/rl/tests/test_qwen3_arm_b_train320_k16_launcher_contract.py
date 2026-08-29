from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from src.rl.experiment_config import RLExperimentConfig


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "src/rl/experiments/run_qwen3_8b_atomic_v26_arm_b_train320_k16_vanilla_grpo_table_rl.sh"
CONFIG = ROOT / "src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_arm_b_train320_k16.yaml"
VALIDATOR = ROOT / "src/rl/diagnostics/validate_arm_b_training_inputs.py"
SHARED_CONTRACT = ROOT / "src/rl/vanilla_grpo_arm_b.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_arm_b_config_is_exact_binary_k16_two_pass_contract() -> None:
    defaults = RLExperimentConfig.load(CONFIG).argparse_defaults(ROOT)
    for field, expected in {
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "expected_records": 320,
        "optimizer_steps": 32,
        "prompts_per_update": 20,
        "group_size": 16,
        "ppo_iterations": 1,
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
        "enable_thinking": True,
        "expected_protocol_version": "version26",
        "expected_protocol_hash": "4da19387399bd3a5",
    }.items():
        assert defaults[field] == expected, (field, defaults[field], expected)


def test_launcher_default_is_read_only_dry_run_and_active_modes_need_all_hashes(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    run_root = tmp_path / "must-not-exist"
    result = _run(env={"RUN_ROOT": str(run_root)})
    assert result.returncode == 0, result.stderr
    assert "no files written, no GPU inspected" in result.stdout
    assert "steps 4..28 are diagnostic/resume-only and forbidden for evaluation" in result.stdout
    assert not run_root.exists()

    result = _run("preflight", env={"RUN_ROOT": str(tmp_path / "active")})
    assert result.returncode == 3
    assert "FROZEN_ARM_B_SELECTION_MANIFEST_SHA256" in result.stderr
    assert not (tmp_path / "active").exists()


def test_launcher_binds_selection_train320_and_validation64_admission() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for snippet in (
        "arm_b_selection_manifest.json",
        "train320.jsonl",
        "arm_b_k16_validation_audit.json",
        "FROZEN_ARM_B_SELECTION_MANIFEST_SHA256",
        "FROZEN_TRAIN320_SHA256",
        "FROZEN_VALIDATION64_AUDIT_SHA256",
        '--selection-manifest "$ARM_B_SELECTION_MANIFEST"',
        '--train320 "$TRAIN320_TASKS"',
        '--validation64-audit "$VALIDATION64_AUDIT"',
    ):
        assert snippet in text
    assert "train600" not in text
    assert "boundary300" not in text


def test_training_and_resume_are_exact_save4_limit1_final32_only() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for snippet in (
        "mini_repeat_count=16",
        "batch_size=20",
        "seed=20260812",
        "len(sampled) == 5120",
        "set(collapsed) == set(range(320))",
        '--optimizer-steps 32',
        '--prompts-per-update 20',
        '--group-size 16',
        '--kl-beta 0',
        '--save-steps 4',
        '--save-total-limit 1',
        'checkpoints == ["checkpoint-32"]',
        "len(rollouts) == 10240",
        "0 < step < 32 and step % 4 == 0",
        'resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")',
        '"intermediate_steps": [4, 8, 12, 16, 20, 24, 28]',
        '"intermediate_evaluation_allowed": False',
        '"primary_checkpoint": "checkpoint-32"',
    ):
        assert snippet in text
    assert re.search(r"--save-steps 32(?:\s|$)", text) is None


def test_launcher_pins_current_config_validator_and_owns_only_child_groups() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert f"EXPECTED_CONFIG_SHA256={_sha(CONFIG)}" in text
    assert f"EXPECTED_INPUT_VALIDATOR_SHA256={_sha(VALIDATOR)}" in text
    assert f"EXPECTED_ARM_B_SHARED_CONTRACT_SHA256={_sha(SHARED_CONTRACT)}" in text
    assert 'env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$INPUT_VALIDATOR"' in text
    assert 'require_sha "$ARM_B_SHARED_CONTRACT" "$EXPECTED_ARM_B_SHARED_CONTRACT_SHA256"' in text
    assert re.search(r"__[A-Z0-9_]+__", text) is None
    assert "setsid env" in text
    assert 'kill -TERM -- "-$pgid"' in text
    assert 'kill -KILL -- "-$pgid"' in text
    assert "no process will be stopped" in text
    for forbidden in ("ssh ", "scp ", "pgrep", "pkill"):
        assert forbidden not in text


def test_validator_imports_only_from_explicit_runtime_when_cwd_is_isolated(
    tmp_path: Path,
) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--selection-manifest" in completed.stdout
