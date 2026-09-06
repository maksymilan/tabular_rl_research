from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "src/rl/experiments/run_qwen3_8b_atomic_v26_boundary_screen_continuation.sh"
)


def test_continuation_launcher_is_syntax_valid_and_dry_run_is_read_only() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], check=True, capture_output=True, text=True
    )
    assert "no files written, no GPU inspected" in result.stdout


def test_continuation_is_exact_frozen_s1_worker_only() -> None:
    text = LAUNCHER.read_text()
    for value in (
        "--no-finalize",
        "--group-size 8",
        "--temperature 0.8",
        "--top-p 1",
        "--max-steps 30",
        "--max-new-tokens 2048",
        "--max-context-tokens 16384",
        "--history-turns 4",
        "--enable-thinking",
        "--seed 20260812",
        "--scheduler dynamic",
        "--question-window 4",
    ):
        assert value in text
    assert "finalize-only" not in text
    assert "kill " not in text and "pkill" not in text


def test_continuation_pins_dynamic_assignment_and_all_identity_sources() -> None:
    text = LAUNCHER.read_text()
    assert 'require_sha "$ASSIGNMENT" "$ASSIGNMENT_SHA256"' in text
    assert 'assert set(selected) <= set(ids[32:])' in text
    assert 'digest.hexdigest() == expected_tree' in text
    assert 'len(files) == 101' not in text
    expected = {
        "src/rl/fixed_pool/generate_fixed_rollout_pool.py": "EXPECTED_GENERATOR_SHA256",
        "src/rl/frameworks/trl/rollout.py": "EXPECTED_ROLLOUT_SHA256",
        "src/rl/rollout_scoring.py": "EXPECTED_SCORING_SHA256",
        "src/rl/task_loader.py": "EXPECTED_TASK_LOADER_SHA256",
        "src/rl/tool_environment_v26.py": "EXPECTED_TOOL_ENV_V26_SHA256",
    }
    for relative, variable in expected.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert f"{variable}={digest}" in text


def test_continuation_never_reuses_a_busy_gpu_or_escapes_owned_root() -> None:
    text = LAUNCHER.read_text()
    assert "used <= 512" in text
    assert "no process will be stopped" in text
    assert 'case "$OUTPUT_DIR" in "$RUN_ROOT"/*)' in text
    assert 'flock -n 9' in text
