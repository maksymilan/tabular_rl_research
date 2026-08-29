from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.rl.diagnostics import manage_vanilla_grpo_confirmatory_admission as admission


HELPER = Path(admission.__file__).resolve()


def test_claim_is_persistent_and_same_arm_idempotent(tmp_path: Path) -> None:
    marker = (tmp_path / "one-arm.json").resolve()
    first = admission.claim(admission_path=marker, arm="arm_a")
    before = marker.read_bytes()
    before_stat = marker.stat()

    second = admission.claim(admission_path=marker, arm="arm_a")
    checked = admission.verify(admission_path=marker, arm="arm_a")

    assert first["result_status"] == "admission_claimed"
    assert second["result_status"] == "existing_admission_verified"
    assert checked["result_status"] == "admission_verified"
    assert marker.read_bytes() == before
    assert marker.stat().st_ino == before_stat.st_ino
    assert marker.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert json.loads(before) == {
        "admission_path": str(marker),
        "admitted_arm": "arm_a",
        "lock_path": str(marker) + ".lock",
        "policy": admission.POLICY,
        "schema_version": admission.SCHEMA_VERSION,
        "status": admission.MARKER_STATUS,
    }


def test_other_arm_is_permanently_rejected_after_lock_release(tmp_path: Path) -> None:
    marker = (tmp_path / "one-arm.json").resolve()
    admission.claim(admission_path=marker, arm="arm_b")
    before = marker.read_bytes()

    with pytest.raises(admission.AdmissionError, match="permanently admitted to arm_b"):
        admission.claim(admission_path=marker, arm="arm_a")
    with pytest.raises(admission.AdmissionError, match="permanently admitted to arm_b"):
        admission.verify(admission_path=marker, arm="arm_a")

    assert marker.read_bytes() == before
    assert admission.verify(admission_path=marker, arm="arm_b")["admitted_arm"] == "arm_b"


def test_opposite_concurrent_cli_claims_have_exactly_one_winner(tmp_path: Path) -> None:
    marker = (tmp_path / "one-arm.json").resolve()
    base = [sys.executable, str(HELPER), "claim", "--admission", str(marker), "--arm"]
    processes = [
        subprocess.Popen(
            [*base, arm], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for arm in ("arm_a", "arm_b")
    ]
    completed = [process.communicate(timeout=20) for process in processes]
    returncodes = [process.returncode for process in processes]

    assert sorted(returncodes) == [0, 2], completed
    winner_index = returncodes.index(0)
    winner = ("arm_a", "arm_b")[winner_index]
    assert "permanently admitted" in completed[1 - winner_index][1]
    assert admission.verify(admission_path=marker, arm=winner)["admitted_arm"] == winner
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(schema_version="wrong"), "schema/path contract"),
        (lambda value: value.update(admission_path="/wrong/path"), "schema/path contract"),
        (lambda value: value.update(extra="forbidden"), "unexpected or missing"),
    ],
)
def test_verify_fails_closed_on_noncanonical_marker(
    tmp_path: Path, mutation, message: str
) -> None:
    marker = (tmp_path / "one-arm.json").resolve()
    admission.claim(admission_path=marker, arm="arm_a")
    value = json.loads(marker.read_bytes())
    mutation(value)
    marker.chmod(0o600)
    marker.write_text(json.dumps(value, sort_keys=True) + "\n")
    marker.chmod(0o444)

    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify(admission_path=marker, arm="arm_a")


def test_verify_rejects_missing_writable_and_symlink_markers(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.json").resolve()
    with pytest.raises(admission.AdmissionError, match="does not exist"):
        admission.verify(admission_path=missing, arm="arm_a")

    writable = (tmp_path / "writable.json").resolve()
    admission.claim(admission_path=writable, arm="arm_a")
    writable.chmod(0o644)
    with pytest.raises(admission.AdmissionError, match="read-only"):
        admission.verify(admission_path=writable, arm="arm_a")

    target = tmp_path / "target.json"
    target.write_text("{}\n")
    target.chmod(0o444)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(admission.AdmissionError, match="must not be a symlink"):
        admission.verify(admission_path=symlink.absolute(), arm="arm_a")


def test_custom_lock_is_bound_and_must_be_sibling(tmp_path: Path) -> None:
    marker = (tmp_path / "one-arm.json").resolve()
    lock = (tmp_path / "shared.lock").resolve()
    admission.claim(admission_path=marker, lock_path=lock, arm="arm_b")
    assert admission.verify(
        admission_path=marker, lock_path=lock, arm="arm_b"
    )["lock_path"] == str(lock)
    with pytest.raises(admission.AdmissionError, match="schema/path contract"):
        admission.verify(admission_path=marker, arm="arm_b")
    with pytest.raises(admission.AdmissionError, match="must be siblings"):
        admission.claim(
            admission_path=marker,
            lock_path=(tmp_path / "elsewhere" / "lock").absolute(),
            arm="arm_b",
        )


def test_paths_must_be_absolute_and_lock_must_not_be_symlink(tmp_path: Path) -> None:
    with pytest.raises(admission.AdmissionError, match="absolute path"):
        admission.claim(admission_path=Path("relative.json"), arm="arm_a")

    marker = (tmp_path / "one-arm.json").resolve()
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("")
    lock = tmp_path / "shared.lock"
    lock.symlink_to(lock_target)
    with pytest.raises(admission.AdmissionError, match="lock must not be a symlink"):
        admission.claim(admission_path=marker, lock_path=lock.absolute(), arm="arm_a")
