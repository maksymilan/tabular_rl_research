from __future__ import annotations

import json
from pathlib import Path

import pytest

from recover_policy_boundary_selection import OUTPUT_NAMES, recover_selection


ROOT = Path(__file__).resolve().parents[3]
SELECTOR = ROOT / "src/rl/select_policy_boundary_grpo_tasks.py"
SEED = "qwen3-v26-boundary332-v1-20260812"


def _write_pool(
    root: Path,
    *,
    name: str,
    offset: int,
    mixed: int = 180,
    core: int = 120,
) -> tuple[Path, Path]:
    tasks = []
    groups = []
    for position in range(600):
        index = offset + position
        task_id = f"bird_train_{index:05d}"
        tasks.append(
            {
                "example_id": task_id,
                "instance_id": task_id,
                "example_index": index,
                "db_id": f"db_{position % 60}",
                "question": "q" * (30 + position % 50),
                "external_knowledge": "hint" if position % 2 else None,
            }
        )
        correct = 8
        if position < mixed:
            correct = 4 if position < core else 1
        groups.append(
            {
                "task_id": task_id,
                "example_index": index,
                "correct_count": correct,
                "uncertainty": correct * (8 - correct) if 1 <= correct <= 7 else 0,
                "usable": True,
                "mixed_boundary": 1 <= correct <= 7,
                "core_boundary": 2 <= correct <= 6,
                "contamination": [],
            }
        )
    task_path = root / f"{name}.jsonl"
    task_bytes = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in tasks).encode()
    task_path.write_bytes(task_bytes)
    audit_path = root / f"{name}.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "vanilla-grpo-boundary-screen-audit-v1",
                "contract": {
                    "tasks": 600,
                    "group_size": 8,
                    "optimizer_updates": 0,
                    "initial_adapter_sha256": "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5",
                    "reward_mode": "result-only",
                    "result_reward_profile": "binary",
                },
                "inputs": {
                    "tasks": str(task_path.resolve()),
                    "tasks_sha256": __import__("hashlib").sha256(task_bytes).hexdigest(),
                    "manifest_sha256": "a" * 64,
                    "trajectories_sha256": "b" * 64,
                },
                "groups": groups,
                "issues": [],
                "issue_counts": {},
                "status": {"audit_passes": True, "pool_admitted": True},
            }
        )
    )
    return audit_path, task_path


def _inputs(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    first = _write_pool(tmp_path, name="s1", offset=0)
    second = _write_pool(tmp_path, name="s2", offset=600)
    return [first[0], second[0]], [first[1], second[1]]


def _recover(tmp_path: Path, output: Path) -> dict:
    audits, tasks = _inputs(tmp_path)
    output.mkdir(exist_ok=True)
    return recover_selection(
        selector_path=SELECTOR,
        screen_audits=audits,
        tasks=tasks,
        output_dir=output,
        quarantine_dir=tmp_path / "quarantine",
        seed=SEED,
    )


@pytest.mark.parametrize("published_before_crash", [0, 1, 2, 3, 4])
def test_recovers_every_selector_publication_prefix(
    tmp_path: Path, published_before_crash: int
) -> None:
    output = tmp_path / "selection"
    first = _recover(tmp_path, output)
    assert first["status"] == "complete"
    expected = {name: (output / name).read_bytes() for name in OUTPUT_NAMES}
    for name in OUTPUT_NAMES[published_before_crash:]:
        (output / name).unlink()

    result = _recover(tmp_path, output)
    assert result["status"] == "complete"
    assert len(result["published_missing"]) == 4 - published_before_crash
    assert {name: (output / name).read_bytes() for name in OUTPUT_NAMES} == expected


def test_matching_selector_next_file_is_recovered(tmp_path: Path) -> None:
    output = tmp_path / "selection"
    _recover(tmp_path, output)
    name = OUTPUT_NAMES[-1]
    final = output / name
    temporary = output / f"{name}.next"
    final.replace(temporary)

    result = _recover(tmp_path, output)
    assert final.is_file() and not temporary.exists()
    assert str(final.resolve()) in result["published_missing"]


def test_verification_temporary_after_four_finals_is_quarantined(
    tmp_path: Path,
) -> None:
    output = tmp_path / "selection"
    _recover(tmp_path, output)
    temporary = output / "boundary_selection_verification.json.next"
    temporary.write_text("interrupted uncommitted verification")

    result = _recover(tmp_path, output)
    assert not temporary.exists()
    assert len(result["quarantined_redundant_temporaries"]) == 1
    destination = Path(
        result["quarantined_redundant_temporaries"][0]["to"]
    )
    assert destination.read_text() == "interrupted uncommitted verification"


def test_mismatch_or_unknown_fails_before_publishing_missing(tmp_path: Path) -> None:
    output = tmp_path / "selection"
    _recover(tmp_path, output)
    missing = output / OUTPUT_NAMES[-1]
    missing.unlink()
    first = output / OUTPUT_NAMES[0]
    first.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        _recover(tmp_path, output)
    assert not missing.exists()

    first.unlink()
    (output / "unknown.tmp").write_text("unknown")
    with pytest.raises(ValueError, match="unknown boundary selection"):
        _recover(tmp_path, output)
    assert not missing.exists()
