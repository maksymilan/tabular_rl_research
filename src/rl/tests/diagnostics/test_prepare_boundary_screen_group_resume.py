from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.scenarios.diagnostics.prepare_boundary_screen_group_resume import prepare_group_resume


def _write_tasks(path: Path) -> list[str]:
    ids = [f"bird_train_{index:05d}" for index in range(3)]
    path.write_text(
        "".join(
            json.dumps({"example_id": task_id, "example_index": index}) + "\n"
            for index, task_id in enumerate(ids)
        )
    )
    return ids


def test_quarantines_only_known_atomic_partials_and_is_idempotent(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    ids = _write_tasks(tasks)
    assignment = tmp_path / "assignment.txt"
    assignment.write_text("\n".join(ids[:2]) + "\n")
    groups = tmp_path / "groups"
    groups.mkdir()
    (groups / f"{ids[0]}.json").write_text("[]")
    partial = groups / f"{ids[1]}.json.next"
    partial.write_text("partial")
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    (quarantine / partial.name).write_text("older partial")

    result = prepare_group_resume(
        tasks=tasks,
        task_id_file=assignment,
        groups_dir=groups,
        quarantine_dir=quarantine,
    )
    assert result["status"] == "resume_ready"
    assert result["completed_groups"] == 1
    assert len(result["quarantined_partials"]) == 1
    assert (quarantine / f"{partial.name}.1").read_text() == "partial"
    assert not partial.exists()
    assert (groups / f"{ids[0]}.json").read_text() == "[]"

    again = prepare_group_resume(
        tasks=tasks,
        task_id_file=assignment,
        groups_dir=groups,
        quarantine_dir=quarantine,
    )
    assert again["quarantined_partials"] == []


def test_unknown_entry_fails_before_moving_known_partial(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    ids = _write_tasks(tasks)
    groups = tmp_path / "groups"
    groups.mkdir()
    partial = groups / f"{ids[0]}.json.next"
    partial.write_text("partial")
    (groups / "unknown.tmp").write_text("unknown")

    with pytest.raises(ValueError, match="unknown group entry"):
        prepare_group_resume(
            tasks=tasks,
            groups_dir=groups,
            quarantine_dir=tmp_path / "quarantine",
        )
    assert partial.read_text() == "partial"
    assert not (tmp_path / "quarantine").exists()


def test_symlink_partial_is_rejected_without_touching_target(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    ids = _write_tasks(tasks)
    groups = tmp_path / "groups"
    groups.mkdir()
    target = tmp_path / "target"
    target.write_text("do not move")
    (groups / f"{ids[0]}.json.next").symlink_to(target)

    with pytest.raises(ValueError, match="partial group is not a regular file"):
        prepare_group_resume(
            tasks=tasks,
            groups_dir=groups,
            quarantine_dir=tmp_path / "quarantine",
        )
    assert target.read_text() == "do not move"

