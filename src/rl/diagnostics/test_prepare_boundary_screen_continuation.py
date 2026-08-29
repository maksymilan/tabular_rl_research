from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rl.diagnostics import prepare_boundary_screen_continuation as module


def _task(index: int) -> dict:
    task_id = f"bird_train_{index:05d}"
    return {
        "example_id": task_id,
        "instance_id": task_id,
        "example_index": index,
        "db_id": "missing_on_secondary" if index in {33, 37} else "shared",
    }


def _group(path: Path, position: int, index: int) -> None:
    task_id = f"bird_train_{index:05d}"
    rows = []
    for sample in range(8):
        rows.append(
            {
                "sequence": position * 8 + sample,
                "environment": {"task_id": task_id},
                "sample": {"audit_record": {"example_index": index, "sample_index": sample}},
            }
        )
    path.write_text(json.dumps(rows))


def test_prepare_freezes_disjoint_remaining_assignments(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    rows = [_task(index) for index in range(40)]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(module, "EXPECTED_TASKS_SHA256", module._sha(tasks))
    monkeypatch.setattr(module, "EXPECTED_RECORDS", 40)
    groups = tmp_path / "groups"
    groups.mkdir()
    for position, row in enumerate(rows[:32]):
        _group(groups / f"{row['example_id']}.json", position, row["example_index"])
    output = tmp_path / "plan"
    manifest = module.prepare(
        tasks_path=tasks, group_dirs=[groups], output_dir=output, shards=4
    )
    assert manifest["completed"]["tasks"] == 32
    assert manifest["remaining"]["tasks"] == 8
    assignments = [set(record["task_ids"]) for record in manifest["assignments"]]
    assert all(len(values) == 2 for values in assignments)
    assert set().union(*assignments) == {row["example_id"] for row in rows[32:]}
    assert all(assignments[left].isdisjoint(assignments[right]) for left in range(4) for right in range(left + 1, 4))


def test_prepare_rejects_duplicate_or_corrupt_completed_group(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    rows = [_task(index) for index in range(40)]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(module, "EXPECTED_TASKS_SHA256", module._sha(tasks))
    monkeypatch.setattr(module, "EXPECTED_RECORDS", 40)
    first = tmp_path / "first"; second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    for position, row in enumerate(rows[:32]):
        _group(first / f"{row['example_id']}.json", position, row["example_index"])
    _group(second / f"{rows[0]['example_id']}.json", 0, rows[0]["example_index"])
    with pytest.raises(ValueError, match="duplicate"):
        module.prepare(tasks_path=tasks, group_dirs=[first, second], output_dir=tmp_path / "out", shards=4)


def test_prepare_routes_restricted_dbs_only_to_capable_prefix_shards(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    rows = [_task(index) for index in range(40)]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(module, "EXPECTED_TASKS_SHA256", module._sha(tasks))
    monkeypatch.setattr(module, "EXPECTED_RECORDS", 40)
    groups = tmp_path / "groups"; groups.mkdir()
    for position, row in enumerate(rows[:32]):
        _group(groups / f"{row['example_id']}.json", position, row["example_index"])
    manifest = module.prepare(
        tasks_path=tasks,
        group_dirs=[groups],
        output_dir=tmp_path / "out",
        shards=4,
        restricted_db_ids=["missing_on_secondary"],
        restricted_shards=2,
    )
    restricted = {"bird_train_00033", "bird_train_00037"}
    assert set(manifest["host_capability_partition"]["restricted_tasks"]) == restricted
    assert restricted <= set(manifest["assignments"][0]["task_ids"] + manifest["assignments"][1]["task_ids"])
    assert restricted.isdisjoint(manifest["assignments"][2]["task_ids"])
    assert restricted.isdisjoint(manifest["assignments"][3]["task_ids"])


def test_prepare_requires_first32_complete(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    rows = [_task(index) for index in range(40)]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(module, "EXPECTED_TASKS_SHA256", module._sha(tasks))
    monkeypatch.setattr(module, "EXPECTED_RECORDS", 40)
    groups = tmp_path / "groups"; groups.mkdir()
    for position, row in enumerate(rows[:31]):
        _group(groups / f"{row['example_id']}.json", position, row["example_index"])
    with pytest.raises(ValueError, match="gates failed"):
        module.prepare(tasks_path=tasks, group_dirs=[groups], output_dir=tmp_path / "out", shards=4)
