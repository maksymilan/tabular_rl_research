from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl.scenarios.data.build_saam180_cohort import allocate, selection_order


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "src/rl/scenarios/data/build_saam180_cohort.py"


def test_allocate_respects_supply_and_total() -> None:
    quota = allocate(120, {2: 37, 3: 25, 4: 21, 5: 27, 6: 30})
    assert sum(quota.values()) == 120
    assert all(quota[bucket] <= supply for bucket, supply in {2: 37, 3: 25, 4: 21, 5: 27, 6: 30}.items())
    assert quota[4] == 18


def test_allocate_rejects_impossible_request() -> None:
    with pytest.raises(ValueError):
        allocate(10, {2: 3, 3: 4})


def test_selection_order_is_deterministic_and_seed_dependent() -> None:
    ids = ["b", "a", "c"]
    first = selection_order("s1", ids)
    assert first == selection_order("s1", ids)
    assert sorted(first) == sorted(ids)
    assert first != selection_order("s2", ids) or len(ids) < 3


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task(task_id: str, question: str, count: int | None, dataset: str = "bird-sql") -> dict:
    metadata = {}
    if count is not None:
        metadata["screened_rl_candidate_pool"] = {
            "screen_correct_count": count,
            "known_set_membership": {"checkpoint6380_sft_training_view": False},
        }
    return {
        "example_id": task_id,
        "instance_id": task_id,
        "dataset": dataset,
        "db_id": "demo",
        "db_path": f"data/{task_id}.sqlite",
        "question": question,
        "metadata": metadata,
    }


def test_cli_extends_cohort_and_excludes_base_and_sft_view(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    pool = tmp_path / "pool.jsonl"
    output = tmp_path / "out.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(
        base,
        [_task("t1", "question one", None), _task("t2", "question two", None)],
    )
    sft_row = _task("t3", "question three", 3)
    sft_row["metadata"]["screened_rl_candidate_pool"]["known_set_membership"] = {
        "checkpoint6380_sft_training_view": True
    }
    _write_jsonl(
        pool,
        [
            _task("t1", "question one", 4),  # duplicate identity -> excluded
            _task("t9", "question nine", 1),  # outside band -> excluded
            sft_row,  # SFT training view -> excluded
            _task("t4", "question four", 3),
            _task("t5", "question five", 4),
            _task("t6", "question six", 3),
        ],
    )
    subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--base-cohort",
            str(base),
            "--source-pool",
            str(pool),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--extension-records",
            "2",
            "--seed",
            "unit-test",
            "--db-path-prefix",
            "/remote",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    assert [row["example_id"] for row in rows[:2]] == ["t1", "t2"]
    assert [row["example_index"] for row in rows] == [0, 1, 2, 3]
    assert all(row["split"] == "train" for row in rows)
    extension_ids = {row["example_id"] for row in rows[2:]}
    assert extension_ids <= {"t4", "t5", "t6"}
    assert all(row["db_path"].startswith("/remote/data/") for row in rows[2:])
    payload = json.loads(manifest.read_text())
    assert payload["output"]["records"] == 4
    assert payload["filters"]["excluded_by_base_identity"] == 1
    assert payload["filters"]["datasets"] is None
    assert payload["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
