from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.data_selection.screened_pool import (
    build_candidate_rows,
    load_screening_observations,
    trainer_tasks,
)


TARGET = {
    "expected_group_size": 8,
    "correct_count_min": 2,
    "correct_count_max": 6,
    "model_checkpoint": "checkpoint-6380",
    "protocol_version": "version26",
    "protocol_hash": "4da19387399bd3a5",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_strict_historical_and_diagnostic_scopes_are_separate(tmp_path: Path) -> None:
    compact = tmp_path / "compact.jsonl"
    historical = tmp_path / "historical.jsonl"
    diagnostic = tmp_path / "diagnostic.jsonl"
    _write_jsonl(
        compact,
        [
            {
                "source": "strict-screen",
                "task_id": "bird_train_00001",
                "example_index": 1,
                "question_sha256": "a" * 64,
                "n_samples": 8,
                "attempted_samples": 8,
                "sample_correct_count": 2,
                "sample_legal_count": 8,
                "protocol_version": "version26",
                "protocol_hash": "4da19387399bd3a5",
            }
        ],
    )
    _write_jsonl(
        historical,
        [
            {
                "source": "old-screen",
                "example_id": "bird_train_00002",
                "n": 8,
                "correct_count": 5,
            }
        ],
    )
    _write_jsonl(
        diagnostic,
        [
            {
                "source": "rescreen",
                "task_id": "bird_train_00003",
                "example_index": 3,
                "question_sha256": "c" * 64,
                "n_samples": 8,
                "attempted_samples": 8,
                "sample_correct_count": 4,
                "sample_legal_count": 8,
                "protocol_version": "version26",
                "protocol_hash": "4da19387399bd3a5",
            }
        ],
    )
    specs = [
        {"label": "strict", "path": str(compact), "format": "compact_passk", "scope": "strict"},
        {"label": "old", "path": str(historical), "format": "legacy_candidate_index", "scope": "historical"},
        {"label": "diag", "path": str(diagnostic), "format": "compact_passk", "scope": "diagnostic"},
    ]
    observations, _ = load_screening_observations(tmp_path, specs, target=TARGET)
    tasks = {
        f"bird_train_{index:05d}": {
            "example_id": f"bird_train_{index:05d}",
            "dataset": "bird-sql",
            "db_id": "db",
            "db_path": f"/root/data/bird/db{index}.sqlite",
            "question": f"q{index}",
            "gold_sql": "select 1",
            "split": "train",
            "example_index": index,
        }
        for index in (1, 2, 3)
    }
    candidates = build_candidate_rows(
        tasks=tasks,
        task_owners={key: "tasks" for key in tasks},
        observations=observations,
        source_precedence=["strict-screen", "old-screen"],
        id_sets={},
        prior_cohort_task_ids=set(),
    )
    assert [row["task_id"] for row in candidates] == [
        "bird_train_00001",
        "bird_train_00002",
    ]
    assert candidates[0]["strict_candidate"] is True
    assert candidates[1]["historical_only"] is True


def test_duplicate_observation_is_deduplicated_and_conflict_fails(tmp_path: Path) -> None:
    compact = tmp_path / "compact.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    base = {
        "source": "same-screen",
        "task_id": "bird_train_00001",
        "example_index": 1,
        "question_sha256": "a" * 64,
        "n_samples": 8,
        "attempted_samples": 8,
        "sample_correct_count": 3,
        "sample_legal_count": 8,
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
    }
    _write_jsonl(compact, [base])
    _write_jsonl(
        legacy,
        [{"source": "same-screen", "example_id": "bird_train_00001", "n": 8, "correct_count": 3}],
    )
    specs = [
        {"label": "strict", "path": str(compact), "format": "compact_passk", "scope": "strict"},
        {"label": "legacy", "path": str(legacy), "format": "legacy_candidate_index", "scope": "historical"},
    ]
    observations, _ = load_screening_observations(tmp_path, specs, target=TARGET)
    assert len(observations) == 1
    assert observations[0]["scope"] == "strict"

    _write_jsonl(
        legacy,
        [{"source": "same-screen", "example_id": "bird_train_00001", "n": 8, "correct_count": 4}],
    )
    with pytest.raises(ValueError, match="conflicting screening observation"):
        load_screening_observations(tmp_path, specs, target=TARGET)


def test_target_range_is_versioned_and_controls_eligibility(tmp_path: Path) -> None:
    compact = tmp_path / "compact.jsonl"
    rows = []
    for index, correct_count in enumerate((1, 7, 8), start=1):
        rows.append(
            {
                "source": "screen",
                "task_id": f"bird_train_{index:05d}",
                "example_index": index,
                "question_sha256": "a" * 64,
                "n_samples": 8,
                "attempted_samples": 8,
                "sample_correct_count": correct_count,
                "sample_legal_count": 8,
                "protocol_version": "version26",
                "protocol_hash": "4da19387399bd3a5",
            }
        )
    _write_jsonl(compact, rows)
    target = {**TARGET, "correct_count_min": 1, "correct_count_max": 7}
    observations, _ = load_screening_observations(
        tmp_path,
        [{"label": "screen", "path": str(compact), "format": "compact_passk", "scope": "strict"}],
        target=target,
    )
    assert [row["eligible"] for row in observations] == [True, True, False]


def test_trainer_tasks_are_portable_and_keep_source_index() -> None:
    identifier = "bird_train_00007"
    tasks = {
        identifier: {
            "example_id": identifier,
            "instance_id": identifier,
            "example_index": 7,
            "index": 7,
            "dataset": "bird-sql",
            "db_id": "db",
            "db_path": "/machine/project/data/bird/train/db.sqlite",
            "question": "q",
            "gold_sql": "select 1",
            "split": "train",
        }
    }
    candidates = [
        {
            "task_id": identifier,
            "strict_candidate": True,
            "historical_only": False,
            "prior_saam700_cohort_member": False,
            "known_set_membership": {"sft": False},
            "fresh_under_known_exclusions": True,
            "selection_observation": {
                "source": "screen",
                "correct_count": 3,
                "n": 8,
            },
        }
    ]
    rows = trainer_tasks(candidates, tasks, pool_kind="strict")
    assert rows[0]["db_path"] == "data/bird/train/db.sqlite"
    assert rows[0]["example_index"] == 0
    metadata = rows[0]["metadata"]["screened_rl_candidate_pool"]
    assert metadata["source_example_index"] == 7
    assert metadata["screen_correct_count"] == 3
