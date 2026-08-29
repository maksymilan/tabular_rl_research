from __future__ import annotations

from src.rl.select_epoch4_pass8_boundary_cohort import (
    build_group_index,
    select_groups,
    sql_structure,
)


def _task(index: int, identifier: str, sql: str = "SELECT * FROM t") -> dict:
    return {
        "example_index": index,
        "example_id": identifier,
        "db_id": f"db_{index}",
        "question": f"question {index}",
        "gold_sql": sql,
    }


def _record(index: int, correct_count: int, *, failure_type: str | None = None) -> dict:
    samples = []
    for sample_index in range(8):
        samples.append(
            {
                "sample_index": sample_index,
                "correct": sample_index < correct_count,
                "failure_type": failure_type if sample_index == 0 else None,
                "api_transport_retries": 0,
                "api_context_retries": 0,
            }
        )
    return {
        "example_index": index,
        "n_samples": 8,
        "protocol_version": "version26",
        "samples": samples,
    }


def test_sql_structure_is_content_free_and_dialect_tolerant() -> None:
    structure = sql_structure(
        "SELECT a, COUNT(*) FROM t JOIN u ON t.id=u.id "
        "WHERE a > 1 GROUP BY a ORDER BY a"
    )
    assert structure["parse_status"] == "ok"
    assert structure["bin"] == "medium"
    assert structure["joins"] == 1
    assert structure["groups"] == 1
    assert "sql" not in structure


def test_only_clean_mixed_groups_are_eligible_and_core_is_preferred() -> None:
    tasks = [
        _task(0, "bird_train_00000"),
        _task(1, "spider_train_00001"),
        _task(2, "synsql_train_00002"),
    ]
    records = [_record(0, 8), _record(1, 0), _record(2, 4)]
    groups = build_group_index(tasks, records, cutpoints=(10, 10, 10, 10))
    assert [group["usable"] for group in groups] == [False, False, True]
    selected, details = select_groups(groups, tasks, count=1)
    assert [group["task_id"] for group in selected] == ["synsql_train_00002"]
    assert details["selected_core_records"] == 1


def test_infrastructure_or_retry_contamination_excludes_a_mixed_group() -> None:
    tasks = [_task(0, "bird_train_00000"), _task(1, "bird_train_00001")]
    dirty = _record(0, 4, failure_type="context_overflow")
    clean = _record(1, 4)
    clean["samples"][0]["api_transport_retries"] = 1
    groups = build_group_index(tasks, [dirty, clean], cutpoints=(10, 10, 10, 10))
    assert all(not group["usable"] for group in groups)
