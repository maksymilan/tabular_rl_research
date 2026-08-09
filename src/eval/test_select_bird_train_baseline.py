from __future__ import annotations

from copy import deepcopy

from select_bird_train_baseline import (
    distribution_audit,
    length_cutpoints,
    select_rows,
    task_id,
)


def make_rows() -> list[dict]:
    rows = []
    index = 0
    for database, count in (("large", 30), ("medium", 18), ("small", 12)):
        for local_index in range(count):
            rows.append(
                {
                    "example_id": f"task-{index:03d}",
                    "db_id": database,
                    "question": "q" * (20 + local_index * 3),
                    "external_knowledge": "evidence" if local_index % 4 else None,
                    "gold_sql": f"SELECT {index}",
                    "gold_exec_results": [[index]],
                }
            )
            index += 1
    return rows


def test_selection_is_deterministic_disjoint_and_covers_databases() -> None:
    rows = make_rows()
    excluded = {"task-000", "task-030", "task-048"}
    first, _ = select_rows(rows, rows, excluded, count=18, seed="test-seed")
    second, _ = select_rows(rows, rows, excluded, count=18, seed="test-seed")

    assert [task_id(row) for row in first] == [task_id(row) for row in second]
    assert len(first) == len({task_id(row) for row in first}) == 18
    assert not ({task_id(row) for row in first} & excluded)
    assert {row["db_id"] for row in first} == {"large", "medium", "small"}


def test_selection_is_invariant_to_gold_fields() -> None:
    rows = make_rows()
    changed_gold = deepcopy(rows)
    for index, row in enumerate(changed_gold):
        row["gold_sql"] = f"SELECT changed_{index}"
        row["query"] = f"SELECT another_{index}"
        row["gold_exec_results"] = [["changed", index]]
        row["gold_sql_path"] = f"/different/{index}.sql"

    original, _ = select_rows(rows, rows, set(), count=18, seed="test-seed")
    changed, _ = select_rows(rows, changed_gold, set(), count=18, seed="test-seed")

    assert [task_id(row) for row in original] == [task_id(row) for row in changed]


def test_distribution_audit_reports_matching_population() -> None:
    rows = make_rows()
    cutpoints = length_cutpoints(rows)
    audit = distribution_audit(rows, rows, cutpoints=cutpoints)

    assert audit["axes"]["database"]["total_variation_distance"] == 0
    assert audit["axes"]["question_length_quintile"]["total_variation_distance"] == 0
    assert audit["axes"]["external_knowledge"]["total_variation_distance"] == 0
    assert audit["question_length_characters"]["relative_mean_difference"] == 0
