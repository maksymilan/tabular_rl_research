from __future__ import annotations

from copy import deepcopy

from select_iterative_sql_v6_target_gate import (
    EXPLICIT_COMBINED_CONTROLS,
    SEPARATE_FIELD_TARGETS,
    SINGLE_FIELD_NAME_CONTROLS,
    select_rows,
    task_id,
)


def reviewed_rows() -> list[dict]:
    rows: list[dict] = []
    for index, row_id in enumerate(SEPARATE_FIELD_TARGETS):
        rows.append({
            "instance_id": row_id,
            "example_index": index,
            "db_id": f"target_{index}",
            "question": "Give the full name.",
            "external_knowledge": "full name refers to first_name, last_name",
            "gold_sql": f"SELECT target_{index}",
        })
    for index, row_id in enumerate(EXPLICIT_COMBINED_CONTROLS):
        rows.append({
            "instance_id": row_id,
            "example_index": 100 + index,
            "db_id": f"combined_{index}",
            "question": "Give the full name.",
            "external_knowledge": "full name = first_name+last_name",
            "gold_sql": f"SELECT combined_{index}",
        })
    for index, row_id in enumerate(SINGLE_FIELD_NAME_CONTROLS):
        rows.append({
            "instance_id": row_id,
            "example_index": 200 + index,
            "db_id": f"single_{index}",
            "question": "Give the full name.",
            "external_knowledge": "full name refers to display_name;",
            "gold_sql": f"SELECT single_{index}",
        })
    for index in range(12):
        rows.append({
            "instance_id": f"ordinary_{index:02d}",
            "example_index": 300 + index,
            "db_id": f"ordinary_db_{index}",
            "question": f"How many records satisfy condition {index}?",
            "external_knowledge": "count records refers to COUNT(*)",
            "gold_sql": f"SELECT ordinary_{index}",
        })
    return rows


def test_selection_is_deterministic_disjoint_and_has_declared_categories() -> None:
    rows = reviewed_rows()
    first, categories = select_rows(rows, {"ordinary_00"})
    second, _ = select_rows(rows, {"ordinary_00"})
    assert [task_id(row) for row in first] == [task_id(row) for row in second]
    assert len(first) == len({task_id(row) for row in first}) == 20
    assert "ordinary_00" not in {task_id(row) for row in first}
    assert {key: len(value) for key, value in categories.items()} == {
        "separate_field_targets": 6,
        "explicit_combined_controls": 4,
        "single_field_name_controls": 4,
        "ordinary_controls": 6,
    }


def test_selection_is_invariant_to_gold_fields() -> None:
    rows = reviewed_rows()
    changed = deepcopy(rows)
    for index, row in enumerate(changed):
        row["gold_sql"] = f"SELECT changed_{index}"
        row["query"] = f"SELECT hidden_{index}"
        row["gold_exec_results"] = [["changed", index]]
    original, _ = select_rows(rows, set())
    mutated, _ = select_rows(changed, set())
    assert [task_id(row) for row in original] == [task_id(row) for row in mutated]
