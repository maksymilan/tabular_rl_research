from __future__ import annotations

import sqlite3

from build_counterfactual_suite_for_fixed_pool import build_task_suite


def test_builds_two_schema_preserving_nonempty_counterfactuals(tmp_path) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, score INTEGER)")
    connection.executemany(
        "INSERT INTO people VALUES(?,?,?)",
        [(1, "a", 10), (2, "b", 20), (3, "c", 30)],
    )
    connection.commit()
    connection.close()
    task = {
        "example_id": "bird_train_test",
        "db_path": str(source),
        "gold_sql": "SELECT name FROM people WHERE id = 2",
    }

    suite = build_task_suite(task, tmp_path / "counterfactuals")

    assert len(suite["databases"]) == 2
    assert suite["databases"][0]["gold_result_sha256"] != suite["source_gold_result_sha256"]
    assert suite["databases"][1]["gold_result_sha256"] != suite["databases"][0]["gold_result_sha256"]
    assert all(row["schema_sha256"] == suite["source_schema_sha256"] for row in suite["databases"])
