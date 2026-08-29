from __future__ import annotations

import sqlite3

import pytest

from src.rl.evaluation.formal_v26_rollout_passk import (
    ToolExecutionTimeoutError,
    bounded_execute_tool,
)


class FakeHarness:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE items(value INTEGER)")
        self.views = {"items": "SELECT * FROM items"}
        self._n = 0


def _long_query(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        WITH RECURSIVE counter(value) AS (
          SELECT 0
          UNION ALL
          SELECT value + 1 FROM counter WHERE value < 1000000000
        )
        SELECT SUM(value) FROM counter
        """
    ).fetchone()


def test_timeout_rolls_back_harness_context_and_sql_transaction() -> None:
    harness = FakeHarness()
    context = {
        "history": [{"before": True}],
        "environment": {"phase": "before"},
    }

    def original(harness, _tool, _arguments, context, _step_id, _rows):
        harness.conn.execute("INSERT INTO items VALUES (1)")
        harness.views["derived_001"] = "SELECT 1"
        harness._n = 1
        context["history"].append({"after": True})
        context["handle_to_step"] = {"derived_001": "step_1"}
        context["environment"] = {"phase": "after"}
        _long_query(harness.conn)

    with pytest.raises(ToolExecutionTimeoutError, match="exceeded"):
        bounded_execute_tool(
            original,
            harness,
            "condition_filter",
            {},
            context,
            "step_1",
            timeout_seconds=1e-9,
        )

    assert harness.views == {"items": "SELECT * FROM items"}
    assert harness._n == 0
    assert context == {
        "history": [{"before": True}],
        "environment": {"phase": "before"},
    }
    assert harness.conn.execute("SELECT COUNT(*) FROM items").fetchone() == (0,)
    # A cleared progress handler must not interrupt the next statement.
    assert harness.conn.execute("SELECT 1").fetchone() == (1,)


def test_success_preserves_original_result_and_clears_progress_handler() -> None:
    harness = FakeHarness()
    context = {"history": None, "environment": {}}

    def original(harness, _tool, _arguments, context, _step_id, _rows):
        harness.views["derived_001"] = "SELECT 1"
        harness._n = 1
        context["history"] = ["success"]
        return {"ok": True}, "derived_001"

    result = bounded_execute_tool(
        original,
        harness,
        "project",
        {},
        context,
        "step_1",
        timeout_seconds=1.0,
    )

    assert result == ({"ok": True}, "derived_001")
    assert harness.views["derived_001"] == "SELECT 1"
    assert context["history"] == ["success"]
    assert harness.conn.execute("SELECT 1").fetchone() == (1,)


def test_non_timeout_operational_error_is_not_relabelled() -> None:
    harness = FakeHarness()
    context = {"history": []}

    def original(harness, *_args):
        harness.conn.execute("SELECT * FROM missing_table")

    with pytest.raises(sqlite3.OperationalError, match="missing_table"):
        bounded_execute_tool(
            original,
            harness,
            "project",
            {},
            context,
            "step_1",
            timeout_seconds=1.0,
        )
    assert harness.conn.execute("SELECT 1").fetchone() == (1,)
