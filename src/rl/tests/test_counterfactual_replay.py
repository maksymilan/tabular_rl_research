#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from trajectory_replay import (  # noqa: E402
    CounterfactualReplayError,
    evaluate_counterfactual_suite,
    replay_terminal_evidence,
)


def write_items(path: Path, rows: list[tuple[int, str, str]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, status TEXT, name TEXT)")
    connection.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)
    connection.commit()
    connection.close()


def step(index: int, tool: str, arguments: dict, output: dict | None = None) -> dict:
    return {
        "step_id": f"step_{index}",
        "tool_call": {"tool": tool, "arguments": arguments},
        "tool_output": output or {},
    }


def trajectory(source_db: Path, *, include_filter: bool) -> dict:
    steps = [
        step(1, "describe_table", {"tables": ["items"]}),
    ]
    if include_filter:
        steps.extend([
            step(
                2,
                "condition_filter",
                {
                    "table": "items",
                    "conditions": {"column": "status", "op": "=", "value": "active"},
                },
                {"table": "filter_001"},
            ),
            step(
                3,
                "project",
                {"table": "filter_001", "expressions": ["name"]},
                {"table": "project_002"},
            ),
            step(
                4,
                "answer_from_context",
                {"evidence": {"table": "project_002"}},
            ),
        ])
    else:
        steps.extend([
            step(
                2,
                "project",
                {"table": "items", "expressions": ["name"]},
                {"table": "project_001"},
            ),
            step(
                3,
                "answer_from_context",
                {"evidence": {"table": "project_001"}},
            ),
        ])
    return {
        "trajectory_id": "items_program",
        "source": {
            "db_path": str(source_db),
            "gold_sql": "SELECT name FROM items WHERE status = 'active'",
        },
        "steps": steps,
    }


class CounterfactualReplayTests(unittest.TestCase):
    def test_counterexample_rejects_omitted_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            write_items(source, [(1, "active", "alpha")])
            write_items(alternate, [(1, "active", "alpha"), (2, "inactive", "beta")])
            program = trajectory(source, include_filter=False)

            replay = replay_terminal_evidence(program, alternate)
            suite = evaluate_counterfactual_suite(program, [alternate])

        self.assertFalse(replay.correct)
        self.assertEqual(replay.predicted_sample, [["alpha"], ["beta"]])
        self.assertEqual(replay.gold_sample, [["alpha"]])
        self.assertFalse(suite.passed)
        self.assertEqual(suite.reason, "counterexample_found")

    def test_equivalent_program_passes_changed_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            write_items(source, [(1, "active", "alpha")])
            write_items(
                alternate,
                [(1, "active", "gamma"), (2, "inactive", "beta")],
            )
            suite = evaluate_counterfactual_suite(
                trajectory(source, include_filter=True),
                [alternate],
            )

        self.assertTrue(suite.passed)
        self.assertEqual(suite.informative_databases, 1)

    def test_vacuous_suite_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            duplicate = root / "duplicate.sqlite"
            rows = [(1, "active", "alpha")]
            write_items(source, rows)
            write_items(duplicate, rows)
            suite = evaluate_counterfactual_suite(
                trajectory(source, include_filter=True),
                [duplicate],
            )

        self.assertFalse(suite.passed)
        self.assertEqual(suite.reason, "insufficient_informative_databases")

    def test_schema_mismatch_is_not_a_semantic_counterexample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            write_items(source, [(1, "active", "alpha")])
            connection = sqlite3.connect(alternate)
            connection.execute("CREATE TABLE other(value TEXT)")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(CounterfactualReplayError, "schema differs"):
                evaluate_counterfactual_suite(
                    trajectory(source, include_filter=True),
                    [alternate],
                )


if __name__ == "__main__":
    unittest.main()
