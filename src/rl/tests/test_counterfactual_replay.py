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
    derive_observation_bindings,
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


def write_observation_binding_db(
    path: Path,
    *,
    entity_id: int,
    entity_name: str,
) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE events(event_key INTEGER PRIMARY KEY, entity_ref INTEGER);
        CREATE TABLE entities(id INTEGER PRIMARY KEY, name TEXT);
        """
    )
    connection.execute("INSERT INTO events VALUES (100, ?)", (entity_id,))
    connection.execute("INSERT INTO entities VALUES (?, ?)", (entity_id, entity_name))
    connection.commit()
    connection.close()


def observation_binding_trajectory(source_db: Path) -> dict:
    return {
        "trajectory_id": "observation_binding_program",
        "question": "What is the entity name for event key 100?",
        "source": {
            "db_path": str(source_db),
            "gold_sql": (
                "SELECT entities.name FROM events "
                "JOIN entities ON events.entity_ref = entities.id "
                "WHERE events.event_key = 100"
            ),
        },
        "steps": [
            step(1, "describe_table", {"tables": ["events", "entities"]}),
            step(
                2,
                "condition_filter",
                {
                    "table": "events",
                    "conditions": {"column": "event_key", "op": "=", "value": 100},
                    "return_columns": ["entity_ref"],
                },
                {"table": "filter_001", "columns": ["entity_ref"], "rows": [[7]]},
            ),
            step(
                3,
                "condition_filter",
                {
                    "table": "entities",
                    "conditions": {"column": "id", "op": "=", "value": 7},
                    "return_columns": ["name"],
                },
                {"table": "filter_002", "columns": ["name"], "rows": [["alpha"]]},
            ),
            step(
                4,
                "answer_from_context",
                {"evidence": {"table": "filter_002"}},
            ),
        ],
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

    def test_visible_observation_literal_is_rebound_on_counterfactual_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            write_observation_binding_db(source, entity_id=7, entity_name="alpha")
            write_observation_binding_db(alternate, entity_id=42, entity_name="gamma")
            program = observation_binding_trajectory(source)

            bindings = derive_observation_bindings(program)
            fixed_literal = replay_terminal_evidence(
                program,
                alternate,
                observation_bindings=(),
            )
            adaptive = replay_terminal_evidence(program, alternate)

        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].consumer_step_id, "step_3")
        self.assertEqual(bindings[0].source.step_id, "step_2")
        self.assertEqual(bindings[0].argument_path, ("conditions", "value"))
        self.assertFalse(fixed_literal.correct)
        self.assertTrue(adaptive.correct)
        self.assertEqual(adaptive.predicted_sample, [["gamma"]])
        self.assertEqual(adaptive.observation_bindings_applied, 1)

    def test_multi_row_choice_is_not_treated_as_a_replay_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            for path, first_score, second_score in (
                (source, 8, 7),
                (alternate, 7, 8),
            ):
                connection = sqlite3.connect(path)
                connection.executescript(
                    """
                    CREATE TABLE rankings(id INTEGER, score INTEGER);
                    CREATE TABLE entities(id INTEGER PRIMARY KEY, name TEXT);
                    INSERT INTO entities VALUES (1, 'alpha'), (2, 'beta');
                    """
                )
                connection.executemany(
                    "INSERT INTO rankings VALUES (?, ?)",
                    [(1, first_score), (2, second_score)],
                )
                connection.commit()
                connection.close()
            program = {
                "trajectory_id": "implicit_argmax",
                "question": "Which entity has the greatest score?",
                "source": {
                    "db_path": str(source),
                    "gold_sql": (
                        "SELECT entities.name FROM rankings JOIN entities USING(id) "
                        "ORDER BY rankings.score DESC LIMIT 1"
                    ),
                },
                "steps": [
                    step(1, "describe_table", {"tables": ["rankings", "entities"]}),
                    step(2, "read_subtable", {"table": "rankings", "limit": 2}),
                    step(
                        3,
                        "condition_filter",
                        {
                            "table": "entities",
                            "conditions": {"column": "id", "op": "=", "value": 1},
                            "return_columns": ["name"],
                        },
                        {"table": "filter_001", "columns": ["name"], "rows": [["alpha"]]},
                    ),
                    step(
                        4,
                        "answer_from_context",
                        {"evidence": {"table": "filter_001"}},
                    ),
                ],
            }

            bindings = derive_observation_bindings(program)
            replay = replay_terminal_evidence(program, alternate)

        self.assertEqual(bindings, ())
        self.assertFalse(replay.correct)
        self.assertEqual(replay.predicted_sample, [["alpha"]])
        self.assertEqual(replay.gold_sample, [["beta"]])


if __name__ == "__main__":
    unittest.main()
