#!/usr/bin/env python3
from __future__ import annotations

import unittest

from select_join_pilot_examples import behavior_signals, select_ids


def record(trajectory_id: str, arguments: dict, **terminal_fields) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "example_index": int(trajectory_id.rsplit("_", 1)[-1]),
        "db_id": "db",
        "turns": [{"parsed": {"tool": "join_tables", "arguments": arguments}}],
        "error_events": [],
        **terminal_fields,
    }


class JoinPilotSelectionTests(unittest.TestCase):
    def test_terminal_and_gold_fields_cannot_change_ranking(self):
        args = {
            "tables": ["a", "b", "c"],
            "on": [[{"left": "id", "right": "id"}], [{"left": "id", "right": "id"}]],
        }
        left = record(
            "task_00001", args, correct=True, legal=True, difficulty="easy", gold_sql="SELECT 1",
        )
        right = record(
            "task_00001", args, correct=False, legal=False, difficulty="hard", gold_sql="SELECT 2",
        )
        self.assertEqual(behavior_signals(left), behavior_signals(right))

    def test_join_local_recovery_ranks_before_plain_binary_join(self):
        recovered = record(
            "task_00002",
            {"tables": ["a", "b"], "on": [[{"left": "id", "right": "id"}]]},
        )
        recovered["error_events"] = [{
            "attempted_tool": "join_tables",
            "error_type": "execution_error",
        }]
        plain = record(
            "task_00001",
            {"tables": ["a", "b", "c"], "on": [[], []]},
        )
        self.assertEqual(select_ids([plain, recovered], 1)[0]["trajectory_id"], "task_00002")

    def test_non_join_trajectory_is_not_a_candidate(self):
        non_join = {
            "trajectory_id": "task_00001",
            "turns": [{"parsed": {"tool": "project", "arguments": {}}}],
        }
        self.assertIsNone(behavior_signals(non_join))


if __name__ == "__main__":
    unittest.main()
