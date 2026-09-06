#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402


class ExternalFailureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "example_id": "bird_train_x",
            "dataset": "bird-sql",
            "split": "train",
            "db_id": "db",
            "db_path": "/tmp/db.sqlite",
            "gold_sql": "SELECT 1",
            "question": "q",
        }

    def test_preserves_error_indices_and_excludes_error_actions_from_steps(self):
        record = {
            "trajectory_id": "bird_train_x",
            "failure_type": "wrong_answer",
            "correct": False,
            "legal": True,
            "turns": [
                {
                    "turn_index": 0,
                    "parsed": {"think": "look", "tool": "describe_table", "arguments": {"tables": ["t"]}},
                    "tool_output": {"tables": []},
                },
                {
                    "turn_index": 1,
                    "execution_error_type": "protocol_error",
                    "execution_error": "bad carrier",
                    "error_event": {"action_index": 2, "step_id": "step_2"},
                },
                {
                    "turn_index": 2,
                    "parsed": {"think": "answer", "tool": "answer_from_context", "arguments": {"answer": [2]}},
                },
            ],
        }
        normalized, reason = normalize_failure_record(record, self.task)
        self.assertIsNone(reason)
        self.assertEqual([step["step_id"] for step in normalized["steps"]], ["step_1", "step_3"])
        self.assertEqual(
            [event["action_index"] for event in normalized["rollout_generation"]["error_events"]],
            [2],
        )
        self.assertFalse(normalized["rollout_generation"]["error_actions_are_sft_targets"])
        self.assertEqual(normalized["label_status"], "rollout_failure")

    def test_excludes_api_transport_failure(self):
        normalized, reason = normalize_failure_record(
            {"trajectory_id": "bird_train_x", "failure_type": "api_error", "turns": []},
            self.task,
        )
        self.assertIsNone(normalized)
        self.assertEqual(reason, "nonsemantic_runtime_failure")

    def test_excludes_generation_oom(self):
        normalized, reason = normalize_failure_record(
            {"trajectory_id": "bird_train_x", "failure_type": "generation_oom", "turns": []},
            self.task,
        )
        self.assertIsNone(normalized)
        self.assertEqual(reason, "nonsemantic_runtime_failure")

    def test_rejects_historical_denotation_metric(self):
        record = {
            "trajectory_id": "bird_train_x",
            "failure_type": "wrong_answer",
            "denotation_comparison": "strict-multiset",
            "turns": [],
        }
        with self.assertRaisesRegex(ValueError, "bird-set"):
            normalize_failure_record(record, self.task)


if __name__ == "__main__":
    unittest.main()
