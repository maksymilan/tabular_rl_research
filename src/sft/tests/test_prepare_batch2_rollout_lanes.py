import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_batch2_rollout_lanes import (  # noqa: E402
    legal_prefix_before,
    recoverable_error_events,
    recovery_package,
)


class PrepareBatch2RolloutLanesTest(unittest.TestCase):
    def setUp(self):
        self.task = {
            "example_id": "bird_train_00001",
            "example_index": 1,
            "db_id": "db",
            "question": "Which row?",
            "external_knowledge": None,
            "gold_sql": "SELECT secret FROM answer",
            "metadata": {"difficulty_proxy": "medium"},
        }
        self.legal_turn = {
            "turn_index": 0,
            "parsed": {
                "think": "inspect",
                "tool": "describe_table",
                "arguments": {"tables": ["items"]},
            },
            "tool_output": {"tables": [{"table_name": "items"}]},
        }
        self.failed_turn = {
            "turn_index": 1,
            "parsed": {
                "think": "filter",
                "tool": "condition_filter",
                "arguments": {"table": "items", "conditions": {}},
            },
            "execution_error_type": "execution_error",
            "error_event": {"error_type": "execution_error"},
            "raw_model_output": "must never become a prefix target",
        }

    def test_prefix_excludes_rejected_and_future_turns(self):
        future = {
            "turn_index": 2,
            "parsed": {"tool": "read_subtable", "arguments": {"table": "items"}},
            "tool_output": {"rows": []},
        }
        prefix = legal_prefix_before([self.legal_turn, self.failed_turn, future], 2)
        self.assertEqual(1, len(prefix))
        self.assertEqual("describe_table", prefix[0]["tool"])
        self.assertNotIn("raw_model_output", json.dumps(prefix))

    def test_only_state_preserving_errors_are_candidates(self):
        sample = {
            "error_events": [
                {
                    "action_index": 2,
                    "error_type": "execution_error",
                    "message": "bad column",
                    "state_before_hash": "same",
                    "state_after_hash": "same",
                    "attempted_tool": "condition_filter",
                    "attempted_arguments": {"table": "items"},
                },
                {
                    "action_index": 3,
                    "error_type": "nonrecoverable_execution_error",
                    "message": "changed",
                    "state_before_hash": "before",
                    "state_after_hash": "after",
                },
            ]
        }
        events = recoverable_error_events(sample)
        self.assertEqual([2], [event["action_index"] for event in events])

    def test_recovery_package_is_causal_and_gold_free(self):
        sample = {
            "sample_index": 0,
            "failure_type": "execution_error",
            "legal": False,
            "steps": 2,
            "errors": 1,
            "turns": [self.legal_turn, self.failed_turn],
            "error_events": [
                {
                    "action_index": 2,
                    "step_id": "step_2",
                    "error_type": "execution_error",
                    "message": "bad column",
                    "state_before_hash": "same",
                    "state_after_hash": "same",
                    "attempted_tool": "condition_filter",
                    "attempted_arguments": {"table": "items"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            package = recovery_package(
                self.task,
                sample,
                rollout_source=Path(directory) / "all.jsonl",
            )
        self.assertIsNotNone(package)
        rendered = json.dumps(package, ensure_ascii=False)
        self.assertNotIn(self.task["gold_sql"], rendered)
        self.assertFalse(package["selection_contract"]["future_suffix_visible"])
        self.assertFalse(package["selection_contract"]["rejected_action_is_sft_target"])
        self.assertEqual(1, len(package["candidates"][0]["legal_prefix"]))


if __name__ == "__main__":
    unittest.main()
