import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from select_repeated_failure_recovery_anchors import (  # noqa: E402
    choose_repeated_failure_sample,
    selected_anchor,
)


class SelectRepeatedFailureRecoveryAnchorsTest(unittest.TestCase):
    def setUp(self):
        self.task = {
            "example_id": "bird_train_00001",
            "example_index": 1,
            "db_id": "db",
            "question": "Which row?",
            "external_knowledge": None,
            "gold_sql": "SELECT hidden_answer FROM secret",
            "metadata": {"difficulty_proxy": "hard"},
        }
        action = {
            "think": "Inspect the current result.",
            "tool": "read_subtable",
            "arguments": {"table": "project_001", "limit": 1},
        }
        self.sample = {
            "sample_index": 0,
            "correct": False,
            "legal": False,
            "failure_type": "max_steps",
            "steps": 30,
            "errors": 3,
            "turns": [
                {
                    "turn_index": 0,
                    "parsed": action,
                    "tool_output": {"rows": [["x"]]},
                },
                {
                    "turn_index": 1,
                    "execution_error_type": "no_progress_error",
                    "error_event": {"error_type": "no_progress_error"},
                },
                {
                    "turn_index": 2,
                    "execution_error_type": "no_progress_error",
                    "error_event": {"error_type": "no_progress_error"},
                },
                {
                    "turn_index": 3,
                    "execution_error_type": "no_progress_error",
                    "error_event": {"error_type": "no_progress_error"},
                },
            ],
            "error_events": [
                self.repeat_event(index)
                for index in (2, 3, 4)
            ],
        }

    @staticmethod
    def repeat_event(action_index):
        return {
            "action_index": action_index,
            "step_id": f"step_{action_index}",
            "error_type": "no_progress_error",
            "error_code": "adjacent_identical_action",
            "message": "exactly matches the immediately preceding action",
            "details": {
                "previous_step_id": f"step_{action_index - 1}",
                "state_changed_by_rejected_call": False,
            },
            "state_before_hash": "same",
            "state_after_hash": "same",
            "attempted_tool": "read_subtable",
            "attempted_arguments": {"table": "project_001", "limit": 1},
        }

    def test_selects_first_repeat_after_persistent_loop(self):
        chosen = choose_repeated_failure_sample(
            {"samples": [self.sample]},
            min_repeat_errors=3,
        )
        self.assertIsNotNone(chosen)
        _, events = chosen
        self.assertEqual([2, 3, 4], [event["action_index"] for event in events])

    def test_package_is_gold_free_and_context_only(self):
        with tempfile.TemporaryDirectory() as directory:
            package = selected_anchor(
                self.task,
                {"samples": [self.sample]},
                Path(directory) / "all.jsonl",
                min_repeat_errors=3,
            )
        self.assertIsNotNone(package)
        rendered = json.dumps(package, ensure_ascii=False)
        self.assertNotIn(self.task["gold_sql"], rendered)
        self.assertEqual(
            "no_progress_error",
            package["selected_candidate"]["last_tool_error"]["error"]["type"],
        )
        self.assertEqual(
            "adjacent_identical_action",
            package["selected_candidate"]["last_tool_error"]["error"]["code"],
        )
        self.assertFalse(
            package["selection_audit"]["rationale_is_sft_target"],
        )
        self.assertEqual(1, len(package["selected_candidate"]["legal_prefix"]))

    def test_excludes_infrastructure_invalidated_sample(self):
        invalid = dict(self.sample)
        invalid["failure_type"] = "context_overflow"
        self.assertIsNone(
            choose_repeated_failure_sample(
                {"samples": [invalid]},
                min_repeat_errors=3,
            )
        )

    def test_rejects_nonmatching_first_repeat(self):
        broken = json.loads(json.dumps(self.sample))
        broken["error_events"][0]["attempted_arguments"]["limit"] = 2
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            choose_repeated_failure_sample(
                {"samples": [broken]},
                min_repeat_errors=3,
            )

    def test_accepts_repeat_after_structured_rejected_action(self):
        rejected_action = {
            "tool": "group_aggregate",
            "arguments": {
                "table": "filter_001",
                "group_by": [],
                "aggregations": [
                    {
                        "op": "date_diff_days",
                        "column": "STOP",
                        "column2": "START",
                        "as": "duration_days",
                    }
                ],
            },
        }
        sample = json.loads(json.dumps(self.sample))
        sample["turns"] = [
            self.sample["turns"][0],
            {
                "turn_index": 1,
                "execution_error_type": "argument_validation_error",
                "parsed": None,
            },
            *[
                {
                    "turn_index": index - 1,
                    "execution_error_type": "no_progress_error",
                    "parsed": None,
                }
                for index in (3, 4, 5)
            ],
        ]
        rejected_event = {
            "action_index": 2,
            "step_id": "step_2",
            "error_type": "argument_validation_error",
            "error_code": "argument_validation_error",
            "message": "unexpected field column2",
            "state_before_hash": "same",
            "state_after_hash": "same",
            "attempted_tool": rejected_action["tool"],
            "attempted_arguments": rejected_action["arguments"],
        }
        sample["error_events"] = [
            rejected_event,
            *[
                {
                    **self.repeat_event(index),
                    "attempted_tool": rejected_action["tool"],
                    "attempted_arguments": rejected_action["arguments"],
                }
                for index in (3, 4, 5)
            ],
        ]

        chosen = choose_repeated_failure_sample(
            {"samples": [sample]},
            min_repeat_errors=3,
        )

        self.assertIsNotNone(chosen)
        _, events = chosen
        self.assertEqual([3, 4, 5], [event["action_index"] for event in events])


if __name__ == "__main__":
    unittest.main()
