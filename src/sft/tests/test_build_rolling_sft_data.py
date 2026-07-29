#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from build_rolling_sft_data import convert_step, is_sft_target_step  # noqa: E402
from select_verified_rollouts import quality_reason  # noqa: E402


def trajectory() -> dict:
    state0 = {"plan": [], "tables": {}, "values": {}}
    state1 = {"plan": [], "tables": {"items": {"row_count": 2}}, "values": {}}
    return {
        "trajectory_id": "rolling_1",
        "question": "Count items.",
        "source": {"gold_sql": "SELECT count(*) FROM items"},
        "initial_state": {"dataset_overview": {"tables": [], "relations": []}},
        "steps": [
            {
                "step_id": "step_1", "think": "Inspect items.",
                "tool_call": {"tool": "describe_table", "arguments": {"tables": ["items"]}},
                "tool_output": {"tables": {"items": {"columns": ["id"]}}},
                "environment_state_before": state0, "environment_state": state1,
                "last_tool_error_before": None,
            },
            {
                "step_id": "step_3", "think": "Return the known count.",
                "tool_call": {"tool": "answer_from_context", "arguments": {"answer": [[2]], "evidence": None}},
                "tool_output": {"final_answer": [[2]]},
                "environment_state_before": state1, "environment_state": state1,
                "last_tool_error_before": {"step_id": "step_2", "status": "error", "error": {"type": "protocol_error"}},
                "feedback_recovery": True, "recovered_from_error_type": "protocol_error",
            },
        ],
    }


class BuildRollingSftTests(unittest.TestCase):
    def test_recovery_prefix_can_be_context_only(self):
        item = trajectory()
        item["steps"][0]["sft_target_eligible"] = False
        self.assertFalse(is_sft_target_step(item["steps"][0]))
        self.assertTrue(is_sft_target_step(item["steps"][1]))
        record, _ = convert_step(item, 1, 4, "rolling system")
        self.assertIn("Inspect items.", record["conversations"][1]["value"])

    def test_second_target_keeps_prior_legal_pair_and_current_error(self):
        record, index = convert_step(trajectory(), 1, 4, "rolling system")
        roles = [message["from"] for message in record["conversations"]]
        self.assertEqual(roles, ["human", "gpt", "human", "gpt"])
        self.assertIn("Inspect items.", record["conversations"][1]["value"])
        self.assertIn("LAST TOOL ERROR", record["conversations"][2]["value"])
        self.assertIn("step_2", record["conversations"][2]["value"])
        self.assertIn("Return the known count.", record["conversations"][-1]["value"])
        self.assertTrue(index["feedback_recovery"])

    def test_target_is_not_in_its_own_prompt(self):
        record, _ = convert_step(trajectory(), 0, 4, "rolling system")
        prompt = "\n".join(message["value"] for message in record["conversations"][:-1])
        self.assertNotIn(record["conversations"][-1]["value"], prompt)

    def test_prior_reasoning_can_name_a_future_planned_step(self):
        item = trajectory()
        item["steps"][0]["think"] = "Inspect items, then do step_3 to answer."
        record, _ = convert_step(item, 1, 4, "rolling system")
        self.assertIn("step_3", record["conversations"][1]["value"])

    def test_plan_item_id_can_match_current_action_step(self):
        item = trajectory()
        item["steps"][1]["environment_state_before"] = {
            "plan": [{"id": "step_3", "goal": "Return the answer", "status": "pending"}],
            "tables": {},
            "values": {},
        }
        record, _ = convert_step(item, 1, 4, "rolling system")
        self.assertIn('"id":"step_3"', record["conversations"][2]["value"])

    def test_current_factual_provenance_reference_is_rejected(self):
        item = trajectory()
        item["steps"][1]["environment_state_before"] = {
            "plan": [],
            "tables": {"items": {"schema": {"from_step": "step_3"}}},
            "values": {},
        }
        with self.assertRaisesRegex(ValueError, "future/current reference step_3"):
            convert_step(item, 1, 4, "rolling system")

    def test_prior_schema_output_is_compact_but_current_state_remains(self):
        item = trajectory()
        item["steps"][0]["tool_output"] = {
            "tables": [{
                "table_name": "items",
                "row_count": 2,
                "columns": [{"name": "id", "type": "INTEGER"}],
            }],
        }
        item["steps"][1]["environment_state_before"] = {
            "plan": [],
            "tables": {
                "items": {
                    "schema": {
                        "from_step": "step_1",
                        "columns": [{"name": "id", "type": "INTEGER"}],
                    },
                },
            },
            "values": {},
        }
        record, _ = convert_step(item, 1, 4, "rolling system")
        user_context = record["conversations"][2]["value"]
        observation, current_state = user_context.split("\n\nCURRENT ENVIRONMENT STATE\n", 1)
        self.assertNotIn('"type":"INTEGER"', observation)
        self.assertIn('"column_count":1', observation)
        self.assertIn('"type":"INTEGER"', current_state)

    def test_quality_gate_rejects_current_step_evidence(self):
        item = trajectory()
        item["label_status"] = "verified"
        item["steps"][0]["tool_call"] = {
            "tool": "plan",
            "arguments": {
                "ops": [{
                    "op": "create",
                    "id": "p1",
                    "goal": "Inspect items",
                    "status": "in_progress",
                    "evidence": "step_1",
                }],
            },
        }
        self.assertEqual(
            quality_reason(item, max_steps=12, max_think_words=300),
            "current_or_future_reference",
        )

    def test_sft2_quality_gate_can_disable_length_limits(self):
        item = trajectory()
        item["label_status"] = "verified"
        item["steps"][0]["think"] = "reason " * 500
        item["steps"][1]["tool_call"]["arguments"] = {
            "evidence": {"table": "items"},
        }
        self.assertIsNone(
            quality_reason(item, max_steps=None, max_think_words=None),
        )

    def test_sft2_quality_gate_still_rejects_identical_calls(self):
        item = trajectory()
        item["label_status"] = "verified"
        item["steps"][1]["tool_call"] = item["steps"][0]["tool_call"]
        self.assertEqual(
            quality_reason(item, max_steps=None, max_think_words=None),
            "adjacent_repeated_call",
        )

    def test_sft2_quality_gate_allows_non_adjacent_identical_calls(self):
        item = {
            "label_status": "verified",
            "steps": [
                {
                    "step_id": "step_1",
                    "think": "Inspect items.",
                    "tool_call": {
                        "tool": "describe_table",
                        "arguments": {"tables": ["items"]},
                    },
                },
                {
                    "step_id": "step_2",
                    "think": "Inspect the value domain.",
                    "tool_call": {
                        "tool": "inspect_column",
                        "arguments": {"table": "items", "column": "category"},
                    },
                },
                {
                    "step_id": "step_3",
                    "think": "Revisit the schema after other work.",
                    "tool_call": {
                        "tool": "describe_table",
                        "arguments": {"tables": ["items"]},
                    },
                },
            ],
        }
        self.assertIsNone(
            quality_reason(item, max_steps=None, max_think_words=None)
        )


if __name__ == "__main__":
    unittest.main()
