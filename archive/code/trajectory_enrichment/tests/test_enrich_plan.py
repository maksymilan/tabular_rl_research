#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from enrich_plan import validate_plan_payload  # noqa: E402


def tiny_traj() -> dict:
    return {
        "trajectory_id": "traj_plan",
        "question": "Which names are in one set but not another?",
        "steps": [
            {
                "step_id": "step_1",
                "tool_call": {"tool": "set_op", "arguments": {"left": "a", "right": "b", "op": "except"}},
            },
            {
                "step_id": "step_2",
                "tool_call": {"tool": "answer_from_context", "arguments": {"answer": [["x"]]}},
            },
        ],
    }


class EnrichPlanValidationTests(unittest.TestCase):
    def test_rejects_pending_final_like_goal_before_answer(self):
        payload = {
            "initial_think": "I split the task into set construction and final combination.",
            "initial_ops": [
                {"op": "create", "id": "p1", "goal": "find the left set", "status": "done"},
                {"op": "create", "id": "p2", "goal": "combine both sets and answer", "status": "pending"},
            ],
            "updates": [],
        }

        issues = validate_plan_payload(payload, tiny_traj(), {"min_updates": 0, "max_updates": 2})

        self.assertTrue(any("final-like plan item p2 is still" in issue for issue in issues))

    def test_allows_done_final_like_goal_with_evidence(self):
        payload = {
            "initial_think": "I split the task into set construction and final combination.",
            "initial_ops": [
                {"op": "create", "id": "p1", "goal": "find the left set", "status": "pending"},
                {"op": "create", "id": "p2", "goal": "combine both sets and answer", "status": "pending"},
            ],
            "updates": [
                {
                    "after_step": 1,
                    "think": "I have the set difference result, so I can close the final combination goal.",
                    "ops": [
                        {"op": "update", "id": "p1", "status": "done", "evidence": "step_1"},
                        {"op": "update", "id": "p2", "status": "done", "evidence": "step_1"},
                    ],
                }
            ],
        }

        issues = validate_plan_payload(payload, tiny_traj(), {"min_updates": 0, "max_updates": 2})

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
