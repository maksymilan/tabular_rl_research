from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from generate_teacher_rollouts import (  # noqa: E402
    PLAN_POLICY_OPTIONAL,
    PLAN_POLICY_REQUIRED_RESIDENT,
    ResidentPlanPolicyTracker,
    retain_in_rolling_history,
)
from protocol import ProtocolError  # noqa: E402


INITIAL_PLAN = {
    "ops": [
        {"op": "create", "id": "inspect", "goal": "Inspect relevant tables", "status": "pending"},
        {"op": "create", "id": "solve", "goal": "Build exact answer table", "status": "pending"},
    ]
}


class RequiredResidentPlanTest(unittest.TestCase):
    def test_optional_policy_does_not_restrict_actions(self):
        tracker = ResidentPlanPolicyTracker(PLAN_POLICY_OPTIONAL)
        tracker.validate_before_execution("describe_table", {"tables": ["t"]})
        tracker.record_success("describe_table", {"tables": ["t"]})
        tracker.validate_before_execution("answer_from_context", {"evidence": {"table": "t"}})

    def test_required_policy_rejects_non_plan_first_action(self):
        tracker = ResidentPlanPolicyTracker(PLAN_POLICY_REQUIRED_RESIDENT)
        with self.assertRaisesRegex(ProtocolError, "first valid action must be plan"):
            tracker.validate_before_execution("describe_table", {"tables": ["t"]})

    def test_required_policy_requires_meaningful_initial_plan(self):
        tracker = ResidentPlanPolicyTracker(PLAN_POLICY_REQUIRED_RESIDENT)
        with self.assertRaisesRegex(ProtocolError, "2 to 4"):
            tracker.validate_before_execution(
                "plan",
                {"ops": [{"op": "create", "id": "only", "goal": "Do everything"}]},
            )

    def test_terminal_requires_update_after_data_work(self):
        tracker = ResidentPlanPolicyTracker(PLAN_POLICY_REQUIRED_RESIDENT)
        tracker.validate_before_execution("plan", INITIAL_PLAN)
        tracker.record_success("plan", INITIAL_PLAN)
        tracker.validate_before_execution("describe_table", {"tables": ["t"]})
        tracker.record_success("describe_table", {"tables": ["t"]})
        with self.assertRaisesRegex(ProtocolError, "plan must be updated"):
            tracker.validate_before_execution(
                "answer_from_context",
                {"evidence": {"table": "result"}},
            )

    def test_update_after_work_allows_terminal(self):
        tracker = ResidentPlanPolicyTracker(PLAN_POLICY_REQUIRED_RESIDENT)
        tracker.validate_before_execution("plan", INITIAL_PLAN)
        tracker.record_success("plan", INITIAL_PLAN)
        tracker.validate_before_execution("describe_table", {"tables": ["t"]})
        tracker.record_success("describe_table", {"tables": ["t"]})
        update = {
            "ops": [
                {
                    "op": "update",
                    "id": "inspect",
                    "status": "done",
                    "evidence": "step_2",
                }
            ]
        }
        tracker.validate_before_execution("plan", update)
        tracker.record_success("plan", update)
        tracker.validate_before_execution(
            "answer_from_context",
            {"evidence": {"table": "result"}},
        )
        with self.assertRaisesRegex(ProtocolError, "new non-plan progress"):
            tracker.validate_before_execution("plan", update)
        tracker.validate_before_execution("project", {"table": "t", "expressions": ["id"]})
        tracker.record_success("project", {"table": "t", "expressions": ["id"]})
        tracker.validate_before_execution("plan", update)

    def test_required_plan_actions_are_resident_only(self):
        self.assertFalse(retain_in_rolling_history("plan", PLAN_POLICY_REQUIRED_RESIDENT))
        self.assertTrue(retain_in_rolling_history("plan", PLAN_POLICY_OPTIONAL))
        self.assertTrue(
            retain_in_rolling_history("describe_table", PLAN_POLICY_REQUIRED_RESIDENT)
        )


if __name__ == "__main__":
    unittest.main()
