from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SFT_DIR.parents[0] / "harness"
EVAL_DIR = SFT_DIR.parents[0] / "eval"
sys.path[:0] = [str(SFT_DIR), str(HARNESS_DIR), str(EVAL_DIR)]

from executor import Harness  # noqa: E402
from generate_recovery_teacher_rollouts import (  # noqa: E402
    continuation_quality_reason,
    replay_error_under_current_contract,
)
from protocol import AdjacentActionGuard  # noqa: E402
from rollout import new_ctx, overview, state_digest  # noqa: E402


def step(tool: str, arguments: dict) -> dict:
    return {
        "think": "Use the current state.",
        "tool_call": {"tool": tool, "arguments": arguments},
        "sft_target_eligible": True,
    }


class RecoveryContinuationQualityTests(unittest.TestCase):
    def test_non_adjacent_repeat_is_allowed(self):
        trajectory = {
            "steps": [
                step("read_subtable", {"table": "items", "limit": 1}),
                step("describe_table", {"tables": ["orders"]}),
                step("read_subtable", {"table": "items", "limit": 1}),
            ],
        }
        self.assertIsNone(
            continuation_quality_reason(
                trajectory,
                max_teacher_steps=10,
                max_think_words=100,
            )
        )

    def test_adjacent_repeat_is_rejected(self):
        repeated = step("read_subtable", {"table": "items", "limit": 1})
        trajectory = {"steps": [repeated, dict(repeated)]}
        self.assertEqual(
            continuation_quality_reason(
                trajectory,
                max_teacher_steps=10,
                max_think_words=100,
            ),
            "teacher_adjacent_repeated_call",
        )

    def test_historical_read_error_is_revalidated_under_current_contract(self):
        harness = Harness()
        harness.conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT)")
        harness.conn.executemany(
            "INSERT INTO items(id, value) VALUES (?, ?)",
            [(1, "a"), (2, "b")],
        )
        harness.register_sources()
        ctx = new_ctx(overview(harness))
        digest = state_digest(ctx["environment"].snapshot())
        arguments = {"table": "items", "limit": 1, "offset": 1}
        old_event = {
            "action_index": 1,
            "step_id": "step_1",
            "error_type": "argument_validation_error",
            "error_code": "argument_validation_error",
            "message": "read_subtable has no offset or cursor",
            "state_before_hash": digest,
            "state_after_hash": digest,
            "attempted_tool": "read_subtable",
            "attempted_arguments": arguments,
        }
        guard = AdjacentActionGuard()
        current = replay_error_under_current_contract(
            harness,
            ctx,
            old_event,
            guard,
        )
        self.assertEqual(current["error_type"], "argument_validation_error")
        self.assertIn("order_by", current["message"])
        self.assertNotIn("has no offset or cursor", current["message"])
        self.assertEqual(
            current["source_error_contract"]["error_code"],
            "argument_validation_error",
        )

        repeated_event = {
            **old_event,
            "action_index": 2,
            "step_id": "step_2",
            "error_type": "no_progress_error",
            "error_code": "adjacent_identical_action",
        }
        repeated = replay_error_under_current_contract(
            harness,
            ctx,
            repeated_event,
            guard,
        )
        self.assertEqual(repeated["error_type"], "no_progress_error")
        self.assertEqual(repeated["error_code"], "adjacent_identical_action")
        previous = repeated["details"]["previous_rejection"]
        self.assertIn("order_by", previous["message"])
        harness.conn.close()


if __name__ == "__main__":
    unittest.main()
