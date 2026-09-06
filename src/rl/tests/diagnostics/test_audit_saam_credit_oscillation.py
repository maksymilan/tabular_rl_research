from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.scenarios.diagnostics.audit_saam_credit_oscillation import audit_rollouts  # noqa: E402


def _turn(tool: str, table: str) -> dict:
    return {
        "turn_index": 0,
        "parsed": {
            "think": "not reward authority",
            "tool": tool,
            "arguments": {"tables": [table]},
        },
    }


def _row(
    *,
    policy_step: int,
    sample_index: int,
    correct: bool,
    table: str,
) -> dict:
    return {
        "policy_global_step": policy_step,
        "example_index": 1,
        "trajectory_id": f"step{policy_step}_sample{sample_index}",
        "sample_index": sample_index,
        "correct": correct,
        "process_update": True,
        "turns": [_turn("describe_table", table)],
        "gold_sql": object(),
    }


def _three_step_rows() -> list[dict]:
    return [
        # Step 0: exact same semantic action receives opposite terminal credit.
        _row(policy_step=0, sample_index=0, correct=True, table="orders"),
        _row(policy_step=0, sample_index=1, correct=False, table="orders"),
        # Step 1: orders is positive and users is negative.
        _row(policy_step=1, sample_index=0, correct=True, table="orders"),
        _row(policy_step=1, sample_index=1, correct=False, table="users"),
        # Step 2: signs reverse across optimizer batches.
        _row(policy_step=2, sample_index=0, correct=False, table="orders"),
        _row(policy_step=2, sample_index=1, correct=True, table="users"),
    ]


class SAAMCreditOscillationAuditTest(unittest.TestCase):
    def test_legacy_raw_rollout_uses_structured_exclusion(self):
        rows = _three_step_rows()[:2]
        for row in rows:
            row.pop("process_update")
            row["result_reward"] = {
                "profile": "binary",
                "correct": row["correct"],
                "executable_terminal": True,
                "value": float(row["correct"]),
            }
        rows[1]["optimization_exclusion"] = "nonsemantic_runtime_failure"
        result = audit_rollouts(
            rows,
            credit_assignment="trajectory",
            expected_group_size=2,
        )
        self.assertEqual(result["eligible_episodes"], 1)
        self.assertEqual(result["groups_with_mixed_reward"], 0)

    def test_saam_eliminates_batch_local_mixed_key_credit(self):
        vanilla = audit_rollouts(
            _three_step_rows(),
            credit_assignment="trajectory",
            expected_group_size=2,
        )
        saam = audit_rollouts(
            _three_step_rows(),
            credit_assignment="saam-strict",
            expected_group_size=2,
        )
        self.assertGreater(vanilla["applied_two_sided_direct_conflict_mass"], 0.0)
        self.assertEqual(saam["applied_two_sided_direct_conflict_mass"], 0.0)
        self.assertTrue(saam["batch_local_guarantee_passed"])
        self.assertFalse(saam["gold_sql_read"])

    def test_saam_does_not_claim_to_prevent_cross_batch_sign_reversal(self):
        saam = audit_rollouts(
            _three_step_rows(),
            credit_assignment="saam-strict",
            expected_group_size=2,
        )
        self.assertGreater(saam["recurrent_semantic_keys"], 0)
        self.assertGreater(saam["credit_sign_flips"], 0)
        self.assertGreater(
            saam["credit_sign_flips_per_recurrent_transition"], 0.0
        )
        self.assertFalse(saam["cross_update_no_flip_guaranteed"])

    def test_group_size_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "group size"):
            audit_rollouts(
                _three_step_rows()[:-1],
                credit_assignment="saam-strict",
                expected_group_size=2,
            )


if __name__ == "__main__":
    unittest.main()
