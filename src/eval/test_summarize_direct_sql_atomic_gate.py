from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_direct_sql_atomic_gate import build_summary, exact_mcnemar


def _record(task_id: str, *, direct: bool, correct: bool, steps: int) -> dict:
    return {
        "instance_id" if direct else "trajectory_id": task_id,
        "correct": correct,
        "legal": True,
        "steps": steps,
        "errors": 0,
        "failure_type": None if correct else "wrong_answer",
        "outcome": "clean_success" if correct else None,
        "protocol_version": "direct-sql-search-v2" if direct else "version39",
        "tool_scheme": "direct-sql-search" if direct else "atomic",
        "usage": {"total_tokens": steps * 100},
        "turns": [],
        "error_events": [],
    }


class DirectSqlAtomicGateSummaryTest(unittest.TestCase):
    def test_build_summary_pairs_outcomes_and_applies_preregistered_gates(self) -> None:
        direct = [
            _record("a", direct=True, correct=True, steps=2),
            _record("b", direct=True, correct=False, steps=3),
            _record("c", direct=True, correct=False, steps=2),
        ]
        atomic = [
            _record("a", direct=False, correct=False, steps=4),
            _record("b", direct=False, correct=True, steps=5),
            _record("c", direct=False, correct=True, steps=6),
        ]
        summary = build_summary(
            direct,
            atomic,
            {
                "retained_diagnostic_success": 1,
                "structural_pass": 3,
                "replay_correct": 1,
            },
            {
                "verified_episodes": 2,
                "structural_gate": "pass",
                "prompt_variant_gate": "pass",
            },
        )

        self.assertEqual(summary["paired"], {
            "tasks": 3,
            "both_correct": 0,
            "direct_only": 1,
            "atomic_only": 2,
            "neither_correct": 0,
            "direct_only_ids": ["a"],
            "atomic_only_ids": ["b", "c"],
            "exact_mcnemar_p": 1.0,
        })
        self.assertEqual(summary["direct"]["retained_diagnostic_success"], 1)
        self.assertEqual(summary["atomic"]["retained_diagnostic_success"], 2)
        self.assertEqual(summary["preregistered_gates"], {
            "engineering_stability": True,
            "worth_larger_fair_comparison": True,
            "no_observed_accuracy_deficit": False,
            "direct_is_at_least_two_lower": False,
        })
        self.assertEqual(summary["training_admission"], 0)

    def test_exact_mcnemar_matches_gate15_discordance(self) -> None:
        self.assertEqual(exact_mcnemar(1, 5), 0.21875)


if __name__ == "__main__":
    unittest.main()
