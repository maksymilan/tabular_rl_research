from __future__ import annotations

import unittest

from training_result_quality import (
    EMPTY_RESULT_TARGET_REASON,
    apply_empty_result_target_annotation,
    is_empty_row_result,
    training_quality_summary,
    trajectory_has_empty_terminal_evidence,
)


class TrainingResultQualityTests(unittest.TestCase):
    def test_empty_relation_is_context_only_but_scalar_zero_is_not_empty(self) -> None:
        step = {
            "step_id": "step_1",
            "tool_output": {"table": "filter_001", "row_count": 0, "columns": ["id"]},
        }
        apply_empty_result_target_annotation(step)
        self.assertFalse(step["sft_target_eligible"])
        self.assertEqual(step["sft_target_exclusion_reason"], EMPTY_RESULT_TARGET_REASON)
        self.assertFalse(is_empty_row_result({"row_count": 1, "rows": [[0]]}))

    def test_empty_terminal_evidence_excludes_whole_trajectory(self) -> None:
        terminal = {
            "step_id": "step_3",
            "tool_call": {
                "tool": "answer_from_context",
                "arguments": {"evidence": {"table": "project_001"}},
            },
            "environment_state_before": {
                "tables": {"project_001": {"row_count": 0}}
            },
        }
        self.assertTrue(trajectory_has_empty_terminal_evidence([terminal]))
        summary = training_quality_summary([terminal])
        self.assertTrue(summary["empty_terminal_evidence"])
        self.assertFalse(summary["trajectory_sft_eligible_under_empty_result_policy"])

    def test_zero_valued_scalar_table_remains_eligible(self) -> None:
        terminal = {
            "tool_call": {
                "tool": "answer_from_context",
                "arguments": {"evidence": {"table": "group_001"}},
            },
            "environment_state_before": {
                "tables": {"group_001": {"row_count": 1}}
            },
        }
        self.assertFalse(trajectory_has_empty_terminal_evidence([terminal]))


if __name__ == "__main__":
    unittest.main()
