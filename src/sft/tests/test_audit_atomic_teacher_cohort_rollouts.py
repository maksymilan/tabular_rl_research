from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_atomic_teacher_cohort_rollouts import build_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class AtomicTeacherCohortRolloutAuditTests(unittest.TestCase):
    def _inputs(self, root: Path, *, diagnostic: bool = True) -> dict[str, Path]:
        cohort = [
            {
                "example_id": "task_1",
                "db_id": "db_a",
                "question": "Count the matching rows.",
                "external_knowledge": "match means active",
            },
            {
                "example_id": "task_2",
                "db_id": "db_b",
                "question": "List every requested name.",
                "external_knowledge": "",
            },
        ]
        profiles = [
            {
                "example_id": row["example_id"],
                "tool_proxy_counts": {
                    "condition_filter": 1,
                    "project": 1,
                    "scalar_compute": 0,
                    "join_tables": 0,
                    "group_aggregate": 1,
                    "extreme_value_select": 0,
                    "set_op": 0,
                    "answer_from_context": 1,
                },
            }
            for row in cohort
        ]
        admission = (
            "diagnostic_only_pending_protocol_scale_gate"
            if diagnostic
            else "eligible_by_current_context_contract"
        )
        verified = [{
            "trajectory_id": "task_1",
            "tool_scheme": "native-tool-bundle",
            "label_status": "verified",
            "training_admission": admission,
            "sft_export_eligible": not diagnostic,
            "steps": [
                {"tool_call": {"tool": "describe_table", "arguments": {}}},
                {"tool_call": {"tool": "condition_filter", "arguments": {}}},
                {"tool_call": {"tool": "group_aggregate", "arguments": {}}},
                {"tool_call": {"tool": "answer_from_context", "arguments": {}}},
            ],
        }]
        attempts = [
            {
                "trajectory_id": "task_1",
                "attempt_index": 1,
                "correct": True,
                "legal": True,
                "turns": [{
                    "native_bundle_results": [
                        {"tool": "describe_table", "status": "success"},
                        {"tool": "condition_filter", "status": "success"},
                    ]
                }],
            },
            {
                "trajectory_id": "task_2",
                "attempt_index": 1,
                "correct": False,
                "legal": False,
                "turns": [{
                    "native_bundle_results": [
                        {"tool": "project", "status": "error"},
                    ]
                }],
            },
        ]
        paths = {
            "cohort_manifest": root / "cohort.manifest.json",
            "cohort": root / "cohort.jsonl",
            "profiles": root / "profiles.jsonl",
            "verified": root / "verified.jsonl",
            "all": root / "all.jsonl",
        }
        paths["cohort_manifest"].write_text(
            json.dumps({"all_acceptance_gates_passed": True}),
            encoding="utf-8",
        )
        _write_jsonl(paths["cohort"], cohort)
        _write_jsonl(paths["profiles"], profiles)
        _write_jsonl(paths["verified"], verified)
        _write_jsonl(paths["all"], attempts)
        return paths

    def test_reports_actual_usage_and_attrition_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._inputs(Path(directory))
            report = build_report(
                cohort_manifest_path=paths["cohort_manifest"],
                cohort_path=paths["cohort"],
                profile_path=paths["profiles"],
                verified_path=paths["verified"],
                all_path=paths["all"],
            )
        self.assertEqual(report["coverage"]["attempted_unique_tasks"], 2)
        self.assertEqual(report["coverage"]["verified_success_tasks"], 1)
        self.assertEqual(
            report["actual_tool_usage"]["verified_exploration_calls"]["describe_table"],
            1,
        )
        self.assertEqual(
            report["actual_tool_usage"]["all_attempted_error_calls"]["project"],
            1,
        )
        self.assertFalse(
            report["protocol_and_training_boundary"]["candidate_sft_export_allowed"]
        )
        self.assertIsNotNone(report["verified_success_attrition_distribution"])

    def test_current_single_action_admission_is_reported_but_not_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._inputs(Path(directory), diagnostic=False)
            report = build_report(
                cohort_manifest_path=paths["cohort_manifest"],
                cohort_path=paths["cohort"],
                profile_path=paths["profiles"],
                verified_path=paths["verified"],
                all_path=paths["all"],
            )
        self.assertTrue(
            report["protocol_and_training_boundary"]["candidate_sft_export_allowed"]
        )

    def test_empty_result_must_be_context_only_and_empty_terminal_blocks_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._inputs(Path(directory), diagnostic=False)
            verified = [json.loads(line) for line in paths["verified"].read_text().splitlines()]
            verified[0]["steps"][1].update({
                "step_id": "step_2",
                "tool_output": {"table": "filter_001", "row_count": 0},
                "sft_target_eligible": False,
                "sft_target_exclusion_reason": "empty_table_result",
            })
            verified[0]["steps"][-1].update({
                "step_id": "step_4",
                "tool_call": {
                    "tool": "answer_from_context",
                    "arguments": {"evidence": {"table": "filter_001"}},
                },
                "environment_state_before": {
                    "tables": {"filter_001": {"row_count": 0}}
                },
            })
            _write_jsonl(paths["verified"], verified)
            report = build_report(
                cohort_manifest_path=paths["cohort_manifest"],
                cohort_path=paths["cohort"],
                profile_path=paths["profiles"],
                verified_path=paths["verified"],
                all_path=paths["all"],
            )
        empty = report["protocol_and_training_boundary"]["empty_result_training_filter"]
        self.assertEqual(empty["empty_result_steps"], 1)
        self.assertEqual(empty["unmarked_empty_result_target_steps"], 0)
        self.assertEqual(empty["empty_terminal_trajectories"], 1)
        self.assertFalse(report["protocol_and_training_boundary"]["candidate_sft_export_allowed"])

    def test_rejects_attempts_outside_frozen_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._inputs(Path(directory))
            with paths["all"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"trajectory_id": "other", "turns": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "outside frozen cohort"):
                build_report(
                    cohort_manifest_path=paths["cohort_manifest"],
                    cohort_path=paths["cohort"],
                    profile_path=paths["profiles"],
                    verified_path=paths["verified"],
                    all_path=paths["all"],
                )


if __name__ == "__main__":
    unittest.main()
