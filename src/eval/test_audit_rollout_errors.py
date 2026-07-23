#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from audit_rollout_errors import audit, execution_cluster, protocol_cluster  # noqa: E402


class RolloutErrorAuditTests(unittest.TestCase):
    def test_classifies_common_failure_shapes(self):
        self.assertEqual(
            protocol_cluster(
                "ProtocolError: expected exactly one non-empty <think>...</think> block"
            ),
            "missing_or_invalid_think",
        )
        self.assertEqual(
            execution_cluster("OperationalError: no such column: joined.total"),
            "column_reference",
        )
        self.assertEqual(
            execution_cluster("ValueError: x is not an introduced logical column"),
            "join_logical_namespace",
        )

    def test_audits_nested_samples_and_infers_protocol_tool_from_raw_json(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.jsonl"
            path.write_text(json.dumps({
                "example_index": 7,
                "dataset_split": "train",
                "question": "test",
                "samples": [{
                    "sample_index": 0,
                    "correct": True,
                    "legal": True,
                    "failure_type": None,
                    "error_events": [{
                        "action_index": 1,
                        "error_type": "protocol_error",
                        "message": (
                            "ProtocolError: expected exactly one non-empty "
                            "<think>...</think> block"
                        ),
                    }],
                    "turns": [{
                        "turn_index": 0,
                        "raw_model_output": (
                            '{"tool":"describe_table","arguments":{"tables":["items"]}}'
                        ),
                    }],
                }],
            }) + "\n", encoding="utf-8")
            report = audit([path], "nested")

        self.assertEqual(report["episodes"], 1)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["error_event_histogram"], {"protocol_error": 1})
        self.assertEqual(
            report["error_tools_by_type"]["protocol_error"],
            {"describe_table": 1},
        )
        self.assertEqual(
            report["error_tool_attribution_source"],
            {"raw_visible_inference": 1},
        )


if __name__ == "__main__":
    unittest.main()
