#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "sft")]

from tool_modules.native_tool_bundle.bundle_credit import (  # noqa: E402
    aggregate_summaries,
    analyze_record,
)


class NativeBundleCreditTests(unittest.TestCase):
    def test_error_feedback_and_later_correction_are_counted_without_dropping_error(self):
        error_payload = {
            "error": {
                "type": "argument_validation_error",
                "code": "unknown_column",
                "message": "column missing",
            }
        }
        record = {
            "turns": [
                {
                    "turn_index": 0,
                    "parsed": {"calls": [{
                        "id": "bad",
                        "tool": "condition_filter",
                        "arguments": {
                            "table": "items",
                            "conditions": {"column": "titel", "op": "=", "value": "x"},
                        },
                    }]},
                    "native_bundle_results": [{
                        "call_id": "bad",
                        "tool": "condition_filter",
                        "status": "error",
                    }],
                },
                {
                    "turn_index": 1,
                    "parsed": {"calls": [{
                        "id": "fixed",
                        "tool": "condition_filter",
                        "arguments": {
                            "table": "items",
                            "conditions": {"column": "title", "op": "=", "value": "x"},
                        },
                    }]},
                    "native_bundle_results": [{
                        "call_id": "fixed",
                        "step_id": "step_2",
                        "tool": "condition_filter",
                        "status": "success",
                        "table": "filter_001",
                    }],
                },
                {
                    "turn_index": 2,
                    "parsed": {"calls": [{
                        "id": "answer",
                        "tool": "answer_from_context",
                        "arguments": {"evidence": {"table": "filter_001"}},
                    }]},
                    "native_bundle_results": [{
                        "call_id": "answer",
                        "tool": "answer_from_context",
                        "status": "success",
                        "output": {"correct": True},
                    }],
                },
            ],
            "provider_native_history": [{
                "model_turn_index": 1,
                "assistant": {"tool_calls": [{"id": "bad"}]},
                "tool_messages": [{
                    "role": "tool",
                    "tool_call_id": "bad",
                    "content": json.dumps(error_payload),
                }],
            }],
        }
        summary = analyze_record(record)
        counts = summary["counts"]
        self.assertEqual(counts["error_calls"], 1)
        self.assertEqual(counts["error_feedback_preserved"], 1)
        self.assertEqual(counts["corrected_same_tool_next_turn"], 1)
        self.assertEqual(counts["created_handle_referenced_later"], 1)
        self.assertEqual(summary["error_feedback_preservation_rate"], 1.0)
        self.assertEqual(
            summary["error_target_policy"],
            "preserve_error_turn_and_feedback_as_history; target_later_corrected_bundle",
        )

    def test_perception_credit_remains_unresolved_and_aggregate_is_fact_only(self):
        record = {
            "turns": [{
                "turn_index": 0,
                "parsed": {"calls": [{
                    "id": "schema",
                    "tool": "describe_table",
                    "arguments": {"tables": ["items"]},
                }]},
                "native_bundle_results": [{
                    "call_id": "schema",
                    "tool": "describe_table",
                    "status": "success",
                }],
            }],
        }
        summary = analyze_record(record)
        self.assertEqual(summary["counts"]["perception_credit_unresolved"], 1)
        self.assertEqual(
            summary["call_annotations"][0]["utility_class"],
            "observational_credit_unresolved",
        )
        aggregate = aggregate_summaries([summary, summary])
        self.assertTrue(aggregate["statistics_only_not_reward"])
        self.assertEqual(aggregate["counts"]["perception_credit_unresolved"], 2)


if __name__ == "__main__":
    unittest.main()
