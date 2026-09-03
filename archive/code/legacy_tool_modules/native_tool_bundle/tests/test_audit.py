#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "sft")]

from tool_modules.native_tool_bundle.audit import audit_records  # noqa: E402


def _call(call_id: str, table: str = "items") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "describe_table",
            "arguments": json.dumps({"tables": [table]}, separators=(",", ":")),
        },
    }


def _result(call_id: str, table: str = "items") -> dict:
    state = {"tables": {}}
    return {
        "call_id": call_id,
        "tool": "describe_table",
        "arguments": {"tables": [table]},
        "status": "success",
        "environment_state_before": state,
        "environment_state": state,
    }


class AuditNativeToolBundleTests(unittest.TestCase):
    def test_audit_accepts_exact_lowering_and_rejects_argument_change(self):
        assistant_a = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Inspect schema.",
            "tool_calls": [_call("a")],
        }
        record = {
            "trajectory_id": "episode",
            "tool_scheme": "native-tool-bundle",
            "gold_sql": "SELECT name FROM items",
            "turns": [
                {
                    "model_input": [{"role": "user", "content": "question"}],
                    "provider_reasoning_content": "Inspect schema.",
                    "provider_native_tool_call": [_call("a")],
                    "parsed": {
                        "calls": [{
                            "id": "a",
                            "tool": "describe_table",
                            "arguments": {"tables": ["items"]},
                        }]
                    },
                    "native_bundle_results": [_result("a")],
                },
                {
                    "model_input": [
                        assistant_a,
                        {"role": "tool", "tool_call_id": "a", "content": "result"},
                    ],
                    "provider_reasoning_content": "Inspect again.",
                    "provider_native_tool_call": [_call("b")],
                    "parsed": {
                        "calls": [{
                            "id": "b",
                            "tool": "describe_table",
                            "arguments": {"tables": ["items"]},
                        }]
                    },
                    "native_bundle_results": [_result("b")],
                },
            ],
        }
        self.assertEqual(audit_records([record])["gate"], "pass")
        changed = copy.deepcopy(record)
        changed["turns"][1]["parsed"]["calls"][0]["arguments"] = {
            "tables": ["other"]
        }
        self.assertEqual(audit_records([changed])["gate"], "fail")

    def test_version54_audit_rejects_plan_in_provider_or_history(self):
        plan_call = {
            "id": "p",
            "type": "function",
            "function": {"name": "plan", "arguments": '{"ops":[]}'},
        }
        state = {"tables": {}}
        record = {
            "trajectory_id": "no-plan-episode",
            "tool_scheme": "native-tool-bundle",
            "protocol_version": "version54",
            "turns": [{
                "model_input": [{"role": "user", "content": "question"}],
                "provider_reasoning_content": "Plan first.",
                "provider_native_tool_call": [plan_call],
                "parsed": {"calls": [{"id": "p", "tool": "plan", "arguments": {"ops": []}}]},
                "native_bundle_results": [{
                    "call_id": "p",
                    "tool": "plan",
                    "arguments": {"ops": []},
                    "status": "error",
                    "environment_state_before": state,
                    "environment_state": state,
                }],
            }],
            "provider_native_history": [{
                "tool_messages": [{
                    "role": "tool",
                    "tool_call_id": "p",
                    "content": '{"error":{"type":"protocol_error"}}',
                }],
            }],
        }
        report = audit_records([record])
        self.assertEqual(report["gate"], "fail")
        self.assertTrue(any("version54 provider call contains plan" in item for item in report["issues"]))


if __name__ == "__main__":
    unittest.main()
