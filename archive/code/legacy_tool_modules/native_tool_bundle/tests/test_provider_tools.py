#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "sft")]

from tool_modules.native_tool_bundle.provider_tools import (  # noqa: E402
    NATIVE_ACTION_RULE,
    DeepSeekNativeAtomicDriver,
    NativeToolCallError,
    canonical_messages_to_native,
    native_bundle_history_messages,
    native_atomic_tools,
    native_system_prompt,
)
from prompt_contract import CANONICAL_ACTION_RULE, build_student_system_prompt  # noqa: E402
from protocol import MODEL_ARG_SCHEMA, assistant_message, parse_assistant_strict  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _provider_response(
    *,
    tool: str = "describe_table",
    arguments: dict | str | None = None,
    call_id: str = "call_official_1",
    reasoning: str = "I need the relevant schema first.",
    extra_calls: list[dict] | None = None,
) -> dict:
    if arguments is None:
        arguments = {"tables": ["suppliers"]}
    raw_arguments = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    )
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": raw_arguments},
        }
    ]
    calls.extend(extra_calls or [])
    return {
        "id": "response_1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning,
                    "tool_calls": calls,
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class DeepSeekNativeToolsTests(unittest.TestCase):
    def test_native_tools_exactly_cover_current_atomic_schema(self):
        tools = native_atomic_tools()
        by_name = {item["function"]["name"]: item for item in tools}
        self.assertEqual(set(by_name), set(MODEL_ARG_SCHEMA))
        for name, (required, optional) in MODEL_ARG_SCHEMA.items():
            parameters = by_name[name]["function"]["parameters"]
            self.assertEqual(set(parameters["properties"]), required | optional)
            self.assertEqual(set(parameters["required"]), required)
            self.assertFalse(parameters["additionalProperties"])

    def test_native_prompt_replaces_only_the_text_carrier_rule(self):
        canonical = build_student_system_prompt()
        native = native_system_prompt(canonical)
        self.assertNotIn(CANONICAL_ACTION_RULE, native)
        self.assertNotIn("<think>", native)
        self.assertIn(NATIVE_ACTION_RULE, native)
        self.assertIn("answer_from_context(evidence, reason?)", native)

    def test_fallback_history_becomes_assistant_tool_pair(self):
        canonical = assistant_message(
            "Inspect the supplier schema.",
            "describe_table",
            {"tables": ["suppliers"]},
        )
        converted = canonical_messages_to_native(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": canonical},
                {"role": "user", "content": '{"step_id":"step_1"}'},
            ]
        )
        self.assertEqual([item["role"] for item in converted], ["system", "user", "assistant", "tool"])
        self.assertEqual(converted[2]["reasoning_content"], "Inspect the supplier schema.")
        self.assertEqual(converted[2]["tool_calls"][0]["id"], converted[3]["tool_call_id"])
        self.assertEqual(converted[2]["tool_calls"][0]["function"]["name"], "describe_table")

    def test_live_call_uses_official_native_payload_and_preserves_response(self):
        driver = DeepSeekNativeAtomicDriver(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        response = _provider_response()
        with patch(
            "tool_modules.native_tool_bundle.provider_tools.urllib.request.urlopen",
            return_value=_FakeResponse(response),
        ) as mocked:
            retry_stats: dict = {}
            canonical = driver(
                "https://api.deepseek.com",
                "deepseek-v4-flash",
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                ],
                max_tokens=512,
                retries=0,
                retry_stats=retry_stats,
            )
        think, tool, arguments = parse_assistant_strict(canonical)
        self.assertEqual(think, "I need the relevant schema first.")
        self.assertEqual(tool, "describe_table")
        self.assertEqual(arguments, {"tables": ["suppliers"]})
        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(set(item["function"]["name"] for item in payload["tools"]), set(MODEL_ARG_SCHEMA))
        self.assertEqual(retry_stats["api_request_attempts"], 1)

        converted = canonical_messages_to_native(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": canonical},
                {"role": "user", "content": "tool result"},
            ],
            driver._native_assistant_history,
        )
        self.assertEqual(converted[2]["tool_calls"][0]["id"], "call_official_1")
        self.assertEqual(converted[2]["reasoning_content"], response["choices"][0]["message"]["reasoning_content"])
        self.assertEqual(converted[3]["tool_call_id"], "call_official_1")

    def test_repeated_nonadjacent_calls_keep_distinct_provider_ids(self):
        canonical_a = assistant_message(
            "Inspect suppliers.", "describe_table", {"tables": ["suppliers"]}
        )
        canonical_b = assistant_message(
            "Inspect orders.", "describe_table", {"tables": ["orders"]}
        )
        history = [
            (
                canonical_a,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect suppliers.",
                    "tool_calls": [{"id": "call_a1", "type": "function", "function": {"name": "describe_table", "arguments": '{"tables":["suppliers"]}'}}],
                },
            ),
            (
                canonical_b,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect orders.",
                    "tool_calls": [{"id": "call_b", "type": "function", "function": {"name": "describe_table", "arguments": '{"tables":["orders"]}'}}],
                },
            ),
            (
                canonical_a,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect suppliers.",
                    "tool_calls": [{"id": "call_a2", "type": "function", "function": {"name": "describe_table", "arguments": '{"tables":["suppliers"]}'}}],
                },
            ),
        ]
        converted = canonical_messages_to_native(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": canonical_a},
                {"role": "user", "content": "result a1"},
                {"role": "assistant", "content": canonical_b},
                {"role": "user", "content": "result b"},
                {"role": "assistant", "content": canonical_a},
                {"role": "user", "content": "result a2"},
            ],
            history,
        )
        assistant_ids = [
            item["tool_calls"][0]["id"]
            for item in converted
            if item["role"] == "assistant"
        ]
        self.assertEqual(assistant_ids, ["call_a1", "call_b", "call_a2"])

    def test_native_bundle_history_keeps_one_result_per_call_in_order(self):
        assistant = {
            "role": "assistant",
            "content": "ignored prose",
            "reasoning_content": "Inspect both relevant columns.",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "inspect_column",
                        "arguments": '{"table":"items","column":"name"}',
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "inspect_column",
                        "arguments": '{"table":"items","column":"category"}',
                    },
                },
            ],
        }
        rendered = native_bundle_history_messages(
            system_prompt="system",
            overview={"tables": []},
            question="question",
            state=None,
            last_error=None,
            external_knowledge=None,
            history=[{
                "assistant": assistant,
                "tool_messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": '{"step_id":"step_1","output":{"ok":true}}',
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_b",
                        "content": '{"step_id":"step_2","output":{"ok":true}}',
                    },
                ],
            }],
            history_turns=4,
        )
        self.assertEqual(
            [message["role"] for message in rendered],
            ["system", "user", "assistant", "tool", "tool"],
        )
        self.assertEqual(rendered[2]["reasoning_content"], assistant["reasoning_content"])
        self.assertEqual(
            [rendered[3]["tool_call_id"], rendered[4]["tool_call_id"]],
            ["call_a", "call_b"],
        )

    def test_multiple_calls_are_rejected_before_execution(self):
        driver = DeepSeekNativeAtomicDriver(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        extra = {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "inspect_column",
                "arguments": '{"table":"suppliers","column":"country"}',
            },
        }
        with self.assertRaisesRegex(NativeToolCallError, "exactly one tool call"):
            driver._lower_response(_provider_response(extra_calls=[extra]))

    def test_malformed_arguments_and_unknown_tool_are_rejected(self):
        driver = DeepSeekNativeAtomicDriver(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        with self.assertRaisesRegex(NativeToolCallError, "not valid JSON"):
            driver._lower_response(_provider_response(arguments="{"))
        with self.assertRaisesRegex(NativeToolCallError, "unknown atomic tool"):
            driver._lower_response(_provider_response(tool="run_sql", arguments={}))


if __name__ == "__main__":
    unittest.main()
