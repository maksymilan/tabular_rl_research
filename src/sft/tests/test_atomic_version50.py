#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/eval", "src/harness", "src/sft"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version50 as version50  # noqa: E402
import protocol as version39  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    DATA_GENERATION_SUFFIX,
    chat_with_retries,
    extract_native_function_call,
    protocol_failure_type,
)
from protocol import ProtocolError  # noqa: E402
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
    provider_system_prompt,
)


class AtomicVersion50Tests(unittest.TestCase):
    def test_version50_changes_only_the_provider_carrier_identity(self):
        self.assertEqual(version50.PROTOCOL_VERSION, "version50")
        self.assertEqual(version50.TOOLS, version39.TOOLS)
        self.assertEqual(version50.MODEL_ARG_SCHEMA, version39.MODEL_ARG_SCHEMA)
        self.assertIs(version50.parse_assistant_strict, version39.parse_assistant_strict)
        self.assertEqual(version50.tool_schema_hash(), version39.tool_schema_hash())
        self.assertEqual(
            version50.PROVIDER_ASSISTANT_CARRIER,
            "deepseek-native-function-call-v1",
        )

    def test_version50_prompt_and_hash_are_native_and_distinct(self):
        student = version39.rolling_system_prompt(version39.get_system_prompt())
        canonical_teacher = version39.teacher_system_prompt(student) + DATA_GENERATION_SUFFIX
        provider_prompt = provider_system_prompt(
            "deepseek-v4-flash",
            canonical_teacher,
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        )
        self.assertIn("DEEPSEEK NATIVE FUNCTION-CALL CONTRACT", provider_prompt)
        self.assertNotIn("DEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT", provider_prompt)
        self.assertNotEqual(
            version50.protocol_hash(provider_prompt),
            version39.protocol_hash(provider_prompt),
        )

    def test_native_response_boundary_rejects_zero_multiple_and_content(self):
        base = {"content": None, "reasoning_content": "Choose one call."}
        for tool_calls, expected in (
            ([], "tool_call_count_0"),
            (
                [
                    {"id": "a", "function": {"name": "describe_table", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "describe_table", "arguments": "{}"}},
                ],
                "tool_call_count_2",
            ),
        ):
            with self.subTest(expected=expected):
                action, message, rejection = extract_native_function_call(
                    {**base, "tool_calls": tool_calls}
                )
                self.assertEqual(action, "")
                self.assertEqual(message["tool_calls"], tool_calls)
                self.assertEqual(rejection, expected)
        action, message, rejection = extract_native_function_call(
            {**base, "content": "extra", "tool_calls": []}
        )
        self.assertEqual(action, "")
        self.assertEqual(message["content"], "extra")
        self.assertEqual(rejection, "nonempty_assistant_content")

    def test_multiple_native_calls_consume_one_semantic_turn_without_client_retry(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 40,
            "api_finish_reason": "tool_calls",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "multiple"},
            "provider_native_assistant_message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I will issue two calls.",
                "tool_calls": [
                    {"id": "a", "function": {"name": "describe_table", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "describe_table", "arguments": "{}"}},
                ],
            },
            "provider_native_rejection_reason": "tool_call_count_2",
        }
        from unittest.mock import patch

        with patch(
            "generate_teacher_rollouts.request_chat",
            return_value=("", usage, "I will issue two calls."),
        ) as mocked:
            content, final_usage, reasoning = chat_with_retries(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                retries=3,
                deepseek_carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "I will issue two calls.")
        self.assertEqual(final_usage["api_request_attempts"], 1)
        self.assertEqual(final_usage["api_carrier_retries"], 0)
        self.assertEqual(
            final_usage["provider_native_rejection_reason"],
            "tool_call_count_2",
        )

    def test_native_carrier_feedback_is_not_misclassified_as_argument_validation(self):
        error = ProtocolError(
            "DeepSeek native function-call transport error: the assistant selected 2 native "
            "functions; put only parameters in the function arguments."
        )
        self.assertEqual(protocol_failure_type(error), "protocol_error")


if __name__ == "__main__":
    unittest.main()
