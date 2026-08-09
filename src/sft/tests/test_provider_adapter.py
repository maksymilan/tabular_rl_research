#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
    DEEPSEEK_CARRIER_TOOL_CALL,
    adapt_provider_response,
    provider_default_max_tokens,
    provider_instruction,
    provider_request_messages,
    provider_request_audit_options,
    provider_request_options,
    provider_rejection_message,
    provider_system_prompt,
)
from generate_teacher_rollouts import (  # noqa: E402
    DATA_GENERATION_SUFFIX,
    ProviderCarrierError,
    chat_with_retries,
    protocol_failure_type,
    request_chat,
)
from rollout import ChatAPIError  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    ROLLING_SYSTEM_PROMPT_COMPACT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    TEACHER_SYSTEM_PROMPT,
    parse_assistant_strict,
)


class ProviderAdapterTests(unittest.TestCase):
    def test_local_qwen_request_can_pin_thinking_chat_template(self):
        response_payload = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "<think>reason</think>\n{}"},
            }],
            "usage": {},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with patch(
            "generate_teacher_rollouts.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as mocked:
            request_chat(
                base_url="http://127.0.0.1:8031/v1",
                api_key="local-vllm",
                model="qwen3-8b-iterative-sql-base",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                chat_template_kwargs={"enable_thinking": True},
            )

        sent = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(sent["temperature"], 0)

    def test_nonretryable_payment_error_is_not_repeated(self):
        with patch(
            "generate_teacher_rollouts.request_chat",
            side_effect=ChatAPIError("HTTP 402", status=402),
        ) as mocked, patch("generate_teacher_rollouts.time.sleep") as sleeping:
            with self.assertRaises(ChatAPIError) as captured:
                chat_with_retries(
                    base_url="https://api.deepseek.com",
                    api_key="test-key",
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=2048,
                    timeout=30,
                    retries=10,
                )
        self.assertEqual(captured.exception.status, 402)
        self.assertEqual(mocked.call_count, 1)
        sleeping.assert_not_called()

    def test_retryable_rate_limit_still_uses_transport_retry_budget(self):
        success = (
            '{"tool":"describe_table","arguments":{"tables":["T"]}}',
            {
                "api_finish_reason": "stop",
                "provider_request_options": {},
                "provider_response_metadata": {},
            },
            "Inspect schema.",
        )
        with patch(
            "generate_teacher_rollouts.request_chat",
            side_effect=[ChatAPIError("HTTP 429", status=429), success],
        ) as mocked, patch("generate_teacher_rollouts.time.sleep"):
            _, usage, _ = chat_with_retries(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                retries=3,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(usage["api_transport_retries"], 1)

    def test_deepseek_request_options_explicitly_enable_thinking(self):
        self.assertEqual(
            provider_request_options("deepseek-v4-flash"),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "response_format": {"type": "json_object"},
            },
        )
        native = provider_request_options(
            "deepseek-v4-flash",
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        )
        self.assertEqual(native["thinking"], {"type": "enabled"})
        self.assertEqual(native["reasoning_effort"], "high")
        self.assertEqual(native["tool_choice"], "auto")
        self.assertNotIn("response_format", native)
        self.assertEqual(len(native["tools"]), 12)
        audit = provider_request_audit_options(
            "deepseek-v4-flash",
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        )
        self.assertNotIn("tools", audit)
        self.assertEqual(audit["native_tool_count"], 12)
        self.assertEqual(len(audit["native_tools_sha256"]), 64)
        self.assertEqual(provider_request_options("gpt-5.6-sol"), {})
        self.assertEqual(
            provider_request_options(
                "deepseek-v4-flash",
                carrier=DEEPSEEK_CARRIER_TOOL_CALL,
            ),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        )

    def test_teacher_request_sends_thinking_and_audits_provider_identity(self):
        response_payload = {
            "id": "response_123",
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp_test",
            "object": "chat.completion",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": (
                        '{"tool":"describe_table","arguments":'
                        '{"tables":["Document"]}}'
                    ),
                    "reasoning_content": "Inspect the schema.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with patch(
            "generate_teacher_rollouts.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as mocked:
            content, usage, reasoning = request_chat(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
            )

        request = mocked.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(sent["thinking"], {"type": "enabled"})
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", sent)
        self.assertEqual(reasoning, "Inspect the schema.")
        self.assertTrue(content.startswith('{"tool"'))
        self.assertEqual(
            usage["provider_response_metadata"],
            {
                "id": "response_123",
                "model": "deepseek-v4-flash",
                "system_fingerprint": "fp_test",
                "object": "chat.completion",
            },
        )

    def test_teacher_request_extracts_one_native_function_call_without_editing_arguments(self):
        raw_arguments = '{"tables":["Document"]}'
        response_payload = {
            "id": "response_native_1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "Inspect the schema.",
                    "tool_calls": [{
                        "id": "call_native_1",
                        "type": "function",
                        "function": {
                            "name": "describe_table",
                            "arguments": raw_arguments,
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with patch(
            "generate_teacher_rollouts.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as mocked:
            content, usage, reasoning = request_chat(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                deepseek_carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
            )

        sent = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(sent["tool_choice"], "auto")
        self.assertEqual(len(sent["tools"]), 12)
        self.assertNotIn("response_format", sent)
        self.assertEqual(
            json.loads(content),
            {"tool": "describe_table", "arguments": {"tables": ["Document"]}},
        )
        self.assertEqual(reasoning, "Inspect the schema.")
        native_message = usage["provider_native_assistant_message"]
        self.assertEqual(native_message["tool_calls"][0]["id"], "call_native_1")
        self.assertEqual(
            native_message["tool_calls"][0]["function"]["arguments"],
            raw_arguments,
        )
        self.assertIsNone(usage["provider_native_rejection_reason"])

    def test_teacher_request_accepts_bounded_native_bundle_and_ignores_content(self):
        response_payload = {
            "id": "response_bundle_1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "I will inspect both columns.",
                    "reasoning_content": "Both inspections are independent.",
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
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with patch(
            "generate_teacher_rollouts.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            content, usage, reasoning = request_chat(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                deepseek_carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
            )

        calls = json.loads(content)["calls"]
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual(
            [call["tool"] for call in calls],
            ["inspect_column", "inspect_column"],
        )
        self.assertEqual(reasoning, "Both inspections are independent.")
        self.assertIsNone(usage["provider_native_rejection_reason"])
        self.assertEqual(
            usage["provider_native_assistant_message"]["content"],
            "I will inspect both columns.",
        )

    def test_length_truncation_retries_same_turn_with_larger_budget(self):
        truncated_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2048,
            "completion_tokens_details": {"reasoning_tokens": 2048},
            "api_finish_reason": "length",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "truncated"},
        }
        complete_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 30},
            "api_finish_reason": "stop",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "complete"},
        }
        final_action = '{"tool":"describe_table","arguments":{"tables":["Document"]}}'

        with patch(
            "generate_teacher_rollouts.request_chat",
            side_effect=[
                ("", truncated_usage, "Long truncated reasoning."),
                (final_action, complete_usage, "Inspect the schema."),
            ],
        ) as mocked:
            content, usage, reasoning = chat_with_retries(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                retries=3,
            )

        self.assertEqual(
            [call.kwargs["max_tokens"] for call in mocked.call_args_list],
            [2048, 4096],
        )
        self.assertEqual(content, final_action)
        self.assertEqual(reasoning, "Inspect the schema.")
        self.assertEqual(usage["prompt_tokens"], 20)
        self.assertEqual(usage["completion_tokens"], 2098)
        self.assertEqual(usage["completion_tokens_details.reasoning_tokens"], 2078)
        self.assertEqual(usage["api_request_attempts"], 2)
        self.assertEqual(usage["api_completion_retries"], 1)
        self.assertEqual(usage["api_transport_retries"], 0)
        self.assertEqual(usage["api_finish_reason"], "stop")
        self.assertEqual(usage["provider_response_metadata"], {"id": "complete"})
        self.assertEqual(
            usage["api_retry_events"],
            [{
                "type": "completion_length",
                "request_attempt": 1,
                "max_tokens": 2048,
                "finish_reason": "length",
                "visible_content_present": False,
                "reasoning_content_present": True,
                "provider_response_metadata": {"id": "truncated"},
            }],
        )

    def test_empty_visible_carrier_retries_without_spending_a_semantic_action(self):
        empty_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 40,
            "api_finish_reason": "stop",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "empty"},
        }
        complete_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "api_finish_reason": "stop",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "complete"},
        }
        final_action = '{"tool":"describe_table","arguments":{"tables":["Document"]}}'

        with patch(
            "generate_teacher_rollouts.request_chat",
            side_effect=[
                ("", empty_usage, "Reasoning without a visible action."),
                (final_action, complete_usage, "Inspect the schema."),
            ],
        ) as mocked, patch("generate_teacher_rollouts.time.sleep"):
            content, usage, reasoning = chat_with_retries(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
                timeout=30,
                retries=3,
            )

        self.assertEqual(2, mocked.call_count)
        self.assertEqual(content, final_action)
        self.assertEqual(reasoning, "Inspect the schema.")
        self.assertEqual(usage["api_request_attempts"], 2)
        self.assertEqual(usage["api_carrier_retries"], 1)
        self.assertEqual(usage["api_transport_retries"], 0)
        self.assertEqual(
            usage["api_retry_events"],
            [{
                "type": "provider_carrier_empty",
                "request_attempt": 1,
                "max_tokens": 2048,
                "finish_reason": "stop",
                "visible_content_present": False,
                "reasoning_content_present": True,
                "provider_response_metadata": {"id": "empty"},
            }],
        )

    def test_repeated_empty_visible_carrier_fails_as_provider_event(self):
        empty_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 40,
            "api_finish_reason": "stop",
            "provider_request_options": {"thinking": {"type": "enabled"}},
            "provider_response_metadata": {"id": "empty"},
        }
        with patch(
            "generate_teacher_rollouts.request_chat",
            return_value=("", empty_usage, "Reasoning without a visible action."),
        ), patch("generate_teacher_rollouts.time.sleep"):
            with self.assertRaises(ProviderCarrierError) as captured:
                chat_with_retries(
                    base_url="https://api.deepseek.com",
                    api_key="test-key",
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=2048,
                    timeout=30,
                    retries=3,
                )

        self.assertEqual(3, captured.exception.usage["api_request_attempts"])
        self.assertEqual(2, captured.exception.usage["api_carrier_retries"])
        self.assertEqual(30, captured.exception.usage["prompt_tokens"])
        self.assertEqual(3, len(captured.exception.usage["api_retry_events"]))

    def test_teacher_generator_classifies_split_transport_as_protocol_error(self):
        error = ProtocolError(
            'DeepSeek split-response transport error: visible content must be exactly '
            '{"tool":"...","arguments":{...}}'
        )
        self.assertEqual(protocol_failure_type(error), "protocol_error")

    def test_deepseek_split_response_becomes_canonical_strict_action(self):
        content = '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        adapted, audit = adapt_provider_response(
            "deepseek-v4-flash", content, "Inspect the table schema first.",
        )
        self.assertTrue(audit["applied"])
        think, tool, args = parse_assistant_strict(adapted)
        self.assertEqual(think, "Inspect the table schema first.")
        self.assertEqual(tool, "describe_table")
        self.assertEqual(args["tables"], ["Document"])

    def test_deepseek_tool_call_carrier_becomes_same_canonical_action(self):
        content = (
            '<tool_call>{"tool":"describe_table",'
            '"arguments":{"tables":["Document"]}}</tool_call>'
        )
        adapted, audit = adapt_provider_response(
            "deepseek-v4-flash",
            content,
            "Inspect the table schema first.",
            carrier=DEEPSEEK_CARRIER_TOOL_CALL,
        )
        self.assertTrue(audit["applied"])
        think, tool, args = parse_assistant_strict(adapted)
        self.assertEqual(think, "Inspect the table schema first.")
        self.assertEqual(tool, "describe_table")
        self.assertEqual(args["tables"], ["Document"])

    def test_deepseek_native_carrier_becomes_same_canonical_action(self):
        content = '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        adapted, audit = adapt_provider_response(
            "deepseek-v4-flash",
            content,
            "Inspect the table schema first.",
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        )
        self.assertTrue(audit["applied"])
        self.assertTrue(audit["transport_reconstructed_from_native_tool_calls"])
        think, tool, args = parse_assistant_strict(adapted)
        self.assertEqual(think, "Inspect the table schema first.")
        self.assertEqual(tool, "describe_table")
        self.assertEqual(args["tables"], ["Document"])

    def test_adapter_refuses_partial_think_tag_or_missing_reasoning(self):
        malformed = 'reasoning without an opening tag.</think><tool_call>{"tool":"describe_table","arguments":{"tables":["Document"]}}</tool_call>'
        adapted, audit = adapt_provider_response("deepseek-v4-flash", malformed, "")
        self.assertFalse(audit["applied"])
        self.assertEqual(adapted, malformed)
        with self.assertRaises(ProtocolError):
            parse_assistant_strict(adapted)

    def test_adapter_audits_the_precise_rejection_reason(self):
        content = '{"tool":"describe_table","arguments":{"tables":["Document"]}'
        _, audit = adapt_provider_response("deepseek-v4-flash", content, "Inspect the schema.")
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["rejection_reason"], "visible_content_not_json")

    def test_adapter_rejects_valid_json_with_extra_top_level_keys(self):
        content = '{"tool":"describe_table","arguments":{"tables":["D"]},"reason":"extra"}'
        _, audit = adapt_provider_response("deepseek-v4-flash", content, "Inspect the schema.")
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["rejection_reason"], "visible_json_wrong_shape")

    def test_rejection_feedback_matches_deepseek_split_transport(self):
        _, audit = adapt_provider_response(
            "deepseek-v4-flash",
            'Reason first. {"tool":"describe_table","arguments":{"tables":["D"]}}',
            "Inspect schema.",
        )
        message = provider_rejection_message(audit)
        self.assertIn("valid JSON action object", message)
        self.assertIn("reasoning prose", message)
        self.assertIn("nothing before or after", message)

    def test_transport_instruction_is_model_specific(self):
        self.assertIn("DEEPSEEK SPLIT-RESPONSE", provider_instruction("deepseek-v4-flash"))
        self.assertIn("DEEPSEEK SPLIT-RESPONSE", provider_instruction("deepseek-v4-pro"))
        self.assertEqual(provider_instruction("gpt-5.6-sol"), "")

    def test_transport_instruction_accepts_interface_specific_example(self):
        sql_example = (
            '{"tool":"execute_sql","arguments":{"sql":"SELECT 1"}}'
        )
        instruction = provider_instruction(
            "deepseek-v4-flash", example_visible_content=sql_example,
        )
        self.assertIn(sql_example, instruction)
        self.assertNotIn('"tool":"describe_table"', instruction)

    def test_transport_instruction_can_hide_client_implementation_details(self):
        default = provider_instruction("deepseek-v4-flash")
        provider_facing = provider_instruction(
            "deepseek-v4-flash",
            include_client_implementation=False,
        )
        self.assertIn("The client preserves the separate reason", default)
        self.assertIn("active think-plus-JSON envelope", default)
        self.assertNotIn("The client preserves the separate reason", provider_facing)
        self.assertNotIn("active think-plus-JSON envelope", provider_facing)
        self.assertIn("separate native reasoning channel", provider_facing)

    def test_deepseek_api_prompt_has_one_nonconflicting_response_contract(self):
        prompt = provider_system_prompt(
            "deepseek-v4-flash",
            TEACHER_SYSTEM_PROMPT + DATA_GENERATION_SUFFIX,
        )
        self.assertNotIn("output exactly: <think>brief reasoning</think>", prompt)
        self.assertNotIn("Emit exactly one non-empty <think> block", prompt)
        self.assertNotIn("<tool_call> block in the provider's visible field", prompt)
        self.assertNotIn("exactly one <tool_call> block", prompt)
        self.assertEqual(prompt.count("DEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT"), 1)
        self.assertIn("separate native reasoning channel", prompt)
        self.assertIn("under 120 words", prompt)
        self.assertIn("reasoning is incomplete until the action is emitted", prompt)
        self.assertIn("emit the next legal inspection or data action", prompt)
        self.assertIn("Put every tool parameter inside arguments", prompt)
        self.assertIn("Copy this final-response shape", prompt)
        self.assertNotIn("reasoning_content field value:", prompt)
        self.assertNotIn("visible content field value:", prompt)
        self.assertIn(
            '{"tool":"describe_table","arguments":{"tables":["Document"]}}',
            prompt,
        )
        self.assertIn('"base":"orders"', prompt)
        self.assertIn('"left":"orders.customer_id"', prompt)
        self.assertNotIn('"prefixes"', prompt)

    def test_deepseek_tool_call_prompt_has_one_nonconflicting_contract(self):
        prompt = provider_system_prompt(
            "deepseek-v4-flash",
            TEACHER_SYSTEM_PROMPT + DATA_GENERATION_SUFFIX,
            carrier=DEEPSEEK_CARRIER_TOOL_CALL,
        )
        self.assertNotIn("output exactly: <think>brief reasoning</think>", prompt)
        self.assertNotIn("Emit exactly one non-empty <think> block", prompt)
        self.assertEqual(prompt.count("DEEPSEEK SPLIT-RESPONSE TOOL-CALL CONTRACT"), 1)
        self.assertNotIn("JSON OUTPUT CONTRACT", prompt)
        self.assertNotIn("reasoning_content field value:", prompt)
        self.assertNotIn("visible content field value:", prompt)
        self.assertIn("under 120 words", prompt)
        self.assertIn("reasoning is incomplete until the action is emitted", prompt)
        self.assertIn("Put every tool parameter inside arguments", prompt)
        self.assertIn("Its first character is < and its last character is >", prompt)
        self.assertIn(
            '<tool_call>{"tool":"describe_table",'
            '"arguments":{"tables":["Document"]}}</tool_call>',
            prompt,
        )

    def test_deepseek_native_prompt_uses_functions_and_rewrites_call_cookbook(self):
        prompt = provider_system_prompt(
            "deepseek-v4-flash",
            TEACHER_SYSTEM_PROMPT + DATA_GENERATION_SUFFIX,
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        )
        self.assertNotIn("output exactly: <think>brief reasoning</think>", prompt)
        self.assertNotIn("Emit exactly one non-empty <think> block", prompt)
        self.assertNotIn("CANONICAL CALLS", prompt)
        self.assertEqual(prompt.count("DEEPSEEK NATIVE FUNCTION-CALL CONTRACT"), 1)
        self.assertIn("NATIVE FUNCTION CALL EXAMPLES", prompt)
        self.assertIn(
            'Filter: call function condition_filter with arguments {"table":"people"',
            prompt,
        )
        self.assertIn("Leave assistant content empty", prompt)
        self.assertIn("select exactly one supplied function", prompt)

    def test_deepseek_empty_visible_feedback_requests_json_not_xml(self):
        _, audit = adapt_provider_response(
            "deepseek-v4-flash", "", "Inspect the schema.",
        )
        message = provider_rejection_message(audit)
        self.assertIn("complete JSON action object", message)
        self.assertNotIn("tool_call block", message)

    def test_non_split_api_prompt_remains_canonical(self):
        canonical = TEACHER_SYSTEM_PROMPT + DATA_GENERATION_SUFFIX
        self.assertEqual(provider_system_prompt("gpt-5.6-sol", canonical), canonical)

    def test_deepseek_prompt_supports_both_compact_semantic_variants(self):
        for semantic_prompt in (SYSTEM_PROMPT_COMPACT, ROLLING_SYSTEM_PROMPT_COMPACT):
            with self.subTest(prompt=semantic_prompt[:30]):
                prompt = provider_system_prompt(
                    "deepseek-v4-flash",
                    semantic_prompt + DATA_GENERATION_SUFFIX,
                )
                self.assertNotIn("<think>brief reason for this action</think>", prompt)
                self.assertNotIn(
                    "<think>specific reason for the next action</think>",
                    prompt,
                )
                self.assertEqual(
                    prompt.count("DEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT"),
                    1,
                )

    def test_deepseek_history_exposes_only_prior_tool_calls(self):
        canonical = (
            "<think>Inspect the schema before filtering.</think>\n"
            '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": canonical},
            {"role": "user", "content": "observation"},
        ]
        rendered = provider_request_messages("deepseek-v4-flash", messages)
        self.assertEqual(messages[1]["content"], canonical)
        self.assertEqual(
            rendered[1]["content"],
            '{"tool":"describe_table","arguments":{"tables":["Document"]}}',
        )
        self.assertNotIn("<think>", rendered[1]["content"])

        tool_call_rendered = provider_request_messages(
            "deepseek-v4-flash",
            messages,
            carrier=DEEPSEEK_CARRIER_TOOL_CALL,
        )
        self.assertEqual(
            tool_call_rendered[1]["content"],
            '<tool_call>{"tool":"describe_table",'
            '"arguments":{"tables":["Document"]}}</tool_call>',
        )
        self.assertNotIn("<think>", tool_call_rendered[1]["content"])

        native_history = [
            (
                canonical,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect the schema before filtering.",
                    "tool_calls": [{
                        "id": "call_native_history_1",
                        "type": "function",
                        "function": {
                            "name": "describe_table",
                            "arguments": '{"tables":["Document"]}',
                        },
                    }],
                },
            )
        ]
        native_rendered = provider_request_messages(
            "deepseek-v4-flash",
            messages,
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
            native_assistant_history=native_history,
        )
        self.assertEqual(
            [message["role"] for message in native_rendered],
            ["system", "assistant", "tool"],
        )
        self.assertEqual(
            native_rendered[1]["tool_calls"][0]["id"],
            native_rendered[2]["tool_call_id"],
        )
        self.assertEqual(
            native_rendered[1]["reasoning_content"],
            "Inspect the schema before filtering.",
        )

    def test_deepseek_history_rejects_ambiguous_assistant_text(self):
        with self.assertRaises(ValueError):
            provider_request_messages(
                "deepseek-v4-flash",
                [{"role": "assistant", "content": "reasoning before <tool_call>{}</tool_call>"}],
            )

    def test_deepseek_default_token_budget_is_provider_specific(self):
        self.assertEqual(provider_default_max_tokens("deepseek-v4-flash", 1024), 2048)
        self.assertEqual(provider_default_max_tokens("gpt-5.6-sol", 1024), 1024)


if __name__ == "__main__":
    unittest.main()
