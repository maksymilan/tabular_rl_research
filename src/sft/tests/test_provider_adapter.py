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
    DEEPSEEK_CARRIER_TOOL_CALL,
    adapt_provider_response,
    provider_default_max_tokens,
    provider_instruction,
    provider_request_messages,
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
from protocol import (  # noqa: E402
    ProtocolError,
    ROLLING_SYSTEM_PROMPT_COMPACT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    parse_assistant_strict,
)


class ProviderAdapterTests(unittest.TestCase):
    def test_deepseek_request_options_explicitly_enable_thinking(self):
        self.assertEqual(
            provider_request_options("deepseek-v4-flash"),
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "response_format": {"type": "json_object"},
            },
        )
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

    def test_deepseek_api_prompt_has_one_nonconflicting_response_contract(self):
        prompt = provider_system_prompt(
            "deepseek-v4-flash",
            SYSTEM_PROMPT + DATA_GENERATION_SUFFIX,
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
            SYSTEM_PROMPT + DATA_GENERATION_SUFFIX,
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

    def test_deepseek_empty_visible_feedback_requests_json_not_xml(self):
        _, audit = adapt_provider_response(
            "deepseek-v4-flash", "", "Inspect the schema.",
        )
        message = provider_rejection_message(audit)
        self.assertIn("complete JSON action object", message)
        self.assertNotIn("tool_call block", message)

    def test_non_split_api_prompt_remains_canonical(self):
        canonical = SYSTEM_PROMPT + DATA_GENERATION_SUFFIX
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
            '<tool_call>{"tool":"describe_table","arguments":{"tables":["Document"]}}</tool_call>'
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
