#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from provider_adapter import (  # noqa: E402
    adapt_provider_response,
    provider_default_max_tokens,
    provider_instruction,
)
from protocol import ProtocolError, parse_assistant_strict  # noqa: E402


class ProviderAdapterTests(unittest.TestCase):
    def test_deepseek_split_response_becomes_canonical_strict_action(self):
        content = '<tool_call>{"tool":"describe_table","arguments":{"tables":["Document"]}}</tool_call>'
        adapted, audit = adapt_provider_response(
            "deepseek-v4-flash", content, "Inspect the table schema first.",
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
        content = '<tool_call>{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        _, audit = adapt_provider_response("deepseek-v4-flash", content, "Inspect the schema.")
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["rejection_reason"], "incomplete_or_suffixed_visible_tool_call")

    def test_transport_instruction_is_model_specific(self):
        self.assertIn("DEEPSEEK SPLIT-RESPONSE", provider_instruction("deepseek-v4-flash"))
        self.assertIn("DEEPSEEK SPLIT-RESPONSE", provider_instruction("deepseek-v4-pro"))
        self.assertEqual(provider_instruction("gpt-5.6-sol"), "")

    def test_deepseek_default_token_budget_is_provider_specific(self):
        self.assertEqual(provider_default_max_tokens("deepseek-v4-flash", 1024), 2048)
        self.assertEqual(provider_default_max_tokens("gpt-5.6-sol", 1024), 1024)


if __name__ == "__main__":
    unittest.main()
