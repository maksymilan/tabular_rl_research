#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "sft"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
]

from generate_teacher_rollouts import DATA_GENERATION_SUFFIX  # noqa: E402
from protocol import get_system_prompt, rolling_system_prompt, teacher_system_prompt  # noqa: E402
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    provider_system_prompt,
)
from tool_modules.native_tool_bundle import compact_protocol  # noqa: E402
from tool_modules.native_tool_bundle.compact_prompt import (  # noqa: E402
    ERROR_RECOVERY_CONTRACT,
    NATIVE_STUDENT_SYSTEM_PROMPT,
    NATIVE_TEACHER_SYSTEM_PROMPT,
    SOFT_BUNDLE_POLICY,
    static_request_size_audit,
)
from tool_modules.native_tool_bundle.provider_tools import native_atomic_tools  # noqa: E402


class CompactNativeBundlePromptTests(unittest.TestCase):
    def test_version52_changes_only_prompt_profile(self):
        self.assertEqual(compact_protocol.PROTOCOL_VERSION, "version52")
        self.assertEqual(compact_protocol.STUDENT_SYSTEM_PROMPT, NATIVE_STUDENT_SYSTEM_PROMPT)
        self.assertEqual(compact_protocol.TEACHER_SYSTEM_PROMPT, NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertEqual(len(compact_protocol.protocol_hash(NATIVE_TEACHER_SYSTEM_PROMPT)), 16)

    def test_soft_bundle_rules_and_hard_terminal_rule_are_stated_once(self):
        self.assertIn("Default to one function call", SOFT_BUNDLE_POLICY)
        self.assertIn("no more than three calls", SOFT_BUNDLE_POLICY)
        self.assertIn("independent perception work", SOFT_BUNDLE_POLICY)
        self.assertIn("explicit competing hypotheses", SOFT_BUNDLE_POLICY)
        self.assertIn("Normally issue join_tables", SOFT_BUNDLE_POLICY)
        self.assertEqual(NATIVE_TEACHER_SYSTEM_PROMPT.count("answer_from_context must be"), 1)

    def test_error_feedback_is_preserved_as_causal_history(self):
        self.assertIn("remain in causal history", ERROR_RECOVERY_CONTRACT)
        self.assertIn("correct the next call", ERROR_RECOVERY_CONTRACT)
        self.assertIn("not data evidence", ERROR_RECOVERY_CONTRACT)

    def test_schema_replaces_textual_tool_catalog_and_cookbook(self):
        self.assertNotIn("CANONICAL CALLS", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("NATIVE FUNCTION CALL EXAMPLES", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("condition_filter(table, conditions", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertIn("API-supplied native function schemas", NATIVE_TEACHER_SYSTEM_PROMPT)

    def test_static_request_is_less_than_sixty_percent_of_version51(self):
        student = rolling_system_prompt(get_system_prompt(), compact=False)
        canonical = teacher_system_prompt(student) + DATA_GENERATION_SUFFIX
        version51_prompt = provider_system_prompt(
            "deepseek-v4-flash",
            canonical,
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
        )
        tools = native_atomic_tools()
        old_chars = len(version51_prompt) + len(
            json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        )
        audit = static_request_size_audit(tools)
        self.assertLess(audit["static_request_chars"], old_chars * 0.60)


if __name__ == "__main__":
    unittest.main()
