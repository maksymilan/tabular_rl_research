#!/usr/bin/env python3
from __future__ import annotations

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

from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    provider_request_audit_options,
    provider_request_options,
)
from generate_teacher_rollouts import is_semantic_failure_for_ordered_stop  # noqa: E402
from tool_modules.native_tool_bundle import no_plan_protocol, reviewed_protocol  # noqa: E402
from tool_modules.native_tool_bundle.no_plan_prompt import (  # noqa: E402
    NATIVE_STUDENT_SYSTEM_PROMPT,
    NATIVE_TEACHER_SYSTEM_PROMPT,
    PROMPT_PROFILE,
    static_request_size_audit,
)
from tool_modules.native_tool_bundle.provider_tools import (  # noqa: E402
    native_atomic_tools,
    native_tools_sha256,
)
from protocol import ProtocolError  # noqa: E402


class Version54NoPlanProtocolTests(unittest.TestCase):
    def test_ordered_stop_separates_policy_failures_from_provider_failures(self):
        self.assertTrue(
            is_semantic_failure_for_ordered_stop(
                {"correct": False, "failure_type": "wrong_answer"}
            )
        )
        self.assertTrue(
            is_semantic_failure_for_ordered_stop(
                {"correct": False, "failure_type": "max_steps"}
            )
        )
        for failure_type in (
            "api_error",
            "context_overflow",
            "provider_carrier_error",
        ):
            self.assertFalse(
                is_semantic_failure_for_ordered_stop(
                    {"correct": False, "failure_type": failure_type}
                )
            )
        self.assertFalse(
            is_semantic_failure_for_ordered_stop(
                {"correct": True, "failure_type": None}
            )
        )

    def test_only_plan_is_removed_from_version53_public_surface(self):
        expected = set(reviewed_protocol.MODEL_ARG_SCHEMA) - {"plan"}
        self.assertEqual(no_plan_protocol.PROTOCOL_VERSION, "version54")
        self.assertEqual(no_plan_protocol.REMOVED_MODEL_TOOLS, frozenset({"plan"}))
        self.assertEqual(set(no_plan_protocol.MODEL_ARG_SCHEMA), expected)
        self.assertEqual(no_plan_protocol.TOOLS, expected)
        self.assertEqual(len(expected), 11)
        for tool in expected:
            self.assertEqual(
                no_plan_protocol.MODEL_ARG_SCHEMA[tool],
                reviewed_protocol.MODEL_ARG_SCHEMA[tool],
            )

    def test_student_runtime_is_identical_and_teacher_drops_only_stale_plan_rule(self):
        self.assertEqual(
            NATIVE_STUDENT_SYSTEM_PROMPT,
            reviewed_protocol.STUDENT_SYSTEM_PROMPT,
        )
        self.assertTrue(NATIVE_TEACHER_SYSTEM_PROMPT.startswith(NATIVE_STUDENT_SYSTEM_PROMPT))
        self.assertNotIn("Plan evidence must be grounded", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertIn("TEACHER-ONLY TRAJECTORY GENERATION", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertIn("candidate join paths, not proof of key uniqueness", NATIVE_TEACHER_SYSTEM_PROMPT)
        self.assertEqual(PROMPT_PROFILE, "native-schema-role-separated-no-plan-v1")

    def test_official_provider_request_exposes_exactly_eleven_non_plan_functions(self):
        schema = no_plan_protocol.MODEL_ARG_SCHEMA
        options = provider_request_options(
            "deepseek-v4-flash",
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
            native_model_arg_schema=schema,
        )
        names = [item["function"]["name"] for item in options["tools"]]
        self.assertEqual(set(names), set(schema))
        self.assertNotIn("plan", names)
        audit = provider_request_audit_options(
            "deepseek-v4-flash",
            carrier=DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
            native_model_arg_schema=schema,
        )
        self.assertEqual(audit["native_tool_count"], 11)
        self.assertEqual(audit["native_tools_sha256"], native_tools_sha256(schema))

    def test_removed_plan_is_rejected_before_execution_validation(self):
        with self.assertRaises(ProtocolError) as caught:
            no_plan_protocol.validate_model_action(
                "plan",
                {"ops": [{"op": "create", "id": "p1", "goal": "inspect"}]},
            )
        self.assertEqual(caught.exception.code, "unknown_tool")
        self.assertNotIn("plan", caught.exception.details["legal_tools"])

    def test_schema_and_static_request_hashes_bind_the_ablation(self):
        self.assertNotEqual(
            no_plan_protocol.tool_schema_hash(),
            reviewed_protocol.tool_schema_hash(),
        )
        self.assertNotEqual(
            no_plan_protocol.protocol_hash(NATIVE_TEACHER_SYSTEM_PROMPT),
            reviewed_protocol.protocol_hash(reviewed_protocol.TEACHER_SYSTEM_PROMPT),
        )
        audit = static_request_size_audit(
            native_atomic_tools(no_plan_protocol.MODEL_ARG_SCHEMA)
        )
        self.assertEqual(audit["profile"], PROMPT_PROFILE)
        self.assertLess(audit["static_request_chars"], 15_500)


if __name__ == "__main__":
    unittest.main()
