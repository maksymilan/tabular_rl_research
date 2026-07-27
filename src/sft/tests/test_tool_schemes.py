#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SFT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = SFT_DIR.parent / "eval"
sys.path[:0] = [str(SFT_DIR), str(EVAL_DIR)]

from run_tool_scheme import runner_argv  # noqa: E402
from batch_plan_protocol import (  # noqa: E402
    BATCH_CARRIER_PROVIDER_NATIVE,
    validate_atomic_call,
)
from tool_schemes import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    assert_record_tool_scheme,
    build_action_block_tool_scheme,
    build_atomic_tool_scheme,
    build_relational_program_tool_scheme,
    parse_scheme_action,
    render_scheme_action,
)


class ToolSchemeRegistryTests(unittest.TestCase):
    def test_schemes_keep_distinct_work_actions_and_share_terminal_semantics(self):
        atomic = build_atomic_tool_scheme()
        block = build_action_block_tool_scheme()
        self.assertEqual(atomic.name, ATOMIC_TOOL_SCHEME)
        self.assertEqual(block.name, ACTION_BLOCK_TOOL_SCHEME)
        self.assertNotIn("action_block", atomic.top_level_tools)
        self.assertEqual(
            block.top_level_tools,
            ("action_block", "answer_from_context"),
        )
        self.assertEqual(block.max_batch_calls, 5)
        self.assertNotIn("answer_from_context", block.atomic_tools)
        with self.assertRaisesRegex(ValueError, "1..5"):
            build_action_block_tool_scheme(max_batch_calls=6)
        self.assertNotEqual(atomic.protocol_hash, block.protocol_hash)
        self.assertEqual(atomic.assistant_carrier, block.assistant_carrier)
        self.assertEqual(block.protocol_hash, "34d5122197bda7ea")
        provider_block = build_action_block_tool_scheme(
            assistant_carrier=BATCH_CARRIER_PROVIDER_NATIVE,
        )
        self.assertEqual(provider_block.protocol_hash, "40bb637422e7ca6b")
        self.assertIn("partition_by", atomic.system_prompt)
        self.assertNotIn("partition_by", block.system_prompt)
        self.assertIn("offset", atomic.system_prompt)
        self.assertNotIn("offset", block.system_prompt)

        relational_program = build_relational_program_tool_scheme()
        self.assertEqual(
            relational_program.name,
            RELATIONAL_PROGRAM_TOOL_SCHEME,
        )
        self.assertIn(
            "relational_program",
            relational_program.top_level_tools,
        )
        self.assertIn(
            "observe",
            relational_program.top_level_tools,
        )
        self.assertNotIn(
            "describe_table",
            relational_program.top_level_tools,
        )
        self.assertEqual(relational_program.max_batch_calls, 8)
        self.assertNotEqual(
            relational_program.protocol_hash,
            block.protocol_hash,
        )

    def test_each_scheme_round_trips_its_native_student_action(self):
        atomic = build_atomic_tool_scheme()
        atomic_text = render_scheme_action(
            atomic,
            "Inspect the schema.",
            "describe_table",
            {"tables": ["items"]},
        )
        self.assertEqual(
            parse_scheme_action(atomic, atomic_text),
            ("Inspect the schema.", "describe_table", {"tables": ["items"]}),
        )
        self.assertNotIn("<tool_call>", atomic_text)

        block = build_action_block_tool_scheme()
        block_args = {
            "calls": [{
                "id": "schema",
                "tool": "describe_table",
                "arguments": {"tables": ["items"]},
            }],
        }
        block_text = render_scheme_action(
            block,
            "Inspect the schema.",
            "action_block",
            block_args,
        )
        self.assertEqual(
            parse_scheme_action(block, block_text),
            ("Inspect the schema.", "action_block", block_args),
        )
        with self.assertRaises(Exception):
            parse_scheme_action(atomic, block_text)
        with self.assertRaises(Exception):
            parse_scheme_action(block, atomic_text)

        with self.assertRaises(Exception):
            validate_atomic_call(
                "read_subtable",
                {"table": "items", "offset": 20},
            )

        relational_program = build_relational_program_tool_scheme()
        program_args = {
            "calls": [{
                "id": "exact",
                "operation": "select",
                "arguments": {
                    "table": "items",
                    "expressions": ["category"],
                },
            }],
            "result": "exact",
        }
        program_text = render_scheme_action(
            relational_program,
            "Derive the exact result.",
            "relational_program",
            program_args,
        )
        self.assertEqual(
            parse_scheme_action(relational_program, program_text),
            (
                "Derive the exact result.",
                "relational_program",
                program_args,
            ),
        )

    def test_record_scheme_guard_prevents_dataset_mixing(self):
        assert_record_tool_scheme(
            {"tool_scheme": ATOMIC_TOOL_SCHEME},
            ATOMIC_TOOL_SCHEME,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            assert_record_tool_scheme(
                {"tool_scheme": ACTION_BLOCK_TOOL_SCHEME},
                ATOMIC_TOOL_SCHEME,
            )

    def test_unified_launcher_keeps_scheme_specific_runners(self):
        atomic = runner_argv(ATOMIC_TOOL_SCHEME, ["--", "--n", "1"])
        block = runner_argv(
            ACTION_BLOCK_TOOL_SCHEME,
            ["--", "--limit", "1"],
        )
        relational_program = runner_argv(
            RELATIONAL_PROGRAM_TOOL_SCHEME,
            ["--", "--limit", "1"],
        )
        self.assertTrue(atomic[1].endswith("/rollout.py"))
        self.assertNotIn("--safe-low-friction-interface", atomic)
        self.assertTrue(block[1].endswith("/evaluate_batch_plan.py"))
        self.assertNotIn("--safe-low-friction-interface", block)
        self.assertTrue(
            relational_program[1].endswith(
                "/evaluate_relational_program.py"
            )
        )


if __name__ == "__main__":
    unittest.main()
