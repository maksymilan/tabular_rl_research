#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SFT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = SFT_DIR.parent / "eval"
SRC_DIR = SFT_DIR.parent
sys.path[:0] = [str(SRC_DIR), str(SFT_DIR), str(EVAL_DIR)]

from run_tool_scheme import runner_argv  # noqa: E402
from generate_tool_scheme_rollouts import generator_argv  # noqa: E402
from tool_modules.action_block.protocol import (  # noqa: E402
    BATCH_CARRIER_PROVIDER_NATIVE,
    validate_atomic_call,
)
from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    DIRECT_SQL_SEARCH_TOOL_SCHEME,
    ITERATIVE_SQL_TOOL_SCHEME,
    NATIVE_TOOL_BUNDLE_SCHEME,
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    assert_record_tool_scheme,
    build_action_block_tool_scheme,
    build_atomic_tool_scheme,
    build_direct_sql_search_tool_scheme,
    build_iterative_sql_tool_scheme,
    build_native_tool_bundle_scheme,
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
        self.assertEqual(block.max_batch_calls, 8)
        self.assertNotIn("answer_from_context", block.atomic_tools)
        self.assertIn("join", block.atomic_tools)
        self.assertNotIn("join_tables", block.atomic_tools)
        with self.assertRaisesRegex(ValueError, "1..8"):
            build_action_block_tool_scheme(max_batch_calls=9)
        self.assertNotEqual(atomic.protocol_hash, block.protocol_hash)
        self.assertEqual(atomic.assistant_carrier, block.assistant_carrier)
        self.assertEqual(block.protocol_version, "action-block-v35")
        self.assertEqual(block.protocol_hash, "7d9d924141ea4beb")
        provider_block = build_action_block_tool_scheme(
            assistant_carrier=BATCH_CARRIER_PROVIDER_NATIVE,
        )
        self.assertEqual(provider_block.protocol_hash, "6d926b09e2150e3b")
        self.assertNotIn("partition_by", atomic.system_prompt)
        self.assertNotIn("partition_by", block.system_prompt)
        self.assertIn("read_subtable(table, limit?, columns?, conditions?, order_by?, offset?)",
                      atomic.system_prompt)
        self.assertNotIn("offset", block.system_prompt)

        native_bundle = build_native_tool_bundle_scheme()
        self.assertEqual(native_bundle.name, NATIVE_TOOL_BUNDLE_SCHEME)
        self.assertEqual(native_bundle.protocol_version, "version54")
        self.assertEqual(native_bundle.max_batch_calls, 8)
        self.assertIn("inspect_column", native_bundle.top_level_tools)
        self.assertNotIn("plan", native_bundle.top_level_tools)
        self.assertNotIn("answer_from_context", native_bundle.atomic_tools)
        self.assertNotEqual(native_bundle.protocol_hash, atomic.protocol_hash)

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

        direct_sql_search = build_direct_sql_search_tool_scheme()
        self.assertEqual(direct_sql_search.name, DIRECT_SQL_SEARCH_TOOL_SCHEME)
        self.assertEqual(
            direct_sql_search.top_level_tools,
            ("execute_sql", "search_values"),
        )
        self.assertNotIn("submit_sql", direct_sql_search.system_prompt)
        self.assertNotIn("condition_filter", direct_sql_search.system_prompt)
        self.assertEqual(direct_sql_search.protocol_version, "direct-sql-search-v2")
        self.assertIn("SEMANTIC DECISION DISCIPLINE", direct_sql_search.system_prompt)

        iterative_sql = build_iterative_sql_tool_scheme()
        self.assertEqual(iterative_sql.name, ITERATIVE_SQL_TOOL_SCHEME)
        self.assertEqual(
            iterative_sql.top_level_tools,
            ("execute_sql", "submit_sql"),
        )
        self.assertEqual(iterative_sql.protocol_version, "iterative-sql-v6")
        self.assertIn("Construct and execute the exact final candidate", iterative_sql.system_prompt)
        self.assertIn("Every answer is a SQL result table", iterative_sql.system_prompt)
        self.assertIn("the episode continues", iterative_sql.system_prompt)
        self.assertNotIn("search_values", iterative_sql.system_prompt)
        self.assertNotIn("condition_filter", iterative_sql.system_prompt)
        self.assertNotEqual(iterative_sql.protocol_hash, direct_sql_search.protocol_hash)

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

        native_bundle = build_native_tool_bundle_scheme()
        with self.assertRaisesRegex(ValueError, "structured provider messages"):
            render_scheme_action(
                native_bundle,
                "Inspect schema.",
                "describe_table",
                {"tables": ["items"]},
            )
        with self.assertRaisesRegex(ValueError, "structured provider tool_calls"):
            parse_scheme_action(native_bundle, atomic_text)

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
                    "table": {"source_table": "items"},
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

        direct_sql_search = build_direct_sql_search_tool_scheme()
        sql_arguments = {"sql": "SELECT name FROM items", "mode": "inspect"}
        sql_text = render_scheme_action(
            direct_sql_search,
            "Inspect the exact output.",
            "execute_sql",
            sql_arguments,
        )
        self.assertEqual(
            parse_scheme_action(direct_sql_search, sql_text),
            (
                "Inspect the exact output.",
                "execute_sql",
                sql_arguments,
            ),
        )

        iterative_sql = build_iterative_sql_tool_scheme()
        submit_arguments = {"sql": "SELECT name FROM items"}
        submit_text = render_scheme_action(
            iterative_sql,
            "Submit the inspected answer query.",
            "submit_sql",
            submit_arguments,
        )
        self.assertEqual(
            parse_scheme_action(iterative_sql, submit_text),
            (
                "Submit the inspected answer query.",
                "submit_sql",
                submit_arguments,
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
        native_bundle = runner_argv(
            NATIVE_TOOL_BUNDLE_SCHEME,
            ["--", "--limit", "1", "--out", "bundle.jsonl"],
        )
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
        self.assertTrue(native_bundle[1].endswith("/generate_teacher_rollouts.py"))
        self.assertEqual(
            native_bundle[-5:],
            [
                "--atomic-protocol-version",
                "version54",
                "--deepseek-carrier",
                "native-tool-bundle",
                "--diagnostic-only",
            ],
        )
        generated_bundle = generator_argv(
            NATIVE_TOOL_BUNDLE_SCHEME,
            ["--", "--limit", "1", "--out", "bundle.jsonl"],
        )
        self.assertEqual(generated_bundle, native_bundle)
        self.assertTrue(
            block[1].endswith("/tool_modules/action_block/evaluator.py")
        )
        self.assertNotIn("--safe-low-friction-interface", block)
        self.assertTrue(
            relational_program[1].endswith(
                "/tool_modules/relational_program/evaluator.py"
            )
        )
        direct_sql_search = runner_argv(
            DIRECT_SQL_SEARCH_TOOL_SCHEME,
            ["--", "--n", "1"],
        )
        self.assertTrue(
            direct_sql_search[1].endswith("/tool_modules/sql_common/runner.py")
        )
        self.assertEqual(
            direct_sql_search[-2:],
            ["--interface", "search-values-execute-sql-v2"],
        )
        iterative_sql = runner_argv(
            ITERATIVE_SQL_TOOL_SCHEME,
            ["--", "--n", "1"],
        )
        self.assertTrue(
            iterative_sql[1].endswith("/tool_modules/sql_common/runner.py")
        )
        self.assertEqual(
            iterative_sql[-2:],
            ["--interface", "execute-sql-submit-sql-v6"],
        )
        generated_iterative_sql = generator_argv(
            ITERATIVE_SQL_TOOL_SCHEME,
            ["--", "--n", "1"],
        )
        self.assertEqual(generated_iterative_sql, iterative_sql)


if __name__ == "__main__":
    unittest.main()
