#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from protocol import (  # noqa: E402
    FORMAL_STUDENT_SYSTEM_PROMPT,
    MODEL_ARG_SCHEMA,
    POLICY_PROMPT_CANONICAL,
    POLICY_PROMPT_RELATIONAL_INVARIANTS,
    PROTOCOL_VERSION,
    RELATIONAL_INVARIANTS_SUFFIX,
    SYSTEM_PROMPT,
    STUDENT_PROMPT_FORMAL,
    TEACHER_SYSTEM_PROMPT,
    TOOL_SPECS,
    ProtocolError,
    parse_assistant,
    parse_assistant_strict,
    parse_legacy_assistant_strict,
    policy_system_prompt,
    student_runtime_system_prompt,
    tool_schema_hash,
    tool_output_message,
)


class ProtocolParseTests(unittest.TestCase):
    def test_version26_separates_student_contract_from_teacher_guidance(self):
        self.assertEqual(PROTOCOL_VERSION, "version26")
        self.assertEqual(set(TOOL_SPECS), set(MODEL_ARG_SCHEMA))
        self.assertNotIn("<tool_call>", SYSTEM_PROMPT)
        self.assertLess(len(SYSTEM_PROMPT), 6000)
        self.assertGreater(len(TEACHER_SYSTEM_PROMPT), len(SYSTEM_PROMPT))
        self.assertNotIn("CANONICAL CALLS", SYSTEM_PROMPT)
        self.assertIn("CANONICAL CALLS", TEACHER_SYSTEM_PROMPT)
        self.assertIn('"base":"orders"', TEACHER_SYSTEM_PROMPT)
        self.assertEqual(len(tool_schema_hash()), 64)

    def test_formal_student_variant_adds_shared_action_grammar(self):
        formal = student_runtime_system_prompt(
            student_prompt_variant=STUDENT_PROMPT_FORMAL,
        )
        self.assertIn(FORMAL_STUDENT_SYSTEM_PROMPT, formal)
        self.assertIn("HIGH-ENTROPY ACTION GRAMMAR", formal)
        self.assertNotIn("CANONICAL CALLS", formal)

    def test_tool_output_compacts_contiguous_logical_namespaces(self):
        message = tool_output_message(
            "step_2",
            {
                "table": "join_001",
                "kind": "join",
                "columns": ["orders.id", "orders.customer_id", "customers.id", "customers.name"],
                "row_count": 4,
            },
        )
        self.assertNotIn('"columns":["orders.id"', message)
        self.assertIn(
            '"column_namespaces":{"orders":["id","customer_id"],'
            '"customers":["id","name"]}',
            message,
        )

    def test_relational_policy_prompt_is_an_explicit_noncanonical_ablation(self):
        self.assertEqual(
            policy_system_prompt(SYSTEM_PROMPT, POLICY_PROMPT_CANONICAL),
            SYSTEM_PROMPT,
        )
        guided = policy_system_prompt(
            SYSTEM_PROMPT,
            POLICY_PROMPT_RELATIONAL_INVARIANTS,
        )
        self.assertEqual(guided, SYSTEM_PROMPT + RELATIONAL_INVARIANTS_SUFFIX)
        self.assertIn("form the required join before ranking or aggregation", guided)
        self.assertIn("Use inner join unless", guided)
        self.assertIn("what one row represents", guided)
        with self.assertRaisesRegex(ValueError, "unknown policy prompt variant"):
            policy_system_prompt(SYSTEM_PROMPT, "gold-recipes")

    def test_tool_output_keeps_mixed_projection_order_flat(self):
        message = tool_output_message(
            "step_3",
            {
                "table": "project_002",
                "kind": "project",
                "columns": ["orders.id", "count"],
                "row_count": 1,
            },
        )
        self.assertIn('"columns":["orders.id","count"]', message)
        self.assertNotIn("column_namespaces", message)

    def test_parse_standard_tool_call(self):
        think, tool, args = parse_assistant(
            '<think>Count rows.</think>\n'
            '{"tool":"group_aggregate","arguments":{"table":"items","group_by":[],"aggregations":[{"op":"count","column":"*","as":"count_1"}]}}'
        )

        self.assertEqual(think, "Count rows.")
        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["aggregations"][0]["op"], "count")

    def test_active_parser_rejects_tagged_carrier_but_migration_parser_accepts_it(self):
        tagged = (
            "<think>Inspect the schema.</think>\n"
            '<tool_call>{"tool":"describe_table",'
            '"arguments":{"tables":["items"]}}</tool_call>'
        )
        with self.assertRaisesRegex(ProtocolError, "not part of the active carrier"):
            parse_assistant_strict(tagged)
        self.assertEqual(
            parse_legacy_assistant_strict(tagged),
            (
                "Inspect the schema.",
                "describe_table",
                {"tables": ["items"]},
            ),
        )

    def test_parse_answer_shorthand_missing_tool_key(self):
        _, tool, args = parse_assistant(
            '{"answer_from_context","arguments":{"answer":[[1]],'
            '"evidence":{"table":"project_001"},"reason":"done"}}'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["answer"], [[1]])

    def test_parse_answer_key_shorthand(self):
        _, tool, args = parse_assistant(
            '{"answer_from_context":{"answer":[[1]],'
            '"evidence":{"table":"project_001"},"reason":"done"}}'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"]["table"], "project_001")

    def test_parse_balanced_json_without_closing_tag(self):
        _, tool, args = parse_assistant(
            '{"tool":"group_aggregate","arguments":{"table":"items","group_by":[],"aggregations":[{"op":"count","column":"*","as":"count_1"}]}}'
        )

        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["table"], "items")

    def test_legacy_aggregate_is_accepted_for_compatibility(self):
        _, tool, args = parse_assistant(
            '{"tool":"aggregate","arguments":{"table":"items","column":"*","op":"count"}}'
        )

        self.assertEqual(tool, "aggregate")
        self.assertEqual(args["op"], "count")

    def test_answer_allows_evidence_string_and_missing_answer(self):
        _, tool, args = parse_assistant(
            '{"tool":"answer_from_context","arguments":{"evidence":"project_001"}}'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"], {"table": "project_001"})
        self.assertEqual(args["answer"], [])

    def test_repairs_truncated_long_answer_with_cited_evidence(self):
        _, tool, args = parse_assistant(
            "<think>The evidence table project_002 contains the rows.</think>\n"
            '{"tool":"answer_from_context","arguments":{"answer":[[1],[2],[3]'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["answer"], [])
        self.assertEqual(args["evidence"]["table"], "project_002")

    def test_truncated_json_stays_protocol_error(self):
        with self.assertRaises(ProtocolError):
            parse_assistant(
                '{"tool":"answer_from_context","arguments":{"answer":[[1]],'
            )

    def test_strict_parser_rejects_replay_only_forms(self):
        with self.assertRaises(ProtocolError):
            parse_assistant_strict(
                '<think>Count.</think>{"tool":"aggregate",'
                '"arguments":{"table":"items","column":"*","op":"count"}}'
            )
        with self.assertRaises(ProtocolError):
            parse_assistant_strict(
                '<think>Join.</think>{"tool":"join_tables",'
                '"arguments":{"left":"a","right":"b","on":[{"left":"id","right":"id"}]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "missing arguments"):
            parse_assistant_strict(
                '<think>Join.</think>{"tool":"join_tables","arguments":'
                '{"tables":["a","b"],"on":[[{"left":"id","right":"id"}]]}}'
            )

    def test_version5_join_shape_and_replay_compatibility(self):
        _, tool, args = parse_assistant_strict(
            '<think>Join the connected component.</think>{"tool":"join_tables",'
            '"arguments":{"base":"orders","joins":[{"table":"customers","on":'
            '[{"left":"orders.customer_id","right":"id"}]},{"table":"regions","on":'
            '[{"left":"customers.region_id","right":"id"}]}]}}'
        )
        self.assertEqual(tool, "join_tables")
        self.assertEqual(args["joins"][1]["table"], "regions")

        _, replay_tool, replay_args = parse_assistant(
            '{"tool":"join_tables","arguments":{"tables":["a","b"],'
            '"on":[[{"left":"id","right":"id"}]],"prefixes":["A","B"]}}'
        )
        self.assertEqual(replay_tool, "join_tables")
        self.assertEqual(replay_args["prefixes"], ["A", "B"])

    def test_version5_join_requires_unambiguous_namespaces(self):
        with self.assertRaisesRegex(ProtocolError, "semantic base_role/role"):
            parse_assistant_strict(
                '<think>Self join.</think>{"tool":"join_tables","arguments":'
                '{"base":"employees","joins":[{"table":"employees","on":'
                '[{"left":"employees.manager_id","right":"id"}]}]}}'
            )
        _, _, args = parse_assistant_strict(
            '<think>Self join with semantic roles.</think>{"tool":"join_tables",'
            '"arguments":{"base":"employees","base_role":"employee","joins":'
            '[{"table":"employees","role":"manager","on":'
            '[{"left":"employee.manager_id","right":"id"}]}]}}'
        )
        self.assertEqual(args["base_role"], "employee")

    def test_version5_join_rejects_noncanonical_edge_identifiers(self):
        with self.assertRaisesRegex(ProtocolError, "known_relation.column"):
            parse_assistant_strict(
                '<think>Join.</think>{"tool":"join_tables","arguments":'
                '{"base":"a","joins":[{"table":"b","on":[{"left":"id","right":"id"}]}]}}'
                ''
            )
        with self.assertRaisesRegex(ProtocolError, "bare column"):
            parse_assistant_strict(
                '<think>Join.</think>{"tool":"join_tables","arguments":'
                '{"base":"a","joins":[{"table":"b","on":'
                '[{"left":"a.id","right":"b.id"}]}]}}'
            )

    def test_strict_parser_rejects_unbounded_read_limit(self):
        with self.assertRaisesRegex(ProtocolError, "integer from 1 to 20"):
            parse_assistant_strict(
                '<think>Read everything.</think>{"tool":"read_subtable",'
                '"arguments":{"table":"items","limit":68}}'
            )
        _, tool, args = parse_assistant_strict(
            '<think>Read a bounded sample.</think>{"tool":"read_subtable",'
            '"arguments":{"table":"items","limit":20}}'
        )
        self.assertEqual(tool, "read_subtable")
        self.assertEqual(args["limit"], 20)

    def test_strict_parser_rejects_retired_filter_preview_argument(self):
        with self.assertRaisesRegex(ProtocolError, "unexpected arguments"):
            parse_assistant_strict(
                '<think>Filter.</think>{"tool":"condition_filter",'
                '"arguments":{"table":"items","conditions":{"column":"id","op":">",'
                '"value":1},"preview_k":5}}'
            )

    def test_version13_project_distinct_and_scalar_compute_shapes(self):
        _, tool, args = parse_assistant_strict(
            '<think>Keep unique names.</think>{"tool":"project",'
            '"arguments":{"table":"people","expressions":["first","last"],'
            '"distinct":true}}'
        )
        self.assertEqual(tool, "project")
        self.assertTrue(args["distinct"])

        _, tool, args = parse_assistant_strict(
            '<think>Compute the exact percentage.</think>'
            '{"tool":"scalar_compute","arguments":{"operation":"percent",'
            '"operands":[{"value_ref":"step_3"},{"value_ref":"step_2"}],'
            '"result_name":"percentage"}}'
        )
        self.assertEqual(tool, "scalar_compute")
        self.assertEqual(args["operands"][0], {"value_ref": "step_3"})

        _, tool, args = parse_assistant_strict(
            '<think>Reuse two named metrics from one aggregate row.</think>'
            '{"tool":"scalar_compute","arguments":{"operation":"percent",'
            '"operands":[{"value_ref":"step_7","column":"usa_nominees"},'
            '{"value_ref":"step_7","column":"total_nominees"}],'
            '"result_name":"percentage"}}'
        )
        self.assertEqual(tool, "scalar_compute")
        self.assertEqual(
            args["operands"],
            [
                {"value_ref": "step_7", "column": "usa_nominees"},
                {"value_ref": "step_7", "column": "total_nominees"},
            ],
        )

        with self.assertRaisesRegex(ProtocolError, "exactly value, value_ref, or value_ref\\+column"):
            parse_assistant_strict(
                '<think>Bad operand.</think>{"tool":"scalar_compute",'
                '"arguments":{"operation":"subtract","operands":'
                '[{"value_ref":"step_2","value":2},{"value":1}]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "column must be a non-empty column name"):
            parse_assistant_strict(
                '<think>Bad named metric.</think>{"tool":"scalar_compute",'
                '"arguments":{"operation":"subtract","operands":'
                '[{"value_ref":"step_2","column":""},{"value":1}]}}'
            )

    def test_version15_conditional_aggregate_shape(self):
        _, tool, args = parse_assistant_strict(
            '<think>Compute both counts on the same population.</think>'
            '{"tool":"group_aggregate","arguments":{"table":"patients","group_by":[],'
            '"aggregations":['
            '{"op":"count","column":"*","as":"female_count","where":'
            '{"column":"gender","op":"=","value":"F"}},'
            '{"op":"count","column":"*","as":"male_count","where":'
            '{"column":"gender","op":"=","value":"M"}}]}}'
        )
        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["aggregations"][0]["where"]["value"], "F")

        with self.assertRaisesRegex(ProtocolError, "unexpected fields"):
            parse_assistant_strict(
                '<think>Reject an invented aggregate field.</think>'
                '{"tool":"group_aggregate","arguments":{"table":"patients","group_by":[],'
                '"aggregations":[{"op":"count","column":"*","as":"n","filter":'
                '{"column":"gender","op":"=","value":"F"}}]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "named column"):
            parse_assistant_strict(
                '<think>Reject ambiguous whole-row distinct.</think>'
                '{"tool":"group_aggregate","arguments":{"table":"patients","group_by":[],'
                '"aggregations":[{"op":"count_distinct","column":"*","as":"n","where":'
                '{"column":"gender","op":"=","value":"F"}}]}}'
            )

    def test_version18_group_aggregate_columns_layout(self):
        _, tool, args = parse_assistant_strict(
            '<think>Count both categories into one ordered row.</think>'
            '{"tool":"group_aggregate","arguments":{"table":"patients","group_by":["gender"],'
            '"aggregations":[{"op":"count_distinct","column":"patient","as":"patient_count"}],'
            '"output_layout":"columns","category_values":["M","F"]}}'
        )
        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["category_values"], ["M", "F"])

        with self.assertRaisesRegex(ProtocolError, "same length"):
            parse_assistant_strict(
                '<think>Reject mismatched output columns.</think>'
                '{"tool":"group_aggregate","arguments":{"table":"patients","group_by":["gender"],'
                '"aggregations":[{"op":"count","column":"*","as":"patient_count"}],'
                '"output_layout":"columns","category_values":["M","F"],'
                '"output_columns":["count"]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "category_values must be unique"):
            parse_assistant_strict(
                '<think>Reject duplicate category slots.</think>'
                '{"tool":"group_aggregate","arguments":{"table":"patients",'
                '"group_by":["gender"],"aggregations":'
                '[{"op":"count","column":"*","as":"patient_count"}],'
                '"output_layout":"columns","category_values":["M","M"]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "unknown tool"):
            parse_assistant_strict(
                '<think>The separate reshape tool is retired.</think>'
                '{"tool":"pivot","arguments":{"table":"group_003","key_column":"gender",'
                '"value_column":"patient_count","key_values":["M","F"]}}'
            )
        _, replay_tool, _ = parse_assistant(
            '{"tool":"pivot","arguments":{"table":"group_003",'
            '"key_column":"gender","value_column":"patient_count",'
            '"key_values":["M","F"]}}'
        )
        self.assertEqual(replay_tool, "pivot")

    def test_version13_terminal_requires_grounded_table_only(self):
        _, tool, args = parse_assistant_strict(
            '<think>Cite the exact result table.</think>'
            '{"tool":"answer_from_context","arguments":'
            '{"evidence":{"table":"project_003"},"reason":"Exact result."}}'
        )
        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"], {"table": "project_003"})

        with self.assertRaisesRegex(ProtocolError, "unexpected arguments"):
            parse_assistant_strict(
                '<think>Do not hand-write data.</think>'
                '{"tool":"answer_from_context","arguments":'
                '{"evidence":null,"answer":[42]}}'
            )
        with self.assertRaisesRegex(ProtocolError, "grounded 1x1 table"):
            parse_assistant_strict(
                '<think>A scalar still needs evidence.</think>'
                '{"tool":"answer_from_context","arguments":{"evidence":null}}'
            )


if __name__ == "__main__":
    unittest.main()
