#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from batch_plan_protocol import (  # noqa: E402
    BATCH_CARRIER_INLINE_THINK,
    BATCH_CARRIER_PROVIDER_NATIVE,
    BatchPlanProtocolError,
    LocalReferenceError,
    build_batch_plan_messages,
    build_batch_plan_system_prompt,
    parse_batch_plan_assistant,
    parse_batch_plan_action,
    render_batch_plan_assistant,
    render_batch_observation,
    resolve_local_references,
)
from evaluate_batch_plan import (  # noqa: E402
    _execute_plan_action,
    _materialize_terminal_evidence,
    run_episode,
)
from executor import Harness  # noqa: E402
from rollout import new_ctx, overview  # noqa: E402


def block(calls: list[dict]) -> dict:
    return {"calls": calls}


class ActionBlockProtocolTests(unittest.TestCase):
    def test_prompt_declares_hybrid_blocks_and_branch_local_recovery(self):
        prompt = build_batch_plan_system_prompt(5)
        self.assertIn("action_block arguments are exactly", prompt)
        self.assertIn("calls contains 1 to 5", prompt)
        self.assertIn("A block may mix schema/value inspection", prompt)
        self.assertIn("A blocked\n  call is not attempted", prompt)
        self.assertIn('without "$"', prompt)
        self.assertNotIn("ONE global resident task plan", prompt)

    def test_action_block_requires_nonempty_calls_and_no_plan_ops(self):
        valid = json.dumps({
            "tool": "action_block",
            "arguments": {
                "calls": [{
                    "id": "schema",
                    "tool": "describe_table",
                    "arguments": {"tables": ["items"]},
                }],
            },
        })
        tool, arguments = parse_batch_plan_action(valid, max_batch_calls=8)
        self.assertEqual(tool, "action_block")
        self.assertEqual(arguments["calls"][0]["id"], "schema")

        for invalid_arguments in (
            {"calls": []},
            {"ops": [{"op": "create", "id": "solve"}], "calls": arguments["calls"]},
        ):
            with self.assertRaises(BatchPlanProtocolError):
                parse_batch_plan_action(
                    json.dumps({
                        "tool": "action_block",
                        "arguments": invalid_arguments,
                    }),
                    max_batch_calls=8,
                )

    def test_inline_student_carrier_round_trips_without_tool_call_tag(self):
        arguments = {
            "calls": [{
                "id": "schema",
                "tool": "describe_table",
                "arguments": {"tables": ["items"]},
            }],
        }
        rendered = render_batch_plan_assistant(
            "Inspect the unresolved schema.",
            "action_block",
            arguments,
        )
        self.assertNotIn("<tool_call>", rendered)
        reason, tool, parsed = parse_batch_plan_assistant(
            rendered,
            max_batch_calls=8,
        )
        self.assertEqual(reason, "Inspect the unresolved schema.")
        self.assertEqual(tool, "action_block")
        self.assertEqual(parsed, arguments)
        inline_prompt = build_batch_plan_system_prompt(
            8,
            assistant_carrier=BATCH_CARRIER_INLINE_THINK,
        )
        self.assertIn("<think>brief reason</think>", inline_prompt)
        self.assertIn("No <tool_call> tag", inline_prompt)

        with self.assertRaises(BatchPlanProtocolError):
            parse_batch_plan_assistant(
                json.dumps({"tool": "action_block", "arguments": arguments}),
                max_batch_calls=8,
            )

    def test_terminal_remains_separate_and_grounded(self):
        tool, arguments = parse_batch_plan_action(
            json.dumps({
                "tool": "answer_from_context",
                "arguments": {
                    "evidence": {
                        "table": "project_001",
                        "columns": ["category"],
                    },
                    "reason": "Exact answer relation.",
                },
            }),
            max_batch_calls=8,
        )
        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(arguments["evidence"]["table"], "project_001")
        self.assertEqual(arguments["evidence"]["columns"], ["category"])

        with self.assertRaises(BatchPlanProtocolError):
            parse_batch_plan_action(
                json.dumps({
                    "tool": "answer_from_context",
                    "arguments": {
                        "evidence": {"table": "project_001"},
                    },
                }),
                max_batch_calls=8,
            )

    def test_local_references_resolve_table_column_and_value_step(self):
        bindings = {
            "metric": {
                "status": "success",
                "step_id": "step_4",
                "table": "group_001",
                "columns": ["group_001.count"],
            }
        }
        declared = {"metric", "later"}
        resolved = resolve_local_references(
            {
                "table": "$metric",
                "operands": [{"value_ref": "$metric", "column": "count"}],
            },
            bindings,
            declared,
        )
        self.assertEqual(resolved["table"], "group_001")
        self.assertEqual(resolved["operands"][0]["value_ref"], "step_4")
        self.assertEqual(
            resolve_local_references(
                {"column": "$metric.count"}, bindings, declared
            )["column"],
            "group_001.count",
        )
        with self.assertRaises(LocalReferenceError):
            resolve_local_references(
                {"table": "$later"}, bindings, declared
            )
        with self.assertRaisesRegex(LocalReferenceError, "without"):
            resolve_local_references(
                {"table": "$filter_001"}, bindings, declared
            )

    def test_current_state_is_rendered_after_prior_block(self):
        messages = build_batch_plan_messages(
            system_prompt="system",
            overview={"tables": []},
            question="question",
            external_knowledge=None,
            state={
                "plan": [],
                "tables": {
                    "project_001": {
                        "columns": ["category"],
                        "row_count": 2,
                    }
                },
                "values": {},
            },
            last_error=None,
            legal_history=[{
                "assistant": (
                    '{"tool":"action_block","arguments":{"calls":['
                    '{"id":"x","tool":"describe_table","arguments":{"tables":["items"]}}]}}'
                ),
                "observation": "ACTION BLOCK RESULTS\n{}",
            }],
            history_turns=4,
        )
        self.assertIn('"project_001"', messages[-1]["content"])


class ActionBlockExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name, "test.sqlite")
        seed = sqlite3.connect(db_path)
        seed.executescript(
            "CREATE TABLE items(category TEXT, price INTEGER);"
            "INSERT INTO items VALUES ('a', 0), ('b', 2), ('c', 3);"
            "CREATE TABLE categories(category TEXT, label TEXT);"
            "INSERT INTO categories VALUES ('a', 'A'), ('b', 'B'), ('c', 'C');"
        )
        seed.close()
        self.harness = Harness(str(db_path))
        self.ctx = new_ctx(overview(self.harness))
        self.created: set[str] = set()
        self.error_counts: collections.Counter = collections.Counter()
        self.error_events: list[dict] = []
        self.prior_bindings: dict[str, dict] = {}

    def tearDown(self):
        self.harness.conn.close()
        self.temp_dir.cleanup()

    def execute(
        self,
        arguments,
        atomic_count=0,
        batch_index=1,
        *,
        structured_error_feedback=True,
        low_friction_interface=False,
        safe_low_friction_interface=False,
        interface_resolution_events=None,
    ):
        return _execute_plan_action(
            h=self.harness,
            ctx=self.ctx,
            arguments=arguments,
            created=self.created,
            atomic_count=atomic_count,
            model_turn=batch_index,
            batch_index=batch_index,
            table_output_rows=0,
            error_counts=self.error_counts,
            error_events=self.error_events,
            prior_bindings=self.prior_bindings,
            structured_error_feedback=structured_error_feedback,
            low_friction_interface=low_friction_interface,
            safe_low_friction_interface=safe_low_friction_interface,
            interface_resolution_events=interface_resolution_events,
        )

    def test_successful_chain_uses_only_primitive_step_ids(self):
        arguments = block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "exact",
                "tool": "project",
                "arguments": {
                    "table": "$positive",
                    "expressions": ["category"],
                },
            },
            {
                "id": "rows",
                "tool": "read_subtable",
                "arguments": {"table": "$exact", "limit": 20},
            },
        ])
        atomic_count, results, _, nonrecoverable = self.execute(arguments)
        self.assertEqual(atomic_count, 3)
        self.assertFalse(nonrecoverable)
        self.assertEqual(
            [item["status"] for item in results],
            ["success", "success", "success"],
        )
        self.assertEqual(
            [item["step_id"] for item in results],
            ["step_1", "step_2", "step_3"],
        )
        self.assertEqual(self.ctx["environment"].snapshot()["plan"], [])

    def test_root_failure_does_not_block_independent_later_call(self):
        arguments = block([
            {"id": "bad", "tool": "not_a_tool", "arguments": {}},
            {
                "id": "schema",
                "tool": "describe_table",
                "arguments": {"tables": ["items"]},
            },
            {
                "id": "dependent",
                "tool": "project",
                "arguments": {
                    "table": "$bad",
                    "expressions": ["category"],
                },
            },
        ])
        atomic_count, results, _, nonrecoverable = self.execute(arguments)
        self.assertEqual(atomic_count, 2)
        self.assertFalse(nonrecoverable)
        self.assertEqual(
            [item["status"] for item in results],
            ["error", "success", "blocked"],
        )
        blocked = results[-1]
        self.assertIsNone(blocked["step_id"])
        self.assertEqual(blocked["blocked_by"], ["bad"])
        self.assertEqual(blocked["root_causes"], ["bad"])
        self.assertEqual(len(self.error_events), 1)
        self.assertEqual(sum(self.error_counts.values()), 1)

        visible = render_batch_observation(1, results)
        self.assertIn('"block_status":"partial_failure"', visible)
        self.assertIn('"root_error_calls":["bad"]', visible)
        self.assertIn('"blocked_by":["bad"]', visible)
        self.assertIn('"call_id":"schema"', visible)

    def test_join_namespace_error_returns_exact_column_candidate_fact(self):
        arguments = block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "joined",
                "tool": "join_tables",
                "arguments": {
                    "base": "$positive",
                    "joins": [{
                        "table": "categories",
                        "on": [{
                            "left": "items.category",
                            "right": "category",
                        }],
                    }],
                },
            },
        ])
        _, results, _, _ = self.execute(arguments)
        error = results[1]["error"]
        facts = error["facts"]
        self.assertRegex(facts["input_table"], r"^filter_\d+$")
        resolution = facts["column_resolution"][0]
        self.assertEqual(resolution["provided"], "items.category")
        self.assertEqual(resolution["status"], "unique_suffix_only")
        self.assertRegex(resolution["candidates"][0], r"^filter_\d+\.category$")

        visible = json.loads(
            render_batch_observation(
                1,
                results,
                structured_error_feedback=True,
            ).split("\n", 1)[1]
        )
        reusable = visible["reusable_outputs"]["positive"]
        self.assertRegex(reusable["table"], r"^filter_\d+$")
        self.assertEqual(reusable["columns"], ["category", "price"])

    def test_multiedge_join_facts_use_progressively_introduced_namespaces(self):
        arguments = block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "joined",
                "tool": "join_tables",
                "arguments": {
                    "base": "$positive",
                    "joins": [
                        {
                            "table": "categories",
                            "on": [{
                                "left": "items.category",
                                "right": "category",
                            }],
                        },
                        {
                            "table": "items",
                            "role": "other_items",
                            "on": [{
                                "left": "categories.category",
                                "right": "category",
                            }],
                        },
                    ],
                },
            },
        ])
        _, results, _, _ = self.execute(arguments)
        resolutions = results[1]["error"]["facts"]["column_resolution"]
        self.assertEqual(
            [item["provided"] for item in resolutions],
            ["items.category"],
        )

    def test_v4_default_does_not_change_error_observation_shape(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "schema",
                    "tool": "describe_table",
                    "arguments": {"tables": ["items"]},
                },
                {"id": "bad", "tool": "not_a_tool", "arguments": {}},
            ]),
            structured_error_feedback=False,
        )
        self.assertNotIn("facts", results[1]["error"])
        visible = json.loads(
            render_batch_observation(1, results).split("\n", 1)[1]
        )
        self.assertIsInstance(visible["reusable_outputs"], list)

    def test_low_friction_resolves_unique_stale_join_namespace(self):
        events = []
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "positive",
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "items",
                        "conditions": {"column": "price", "op": ">", "value": 0},
                    },
                },
                {
                    "id": "joined",
                    "tool": "join_tables",
                    "arguments": {
                        "base": "$positive",
                        "joins": [{
                            "table": "categories",
                            "on": [{
                                "left": "items.category",
                                "right": "category",
                            }],
                        }],
                    },
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
            interface_resolution_events=events,
        )
        self.assertEqual([result["status"] for result in results], ["success", "success"])
        resolution = results[1]["interface_resolutions"][0]
        self.assertEqual(resolution["provided"], "items.category")
        self.assertRegex(resolution["resolved"], r"^filter_\d+\.category$")
        self.assertEqual(resolution["rule"], "unique_column_suffix")
        self.assertEqual(events[0]["call_id"], "joined")

    def test_low_friction_resolves_persistent_prior_block_local_id(self):
        self.execute(
            block([{
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            }]),
            batch_index=1,
            structured_error_feedback=False,
            low_friction_interface=True,
        )
        _, results, _, _ = self.execute(
            block([{
                "id": "rows",
                "tool": "read_subtable",
                "arguments": {"table": "$positive", "limit": 20},
            }]),
            atomic_count=1,
            batch_index=2,
            structured_error_feedback=False,
            low_friction_interface=True,
        )
        self.assertEqual(results[0]["status"], "success")
        resolution = results[0]["interface_resolutions"][0]
        self.assertEqual(
            resolution["rule"],
            "persistent_prior_block_call_reference",
        )
        self.assertEqual(resolution["prior_block_index"], 1)

    def test_low_friction_resolves_missing_same_block_sigil(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "positive",
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "items",
                        "conditions": {"column": "price", "op": ">", "value": 0},
                    },
                },
                {
                    "id": "rows",
                    "tool": "read_subtable",
                    "arguments": {"table": "positive", "limit": 20},
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
        )
        self.assertEqual([result["status"] for result in results], ["success", "success"])
        resolution = results[1]["interface_resolutions"][0]
        self.assertEqual(
            resolution["rule"],
            "implicit_same_block_table_reference",
        )
        self.assertRegex(resolution["resolved"], r"^filter_\d+$")

    def test_low_friction_resolves_ambiguous_inner_join_equivalent_key(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "joined",
                    "tool": "join_tables",
                    "arguments": {
                        "base": "items",
                        "joins": [{
                            "table": "categories",
                            "on": [{
                                "left": "items.category",
                                "right": "category",
                            }],
                            "type": "inner",
                        }],
                    },
                },
                {
                    "id": "youngest",
                    "tool": "extreme_value_select",
                    "arguments": {
                        "table": "$joined",
                        "order_by": ["price DESC"],
                        "top_k": 1,
                        "return_columns": ["category", "price", "label"],
                    },
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
        )
        self.assertEqual([result["status"] for result in results], ["success", "success"])
        equivalent = [
            item
            for item in results[1]["interface_resolutions"]
            if item["rule"] == "inner_join_equivalent_columns"
        ]
        self.assertEqual(len(equivalent), 1)
        self.assertEqual(equivalent[0]["provided"], "category")

    def test_safe_low_friction_rejects_column_reference_as_literal(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "positive",
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "items",
                        "conditions": {"column": "price", "op": ">", "value": 0},
                    },
                },
                {
                    "id": "invalid_literal",
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "items",
                        "conditions": {
                            "column": "category",
                            "op": "=",
                            "value": "$positive.category",
                        },
                    },
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
            safe_low_friction_interface=True,
        )
        self.assertEqual(results[1]["status"], "error")
        self.assertIn(
            "cannot supply a literal value",
            results[1]["error"]["message"],
        )

    def test_safe_low_friction_resolves_progressive_join_equivalence(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "positive",
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "items",
                        "conditions": {"column": "price", "op": ">", "value": 0},
                    },
                },
                {
                    "id": "joined",
                    "tool": "join_tables",
                    "arguments": {
                        "base": "$positive",
                        "joins": [
                            {
                                "table": "categories",
                                "on": [{
                                    "left": "items.category",
                                    "right": "category",
                                }],
                            },
                            {
                                "table": "items",
                                "role": "other_items",
                                "on": [{
                                    "left": "items.category",
                                    "right": "category",
                                }],
                            },
                        ],
                    },
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
            safe_low_friction_interface=True,
        )
        self.assertEqual([result["status"] for result in results], ["success", "success"])
        equivalent = [
            item
            for item in results[1]["interface_resolutions"]
            if item["rule"] == "inner_join_equivalent_columns"
        ]
        self.assertEqual(len(equivalent), 1)

    def test_safe_low_friction_resolves_equivalent_local_column(self):
        _, results, _, _ = self.execute(
            block([
                {
                    "id": "joined",
                    "tool": "join_tables",
                    "arguments": {
                        "base": "items",
                        "joins": [{
                            "table": "categories",
                            "on": [{
                                "left": "items.category",
                                "right": "category",
                            }],
                            "type": "inner",
                        }],
                    },
                },
                {
                    "id": "rows",
                    "tool": "read_subtable",
                    "arguments": {
                        "table": "$joined",
                        "columns": ["$joined.category"],
                        "limit": 20,
                    },
                },
            ]),
            structured_error_feedback=False,
            low_friction_interface=True,
            safe_low_friction_interface=True,
        )
        self.assertEqual([result["status"] for result in results], ["success", "success"])
        local = [
            item
            for item in results[1]["interface_resolutions"]
            if item["rule"] == "local_inner_join_equivalent_columns"
        ]
        self.assertEqual(len(local), 1)

    def test_expired_local_reference_returns_prior_binding_fact(self):
        self.execute(block([{
            "id": "positive",
            "tool": "condition_filter",
            "arguments": {
                "table": "items",
                "conditions": {"column": "price", "op": ">", "value": 0},
            },
        }]), batch_index=1)
        _, results, _, _ = self.execute(block([{
            "id": "rows",
            "tool": "read_subtable",
            "arguments": {"table": "$positive", "limit": 20},
        }]), atomic_count=1, batch_index=2)
        fact = results[0]["error"]["facts"]["reference_resolution"][0]
        self.assertEqual(fact["status"], "not_declared_in_current_block")
        self.assertEqual(fact["prior_binding"]["block_index"], 1)
        self.assertRegex(fact["prior_binding"]["table"], r"^filter_\d+$")

    def test_missing_local_sigil_returns_current_binding_fact(self):
        _, results, _, _ = self.execute(block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "rows",
                "tool": "read_subtable",
                "arguments": {"table": "positive", "limit": 20},
            },
        ]))
        fact = results[1]["error"]["facts"]["reference_resolution"][0]
        self.assertEqual(
            fact["status"],
            "matches_current_call_id_without_local_reference",
        )
        self.assertEqual(fact["local_reference"], "$positive")
        self.assertRegex(fact["current_binding"]["table"], r"^filter_\d+$")

    def test_multrow_scalar_error_returns_source_shape_fact(self):
        _, results, _, _ = self.execute(block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "bad_scalar",
                "tool": "scalar_compute",
                "arguments": {
                    "operation": "add",
                    "operands": [
                        {"value_ref": "$positive", "column": "price"},
                        {"value": 1},
                    ],
                    "result_name": "bad",
                },
            },
        ]))
        source = results[1]["error"]["facts"]["scalar_sources"][0]
        self.assertEqual(source["value_ref"], "step_1")
        self.assertEqual(source["row_count"], 2)
        self.assertEqual(source["columns"], ["category", "price"])
        self.assertRegex(source["table"], r"^filter_\d+$")

    def test_blocked_descendants_preserve_transitive_root_cause(self):
        arguments = block([
            {"id": "bad", "tool": "not_a_tool", "arguments": {}},
            {
                "id": "child",
                "tool": "project",
                "arguments": {"table": "$bad", "expressions": ["category"]},
            },
            {
                "id": "grandchild",
                "tool": "read_subtable",
                "arguments": {"table": "$child", "limit": 20},
            },
        ])
        atomic_count, results, _, _ = self.execute(arguments)
        self.assertEqual(atomic_count, 1)
        self.assertEqual(
            [item["status"] for item in results],
            ["error", "blocked", "blocked"],
        )
        self.assertEqual(results[-1]["blocked_by"], ["child"])
        self.assertEqual(results[-1]["root_causes"], ["bad"])
        self.assertEqual(len(self.error_events), 1)

    def test_missing_or_cross_block_dollar_reference_is_root_error(self):
        arguments = block([{
            "id": "bad_ref",
            "tool": "read_subtable",
            "arguments": {"table": "$filter_001", "limit": 20},
        }])
        atomic_count, results, _, _ = self.execute(arguments)
        self.assertEqual(atomic_count, 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(
            results[0]["error"]["type"], "argument_validation_error"
        )
        self.assertIn('without "$"', results[0]["error"]["message"])
        self.assertEqual(len(self.error_events), 1)

    def test_local_column_reference_supports_filter_then_join(self):
        arguments = block([
            {
                "id": "positive",
                "tool": "condition_filter",
                "arguments": {
                    "table": "items",
                    "conditions": {"column": "price", "op": ">", "value": 0},
                },
            },
            {
                "id": "joined",
                "tool": "join_tables",
                "arguments": {
                    "base": "$positive",
                    "joins": [{
                        "table": "categories",
                        "on": [{
                            "left": "$positive.category",
                            "right": "category",
                        }],
                    }],
                },
            },
        ])
        atomic_count, results, _, nonrecoverable = self.execute(arguments)
        self.assertEqual(atomic_count, 2)
        self.assertFalse(nonrecoverable)
        self.assertEqual(
            [item["status"] for item in results],
            ["success", "success"],
        )
        resolved_left = (
            results[-1]["resolved_arguments"]["joins"][0]["on"][0]["left"]
        )
        self.assertRegex(resolved_left, r"^filter_\d+\.category$")

    def test_terminal_projection_drops_helper_columns_without_changing_rows(self):
        _, results, _, _ = self.execute(block([{
            "id": "all_rows",
            "tool": "condition_filter",
            "arguments": {
                "table": "items",
                "conditions": {"column": "price", "op": ">=", "value": 0},
            },
        }]))
        source_table = results[0]["table"]
        score_arguments, projection = _materialize_terminal_evidence(
            h=self.harness,
            ctx=self.ctx,
            arguments={
                "evidence": {
                    "table": source_table,
                    "columns": ["category"],
                },
            },
            created=self.created,
            step_id="step_2",
        )
        projected = score_arguments["evidence"]["table"]
        self.assertEqual(projection["source_columns"], ["category", "price"])
        self.assertEqual(projection["selected_columns"], ["category"])
        self.assertEqual(self.harness.table_columns(projected), ["category"])
        self.assertEqual(
            [list(row) for row in self.harness.rows(projected)],
            [["a"], ["b"], ["c"]],
        )


class ActionBlockEpisodeTests(unittest.TestCase):
    def run_with_responses(
        self,
        db_path: Path,
        responses: list[str],
        *,
        assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
    ) -> dict:
        queued = list(responses)

        def fake_chat(**_kwargs):
            return queued.pop(0), {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "api_request_attempts": 1,
            }, "Use the next grounded action."

        with patch("evaluate_batch_plan.chat_with_retries", side_effect=fake_chat):
            return run_episode(
                example_index=0,
                ex={
                    "trajectory_id": "test_0",
                    "db_id": "test",
                    "db_path": str(db_path),
                    "question": "Return all categories.",
                    "gold_sql": "SELECT category FROM items",
                },
                split="train",
                base_url="http://unused",
                api_key="unused",
                model="deepseek-v4-flash",
                system_prompt=build_batch_plan_system_prompt(
                    8,
                    assistant_carrier=assistant_carrier,
                ),
                protocol_hash="test",
                max_atomic_actions=30,
                max_model_turns=30,
                max_batch_calls=8,
                max_tokens=2048,
                api_retries=1,
                api_timeout=10,
                max_errors_per_type=3,
                table_output_rows=0,
                history_turns=4,
                denotation_comparison="bird-set",
                assistant_carrier=assistant_carrier,
            )

    def test_episode_answers_without_resident_plan_bookkeeping(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "test.sqlite")
            seed = sqlite3.connect(db_path)
            seed.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 0), ('b', 2);"
            )
            seed.close()
            record = self.run_with_responses(db_path, [
                json.dumps({
                    "tool": "action_block",
                    "arguments": {
                        "calls": [{
                            "id": "schema",
                            "tool": "describe_table",
                            "arguments": {"tables": ["items"]},
                        }],
                    },
                }),
                json.dumps({
                    "tool": "action_block",
                    "arguments": {
                        "calls": [
                            {
                                "id": "exact",
                                "tool": "project",
                                "arguments": {
                                    "table": "items",
                                    "expressions": ["category"],
                                },
                            },
                            {
                                "id": "rows",
                                "tool": "read_subtable",
                                "arguments": {"table": "$exact", "limit": 20},
                            },
                        ],
                    },
                }),
                json.dumps({
                    "tool": "answer_from_context",
                    "arguments": {
                        "evidence": {
                            "table": "project_001",
                            "columns": ["category"],
                        },
                        "reason": "Exact category relation.",
                    },
                }),
            ])
        self.assertTrue(
            record["correct"],
            json.dumps({
                "failure_type": record.get("failure_type"),
                "fail": record.get("fail"),
                "errors": record.get("error_events"),
                "turns": record.get("turns"),
            }, default=str),
        )
        self.assertEqual(record["model_turns"], 3)
        self.assertEqual(record["action_blocks"], 2)
        self.assertEqual(record["atomic_actions"], 4)
        self.assertEqual(record["planned_nodes"], 3)
        self.assertEqual(record["blocked_nodes"], 0)
        self.assertEqual(record["final_environment_state"]["plan"], [])

    def test_inline_student_carrier_runs_through_the_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "test.sqlite")
            seed = sqlite3.connect(db_path)
            seed.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 0), ('b', 2);"
            )
            seed.close()
            responses = [
                render_batch_plan_assistant(
                    "Create the exact result relation.",
                    "action_block",
                    {
                        "calls": [
                            {
                                "id": "exact",
                                "tool": "project",
                                "arguments": {
                                    "table": "items",
                                    "expressions": ["category"],
                                },
                            },
                            {
                                "id": "rows",
                                "tool": "read_subtable",
                                "arguments": {"table": "$exact", "limit": 20},
                            },
                        ],
                    },
                ),
                render_batch_plan_assistant(
                    "The resident relation has exactly the requested column.",
                    "answer_from_context",
                    {
                        "evidence": {
                            "table": "project_001",
                            "columns": ["category"],
                        }
                    },
                ),
            ]
            record = self.run_with_responses(
                db_path,
                responses,
                assistant_carrier=BATCH_CARRIER_INLINE_THINK,
            )
        self.assertTrue(record["correct"])
        self.assertEqual(
            record["assistant_carrier"],
            BATCH_CARRIER_INLINE_THINK,
        )
        self.assertEqual(record["model_turns"], 2)

    def test_episode_recovers_from_root_error_with_blocked_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "test.sqlite")
            seed = sqlite3.connect(db_path)
            seed.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 0), ('b', 2);"
            )
            seed.close()
            record = self.run_with_responses(db_path, [
                json.dumps({
                    "tool": "action_block",
                    "arguments": {
                        "calls": [
                            {
                                "id": "wrong",
                                "tool": "condition_filter",
                                "arguments": {
                                    "table": "items",
                                    "conditions": {
                                        "column": "missing_column",
                                        "op": "=",
                                        "value": "a",
                                    },
                                },
                            },
                            {
                                "id": "blocked_read",
                                "tool": "read_subtable",
                                "arguments": {"table": "$wrong", "limit": 20},
                            },
                            {
                                "id": "schema",
                                "tool": "describe_table",
                                "arguments": {"tables": ["items"]},
                            },
                        ],
                    },
                }),
                json.dumps({
                    "tool": "action_block",
                    "arguments": {
                        "calls": [
                            {
                                "id": "exact",
                                "tool": "project",
                                "arguments": {
                                    "table": "items",
                                    "expressions": ["category"],
                                },
                            },
                            {
                                "id": "rows",
                                "tool": "read_subtable",
                                "arguments": {"table": "$exact", "limit": 20},
                            },
                        ],
                    },
                }),
                json.dumps({
                    "tool": "answer_from_context",
                    "arguments": {
                        "evidence": {
                            "table": "project_002",
                            "columns": ["category"],
                        },
                        "reason": "Corrected from the schema feedback.",
                    },
                }),
            ])
        self.assertTrue(
            record["correct"],
            json.dumps({
                "failure_type": record.get("failure_type"),
                "fail": record.get("fail"),
                "errors": record.get("error_events"),
                "turns": record.get("turns"),
            }, default=str),
        )
        self.assertEqual(record["errors"], 1)
        self.assertEqual(record["blocked_nodes"], 1)
        self.assertEqual(record["atomic_actions"], 5)
        first = record["turns"][0]
        self.assertEqual(first["root_error_count"], 1)
        self.assertEqual(first["blocked_count"], 1)
        self.assertIn('"root_error_calls":["wrong"]', first["observation"])
        self.assertIn('"root_causes":["wrong"]', first["observation"])

    def test_episode_recovers_from_invalid_terminal_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "test.sqlite")
            seed = sqlite3.connect(db_path)
            seed.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 0), ('b', 2);"
            )
            seed.close()
            record = self.run_with_responses(db_path, [
                json.dumps({
                    "tool": "action_block",
                    "arguments": {
                        "calls": [{
                            "id": "candidate",
                            "tool": "project",
                            "arguments": {
                                "table": "items",
                                "expressions": ["category", "price"],
                            },
                        }],
                    },
                }),
                json.dumps({
                    "tool": "answer_from_context",
                    "arguments": {
                        "evidence": {
                            "table": "project_001",
                            "columns": ["missing"],
                        },
                    },
                }),
                json.dumps({
                    "tool": "answer_from_context",
                    "arguments": {
                        "evidence": {
                            "table": "project_001",
                            "columns": ["category"],
                        },
                    },
                }),
            ])
        self.assertTrue(record["correct"])
        self.assertEqual(record["errors"], 1)
        self.assertEqual(
            record["error_events"][0]["error_type"],
            "argument_validation_error",
        )
        self.assertIn(
            "available columns",
            record["error_events"][0]["message"],
        )
        self.assertEqual(record["atomic_actions"], 3)
        terminal = record["atomic_events"][-1]
        self.assertEqual(
            terminal["terminal_projection"]["selected_columns"],
            ["category"],
        )


if __name__ == "__main__":
    unittest.main()
