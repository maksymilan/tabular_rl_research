#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from evaluate_batch_plan import _execute_action_block  # noqa: E402
from executor import Harness  # noqa: E402
from relational_program_protocol import (  # noqa: E402
    RelationalProgramProtocolError,
    build_relational_program_system_prompt,
    compile_relational_program,
    parse_relational_program_action,
    prepare_relational_work_action,
    render_relational_program_observation,
    validate_relational_program_call,
)
from rollout import new_ctx, overview  # noqa: E402


def program(calls: list[dict], result: str, exports: list[str] | None = None) -> dict:
    value = {"calls": calls, "result": result}
    if exports is not None:
        value["exports"] = exports
    return value


class RelationalProgramProtocolTests(unittest.TestCase):
    def test_prompt_is_exclusive_to_the_new_three_tool_interface(self):
        prompt = build_relational_program_system_prompt()
        self.assertIn("exactly three top-level tools", prompt)
        self.assertIn("TOP-LEVEL TOOL 1: observe", prompt)
        self.assertIn("TOP-LEVEL TOOL 2: relational_program", prompt)
        self.assertIn("TOP-LEVEL TOOL 3: answer_from_context", prompt)
        self.assertIn("List order is not execution order", prompt)
        self.assertIn("derives the DAG", prompt)
        self.assertIn("Programs may be used multiple", prompt)
        self.assertIn('{"source_table":"orders"}', prompt)
        self.assertIn('{"resident_table":"filter_002"}', prompt)
        self.assertIn('{"node":"filtered"}', prompt)
        self.assertIn('{"resident_step":"step_7"}', prompt)
        self.assertNotIn('"$id"', prompt)
        self.assertNotIn("<think>", prompt)
        for retired_name in (
            "describe_table",
            "inspect_column",
            "read_subtable",
            "condition_filter",
            "project",
            "scalar_compute",
            "join_tables",
            "group_aggregate",
            "extreme_value_select",
            "set_op",
            "action_block",
        ):
            self.assertNotIn(retired_name, prompt)

    def test_unordered_calls_compile_to_a_harness_owned_dag(self):
        arguments = program([
            {
                "id": "exact",
                "operation": "select",
                "arguments": {
                    "table": {"node": "filtered"},
                    "expressions": ["category"],
                },
            },
            {
                "id": "filtered",
                "operation": "filter",
                "arguments": {
                    "table": {"source_table": "items"},
                    "conditions": {
                        "column": "price",
                        "op": ">",
                        "value": 0,
                    },
                },
            },
        ], "exact")
        executable, graph = compile_relational_program(arguments)
        self.assertEqual(
            [call["id"] for call in executable["calls"]],
            ["filtered", "exact"],
        )
        self.assertEqual(graph["dependencies"]["exact"], ["filtered"])
        self.assertEqual(graph["result"], "exact")
        self.assertEqual(graph["reference_schema"], "typed-relational-reference-v2")
        self.assertEqual(
            executable["calls"][1]["arguments"]["table"],
            "$filtered",
        )
        self.assertEqual(
            graph["authored_references"]["filtered"],
            [{
                "path": "table",
                "kind": "source_table",
                "target": "items",
            }],
        )

    def test_typed_reference_kinds_lower_without_cross_program_dependencies(self):
        executable, graph = compile_relational_program(program([
            {
                "id": "combined",
                "operation": "combine",
                "arguments": {
                    "left": {"resident_table": "project_002"},
                    "right": {"source_table": "archive"},
                    "op": "union",
                },
            },
            {
                "id": "metric",
                "operation": "scalar",
                "arguments": {
                    "operation": "percent",
                    "operands": [
                        {"node": "combined"},
                        {"resident_step": "step_7"},
                    ],
                },
            },
        ], "metric"))
        self.assertEqual(graph["dependencies"]["combined"], [])
        self.assertEqual(graph["dependencies"]["metric"], ["combined"])
        self.assertEqual(
            executable["calls"][0]["arguments"],
            {
                "left": "project_002",
                "right": "archive",
                "op": "union",
            },
        )
        self.assertEqual(
            executable["calls"][1]["arguments"]["operands"],
            [
                {"value_ref": "$combined"},
                {"value_ref": "step_7"},
            ],
        )

    def test_scalar_named_cells_and_predicate_value_from_lower_canonically(self):
        executable, graph = compile_relational_program(program([
            {
                "id": "latest",
                "operation": "filter",
                "arguments": {
                    "table": {"source_table": "papers"},
                    "conditions": {
                        "column": "year",
                        "op": "=",
                        "value_from": {"node": "metrics"},
                    },
                },
            },
            {
                "id": "metrics",
                "operation": "aggregate",
                "arguments": {
                    "table": {"source_table": "papers"},
                    "group_by": [],
                    "aggregations": [{
                        "op": "max",
                        "column": "year",
                        "as": "max_year",
                    }],
                },
            },
            {
                "id": "ratio",
                "operation": "scalar",
                "arguments": {
                    "operation": "percent",
                    "operands": [
                        {"node": "metrics", "column": "max_year"},
                        {"resident_step": "step_7", "column": "baseline"},
                    ],
                },
            },
        ], "latest", exports=["ratio"]))
        self.assertEqual(
            graph["dependencies"],
            {
                "metrics": [],
                "latest": ["metrics"],
                "ratio": ["metrics"],
            },
        )
        by_id = {
            call["id"]: call["arguments"]
            for call in executable["calls"]
        }
        self.assertEqual(
            by_id["latest"]["conditions"]["value_ref"],
            "$metrics",
        )
        self.assertEqual(
            by_id["ratio"]["operands"],
            [
                {"value_ref": "$metrics", "column": "max_year"},
                {"value_ref": "step_7", "column": "baseline"},
            ],
        )

    def test_join_node_column_reference_lowers_to_private_executor_carrier(self):
        executable, graph = compile_relational_program(program([
            {
                "id": "joined",
                "operation": "join",
                "arguments": {
                    "base": {"node": "filtered"},
                    "joins": [{
                        "table": {"source_table": "customers"},
                        "on": [{
                            "left": {
                                "node": "filtered",
                                "column": "customer_id",
                            },
                            "right": "id",
                        }],
                    }],
                },
            },
            {
                "id": "filtered",
                "operation": "filter",
                "arguments": {
                    "table": {"source_table": "orders"},
                    "conditions": {
                        "column": "status",
                        "op": "=",
                        "value": "open",
                    },
                },
            },
        ], "joined"))
        self.assertEqual(graph["topological_order"], ["filtered", "joined"])
        self.assertEqual(
            executable["calls"][1]["arguments"]["joins"][0]["on"][0]["left"],
            "$filtered.customer_id",
        )

    def test_raw_or_legacy_references_are_rejected(self):
        raw_source = program([{
            "id": "exact",
            "operation": "select",
            "arguments": {
                "table": "items",
                "expressions": ["category"],
            },
        }], "exact")
        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "must be a typed object",
        ):
            compile_relational_program(raw_source)

        legacy_node = program([{
            "id": "exact",
            "operation": "select",
            "arguments": {
                "table": "$filtered",
                "expressions": ["category"],
            },
        }, {
            "id": "filtered",
            "operation": "filter",
            "arguments": {
                "table": {"source_table": "items"},
                "conditions": {"column": "price", "op": ">", "value": 0},
            },
        }], "exact")
        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "legacy local reference",
        ):
            compile_relational_program(legacy_node)

        nested_value_ref = program([{
            "id": "metric",
            "operation": "scalar",
            "arguments": {
                "operation": "add",
                "operands": [
                    {"value_ref": {"resident_step": "step_7"}},
                    {"value": 1},
                ],
            },
        }], "metric")
        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "value_ref .* is not public",
        ):
            compile_relational_program(nested_value_ref)

    def test_static_gate_rejects_undefined_cycles_and_disconnected_nodes(self):
        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "undefined calls",
        ):
            compile_relational_program(program([
                {
                    "id": "exact",
                    "operation": "select",
                    "arguments": {
                        "table": {"node": "missing"},
                        "expressions": ["category"],
                    },
                },
            ], "exact"))

        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "dependency cycle",
        ):
            compile_relational_program(program([
                {
                    "id": "left",
                    "operation": "select",
                    "arguments": {
                        "table": {"node": "right"},
                        "expressions": ["category"],
                    },
                },
                {
                    "id": "right",
                    "operation": "select",
                    "arguments": {
                        "table": {"node": "left"},
                        "expressions": ["category"],
                    },
                },
            ], "left"))

        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "disconnected calls",
        ):
            compile_relational_program(program([
                {
                    "id": "answer",
                    "operation": "select",
                    "arguments": {
                        "table": {"source_table": "items"},
                        "expressions": ["category"],
                    },
                },
                {
                    "id": "unused",
                    "operation": "filter",
                    "arguments": {
                        "table": {"source_table": "items"},
                        "conditions": {
                            "column": "price",
                            "op": ">",
                            "value": 0,
                        },
                    },
                },
            ], "answer"))

    def test_parser_allows_direct_perception_program_and_terminal_only(self):
        for action in (
            {
                "tool": "observe",
                "arguments": {
                    "operation": "schema",
                    "tables": ["items"],
                },
            },
            {
                "tool": "relational_program",
                "arguments": program([{
                    "id": "exact",
                    "operation": "select",
                    "arguments": {
                        "table": {"source_table": "items"},
                        "expressions": ["category"],
                    },
                }], "exact"),
            },
            {
                "tool": "answer_from_context",
                "arguments": {"evidence": {"table": "project_001"}},
            },
        ):
            tool, arguments = parse_relational_program_action(json.dumps(action))
            self.assertEqual(tool, action["tool"])
            self.assertEqual(arguments, action["arguments"])

        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "operation must be one of",
        ):
            parse_relational_program_action(json.dumps({
                "tool": "relational_program",
                "arguments": program([{
                    "id": "rows",
                    "operation": "rows",
                    "arguments": {"table": "items", "limit": 3},
                }], "rows"),
            }))

    def test_observe_rows_rejects_filter_arguments_with_precise_feedback(self):
        with self.assertRaisesRegex(
            RelationalProgramProtocolError,
            "reads rows from exactly one supplied table without filtering",
        ):
            parse_relational_program_action(json.dumps({
                "tool": "observe",
                "arguments": {
                    "operation": "rows",
                    "table": "items",
                    "conditions": {
                        "column": "price",
                        "op": ">",
                        "value": 0,
                    },
                },
            }))

    def test_observe_maps_to_internal_executor_without_leaking_old_names(self):
        executable, graph = prepare_relational_work_action(
            "observe",
            {"operation": "schema", "tables": ["items"]},
        )
        self.assertEqual(executable["calls"][0]["tool"], "describe_table")
        self.assertEqual(
            graph["authored_operations"],
            {"observation": "observe.schema"},
        )
        rendered = render_relational_program_observation(
            1,
            [{
                "call_id": "observation",
                "step_id": "step_1",
                "tool": "describe_table",
                "status": "success",
                "output": {
                    "tables": {"items": {"columns": ["category"]}},
                    "derivation": {"operator": "describe_table"},
                },
            }],
        )
        self.assertIn('"operation":"observe.schema"', rendered)
        self.assertNotIn("describe_table", rendered)

    def test_private_executor_names_and_refs_do_not_leak_in_error_feedback(self):
        rendered = render_relational_program_observation(
            2,
            [{
                "call_id": "joined",
                "step_id": "step_4",
                "tool": "join_tables",
                "status": "error",
                "error": {
                    "type": "argument_validation_error",
                    "message": (
                        "join_tables on local reference "
                        "'$filtered.customer_id' failed"
                    ),
                    "facts": {
                        "reference": "$filtered",
                        "resident": "project_001",
                    },
                },
            }],
        )
        self.assertIn('"operation":"join"', rendered)
        self.assertIn('"reference":{"node":"filtered"}', rendered)
        self.assertIn(
            '{\\"node\\":\\"filtered\\",\\"column\\":\\"customer_id\\"}',
            rendered,
        )
        self.assertIn("project_001", rendered)
        self.assertNotIn("join_tables", rendered)
        self.assertNotIn("$filtered", rendered)


class RelationalProgramExecutionTests(unittest.TestCase):
    def test_compiled_program_executes_in_topological_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir, "test.sqlite")
            seed = sqlite3.connect(db_path)
            seed.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 0), ('b', 2), ('c', 3);"
            )
            seed.close()
            harness = Harness(str(db_path))
            try:
                ctx = new_ctx(overview(harness))
                executable, graph = prepare_relational_work_action(
                    "relational_program",
                    program([
                        {
                            "id": "exact",
                            "operation": "select",
                            "arguments": {
                                "table": {"node": "filtered"},
                                "expressions": ["category"],
                            },
                        },
                        {
                            "id": "filtered",
                            "operation": "filter",
                            "arguments": {
                                "table": {"source_table": "items"},
                                "conditions": {
                                    "column": "price",
                                    "op": ">",
                                    "value": 0,
                                },
                            },
                        },
                    ], "exact"),
                )
                atomic_count, results, _, nonrecoverable = _execute_action_block(
                    h=harness,
                    ctx=ctx,
                    arguments=executable,
                    created=set(),
                    atomic_count=0,
                    model_turn=1,
                    batch_index=1,
                    table_output_rows=0,
                    error_counts=collections.Counter(),
                    error_events=[],
                    validate_call=validate_relational_program_call,
                )
                self.assertEqual(graph["topological_order"], ["filtered", "exact"])
                self.assertEqual(atomic_count, 2)
                self.assertFalse(nonrecoverable)
                self.assertEqual(
                    [(item["call_id"], item["status"]) for item in results],
                    [("filtered", "success"), ("exact", "success")],
                )
                self.assertEqual(results[-1]["output"]["row_count"], 2)
            finally:
                harness.conn.close()


if __name__ == "__main__":
    unittest.main()
