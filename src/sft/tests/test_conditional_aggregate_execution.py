#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval", "src/rl"):
    sys.path.insert(0, str(ROOT / relative))

from executor import Harness  # noqa: E402
from rollout import execute_tool, new_ctx  # noqa: E402


class ConditionalAggregateExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness(":memory:")
        self.addCleanup(self.harness.conn.close)

    def test_online_execution_resolves_where_value_ref_and_keeps_model_arguments(self):
        harness = self.harness
        harness.conn.executescript(
            """
            CREATE TABLE scores(name TEXT, score INT);
            INSERT INTO scores VALUES ('a', 10), ('b', 20), ('c', 30);
            """
        )
        harness.register_sources()
        context = new_ctx()

        mean_output, _ = execute_tool(
            harness,
            "group_aggregate",
            {
                "table": "scores",
                "group_by": [],
                "aggregations": [{"op": "mean", "column": "score", "as": "mean_score"}],
            },
            context,
            "step_1",
        )
        count_args = {
            "table": "scores",
            "group_by": [],
            "aggregations": [{
                "op": "count",
                "column": "*",
                "as": "above_mean",
                "where": {"column": "score", "op": ">", "value_ref": "step_1"},
            }],
        }
        count_output, table = execute_tool(
            harness,
            "group_aggregate",
            count_args,
            context,
            "step_2",
        )

        self.assertEqual(mean_output["rows"], [[20.0]])
        self.assertEqual(count_output["columns"], ["above_mean"])
        self.assertEqual(harness.rows(table), [(1,)])
        self.assertEqual(
            context["history"]["step_2"]["arguments"]["aggregations"][0]["where"],
            {"column": "score", "op": ">", "value_ref": "step_1"},
        )
        self.assertIn(
            {
                "type": "value",
                "step": "step_1",
                "role": "aggregate_where",
                "target": {"aggregation_index": 0, "column": "score"},
            },
            context["history"]["step_2"]["references"],
        )

    def test_scalar_compute_reuses_named_cells_from_one_multi_metric_row(self):
        harness = self.harness
        harness.conn.executescript(
            """
            CREATE TABLE nominees(country TEXT);
            INSERT INTO nominees VALUES
              ('USA'), ('USA'), ('USA'), ('CANADA'), ('FRANCE');
            """
        )
        harness.register_sources()
        context = new_ctx()

        aggregate_output, aggregate_table = execute_tool(
            harness,
            "group_aggregate",
            {
                "table": "nominees",
                "group_by": [],
                "aggregations": [
                    {"op": "count", "column": "*", "as": "total_nominees"},
                    {
                        "op": "count",
                        "column": "*",
                        "as": "usa_nominees",
                        "where": {"column": "country", "op": "=", "value": "USA"},
                    },
                ],
            },
            context,
            "step_1",
        )
        self.assertEqual(aggregate_output["columns"], ["total_nominees", "usa_nominees"])
        self.assertEqual(aggregate_output["row_count"], 1)
        self.assertNotIn("rows", aggregate_output)
        self.assertEqual(harness.rows(aggregate_table), [(5, 3)])

        scalar_args = {
            "operation": "percent",
            "operands": [
                {"value_ref": "step_1", "column": "usa_nominees"},
                {"value_ref": "step_1", "column": "total_nominees"},
            ],
            "result_name": "percentage",
        }
        scalar_output, _ = execute_tool(
            harness,
            "scalar_compute",
            scalar_args,
            context,
            "step_2",
        )

        self.assertEqual(scalar_output["rows"], [[60.0]])
        self.assertEqual(context["history"]["step_2"]["arguments"], scalar_args)
        self.assertEqual(
            context["history"]["step_2"]["references"],
            [
                {
                    "type": "value",
                    "step": "step_1",
                    "role": "operand",
                    "target": {"operand_index": 0, "column": "usa_nominees"},
                },
                {
                    "type": "value",
                    "step": "step_1",
                    "role": "operand",
                    "target": {"operand_index": 1, "column": "total_nominees"},
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
