#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from candidate_selection import (  # noqa: E402
    ARCTIC_MAJORITY_AGGREGATION,
    QueryResult,
    arctic_soft_result_similarity,
    select_arctic_majority,
)
from denotation import (  # noqa: E402
    DEFAULT_DENOTATION_COMPARISON,
    DENOTATION_COMPARISONS,
    bird_rows_equal,
    compare_denotations,
    get_denotation_metric,
    rows_equal,
)
from protocol import ProtocolError  # noqa: E402
from rollout import (  # noqa: E402
    ChatAPIError,
    DEFAULT_FEWSHOT_IDS,
    _normalize_table_refs,
    answer_row_candidates,
    chat,
    execute_tool,
    fewshot_text,
    new_ctx,
    projected_row_candidates,
    protocol_failure_type,
    score,
)
from executor import Harness  # noqa: E402
from relation_derivation import SUPPORTED_TABLE_OPERATORS  # noqa: E402
from text2sql import extract_sql  # noqa: E402
from text2sql_passk import chat_n, run_one as run_direct_sql_passk, score_sample  # noqa: E402


class FakeHarness:
    def __init__(self, gold_rows, evidence_rows=None):
        self._gold_rows = gold_rows
        self._evidence_rows = evidence_rows or {}
        self.views = set(self._evidence_rows)

    def gold(self, _sql):
        return self._gold_rows

    def rows(self, table):
        return self._evidence_rows[table]


class EvalTests(unittest.TestCase):
    def test_derivation_schema_covers_complete_live_table_action_space(self):
        from protocol import TOOLS

        non_table_tools = {
            "plan",
            "describe_table",
            "inspect_column",
            "read_subtable",
            "answer_from_context",
        }
        self.assertEqual(
            SUPPORTED_TABLE_OPERATORS - {"pivot"},
            TOOLS - non_table_tools,
        )

    def test_direct_sql_sampling_can_pin_repetition_penalty(self):
        response = unittest.mock.MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "<answer>SELECT 1</answer>"}}]}
        ).encode()
        response.__enter__.return_value = response
        with patch("text2sql_passk.urllib.request.urlopen", return_value=response) as urlopen:
            outputs = chat_n(
                "http://example",
                "test-model",
                [{"role": "user", "content": "query"}],
                n=1,
                temperature=0,
                top_p=1,
                max_tokens=32,
                retries=0,
                repetition_penalty=1.05,
            )

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(outputs, ["<answer>SELECT 1</answer>"])
        self.assertEqual(payload["repetition_penalty"], 1.05)

    def test_provider_transport_error_is_not_argument_validation(self):
        error = ProtocolError(
            'DeepSeek split-response transport error: visible content must contain only '
            '<tool_call>{"tool":"...","arguments":{...}}</tool_call>'
        )
        self.assertEqual(protocol_failure_type(error), "protocol_error")

    def test_rows_equal_normalizes_non_finite_numbers(self):
        self.assertTrue(rows_equal([[float("nan"), float("inf"), float("-inf")]],
                                   [["NaN", "+Infinity", "-Infinity"]]))
        self.assertFalse(rows_equal([[float("nan")]], [[float("inf")]]))

    def test_bird_rows_equal_ignores_duplicate_multiplicity(self):
        self.assertTrue(bird_rows_equal([["x"], ["x"]], [["x"]]))
        self.assertFalse(rows_equal([["x"], ["x"]], [["x"]]))

    def test_denotation_registry_keeps_metrics_independent(self):
        self.assertEqual(DENOTATION_COMPARISONS, ("strict-multiset", "bird-set"))
        self.assertEqual(DEFAULT_DENOTATION_COMPARISON, "bird-set")
        self.assertIs(get_denotation_metric("strict-multiset").compare, rows_equal)
        self.assertIs(get_denotation_metric("bird-set").compare, bird_rows_equal)

    def test_denotation_comparison_is_explicit(self):
        predicted = [["x"], ["x"]]
        gold = [["x"]]
        self.assertTrue(compare_denotations(predicted, gold, "bird-set"))
        self.assertFalse(compare_denotations(predicted, gold, "strict-multiset"))
        with self.assertRaises(ValueError):
            compare_denotations(predicted, gold, "unknown")

    def test_bird_set_ignores_row_order_but_not_column_order(self):
        predicted = [[1, "a"], [2, "b"]]
        self.assertTrue(compare_denotations(list(reversed(predicted)), predicted, "bird-set"))
        self.assertFalse(
            compare_denotations([["a", 1], ["b", 2]], predicted, "bird-set")
        )

    def test_arctic_soft_similarity_uses_values_within_named_columns(self):
        first = QueryResult(columns=("city",), rows=(("A",), ("B",)))
        same = QueryResult(columns=("city",), rows=(("B",), ("A",)))
        different_name = QueryResult(columns=("town",), rows=(("A",), ("B",)))
        duplicate_names = QueryResult(columns=("city", "city"), rows=(("A", "B"),))
        numeric_nulls = QueryResult(columns=("city",), rows=((None,), (1,)))
        text_nulls = QueryResult(columns=("city",), rows=((None,), ("A",)))
        self.assertEqual(arctic_soft_result_similarity(first, same), 1.0)
        self.assertEqual(arctic_soft_result_similarity(first, different_name), 0.0)
        self.assertEqual(arctic_soft_result_similarity(first, None), 0.0)
        self.assertEqual(
            arctic_soft_result_similarity(duplicate_names, duplicate_names), 0.0
        )
        self.assertAlmostEqual(
            arctic_soft_result_similarity(numeric_nulls, numeric_nulls), 1 / 3
        )
        self.assertEqual(arctic_soft_result_similarity(text_nulls, text_nulls), 1.0)

    def test_arctic_majority_selects_soft_denotation_medoid_and_first_tie(self):
        first = QueryResult(columns=("value",), rows=((1,), (2,)))
        equivalent = QueryResult(columns=("value",), rows=((2,), (1,)))
        outlier = QueryResult(columns=("value",), rows=((9,),))
        selection = select_arctic_majority([first, equivalent, outlier])
        self.assertEqual(selection.selected_index, 0)
        self.assertGreater(selection.scores[0], selection.scores[2])
        self.assertEqual(select_arctic_majority([None, None]).selected_index, 0)

    def test_direct_sql_sample_keeps_full_named_result_for_arctic_selection(self):
        from executor import Harness

        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            "CREATE TABLE values_table(value INTEGER);"
            "INSERT INTO values_table VALUES (1), (2);"
        )
        sample = score_sample(
            harness,
            "<answer>SELECT value AS selected_value FROM values_table</answer>",
            [(1,), (2,)],
            0,
            "bird-set",
        )
        self.assertTrue(sample["correct"])
        self.assertEqual(
            sample["_query_result"],
            QueryResult(columns=("selected_value",), rows=((1,), (2,))),
        )

    def test_direct_sql_arctic_majority_is_selected_before_artifact_serialization(self):
        from executor import Harness

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "test.sqlite")
            seed = Harness(str(db_path))
            seed.conn.executescript(
                "CREATE TABLE values_table(value INTEGER);"
                "INSERT INTO values_table VALUES (1), (2);"
            )
            seed.conn.close()
            outputs = [
                "<answer>SELECT value FROM values_table</answer>",
                "<answer>SELECT value FROM values_table ORDER BY value DESC</answer>",
                "<answer>SELECT 9 AS value</answer>",
            ]
            with patch("text2sql_passk.chat_n", return_value=outputs):
                record = run_direct_sql_passk(
                    {
                        "index": 0,
                        "db_id": "test",
                        "db_path": str(db_path),
                        "question": "Return the values.",
                        "gold_sql": "SELECT value FROM values_table",
                    },
                    0,
                    "http://unused",
                    "test-model",
                    n_samples=3,
                    pass_k=(1, 3),
                    temperature=0.8,
                    top_p=1.0,
                    max_tokens=32,
                    api_retries=0,
                    execution_timeout_seconds=10,
                    denotation_comparison="bird-set",
                    candidate_aggregation=ARCTIC_MAJORITY_AGGREGATION,
                )
        self.assertEqual(record["selected_sample_index"], 0)
        self.assertTrue(record["correct"])
        self.assertEqual(record["selected_predicted_sql"], "SELECT value FROM values_table")
        self.assertTrue(all("_query_result" not in sample for sample in record["samples"]))

    def test_context_overflow_retry_can_shrink_to_128_tokens(self):
        budgets = []

        def fake_chat_once(_base_url, _model, _messages, max_tokens):
            budgets.append(max_tokens)
            if max_tokens > 128:
                raise ChatAPIError("maximum context length exceeded", status=400)
            return "ok"

        with patch("rollout._chat_once", side_effect=fake_chat_once):
            result = chat("http://example", "model", [], max_tokens=1024)

        self.assertEqual(result, "ok")
        self.assertEqual(budgets, [1024, 512, 256, 128])

    def test_extract_sql(self):
        self.assertEqual(extract_sql("```sql\nSELECT * FROM t;\n```"), "SELECT * FROM t")
        self.assertEqual(
            extract_sql("Here is the query: WITH x AS (SELECT 1) SELECT * FROM x; trailing"),
            "WITH x AS (SELECT 1) SELECT * FROM x",
        )
        self.assertIsNone(extract_sql("I cannot answer this."))

    def test_fixed_fewshot_rendering(self):
        rendered = fewshot_text(DEFAULT_FEWSHOT_IDS)
        self.assertIn("How many heads of the departments are older than 56", rendered)
        self.assertIn("List the name, born state and age", rendered)

    def test_artifacts_split_and_resume(self):
        manifest = {"runner": "test", "model": "model"}
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, manifest, resume=False)
            writer.append({"example_index": 0, "correct": True})
            writer.append({"example_index": 1, "correct": False})
            summary = writer.summarize()
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(len(Path(tmp, "success.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len(Path(tmp, "failure.jsonl").read_text().splitlines()), 1)
            success_case = json.loads(Path(tmp, "success_cases", "q0000.json").read_text())
            failure_case = json.loads(Path(tmp, "failure_cases", "q0001.json").read_text())
            self.assertTrue(success_case["correct"])
            self.assertFalse(failure_case["correct"])

            resumed = ArtifactWriter(tmp, manifest, resume=True)
            self.assertEqual(resumed.completed, {0, 1})

    def test_artifact_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ArtifactWriter(tmp, {"runner": "a"}, resume=False)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                ArtifactWriter(tmp, {"runner": "b"}, resume=True)

    def test_answer_row_candidates_normalize_scalar_shapes(self):
        self.assertIn([[151]], answer_row_candidates([151], [[151]]))
        self.assertIn([["Village"]], answer_row_candidates("Village", [["Village"]]))
        self.assertIn([["a"], ["b"]], answer_row_candidates(["a", "b"], [["a"], ["b"]]))

    def test_score_accepts_answer_when_evidence_table_is_broader(self):
        harness = FakeHarness([[151]], {"join_001": [[151, "Alice", "dept"]]})
        ok, pred, gold = score(
            harness,
            "SELECT 151",
            {"evidence": {"table": "join_001"}, "answer": [151]},
            {"join_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [[151]])
        self.assertEqual(gold, [[151]])

    def test_score_still_accepts_exact_evidence_table(self):
        harness = FakeHarness([["Village"]], {"project_001": [["Village"]]})
        ok, pred, gold = score(
            harness,
            "SELECT 'Village'",
            {"evidence": {"table": "project_001"}, "answer": []},
            {"project_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [["Village"]])
        self.assertEqual(gold, [["Village"]])

    def test_score_defaults_to_bird_set_and_keeps_explicit_historical_comparator(self):
        harness = FakeHarness([["Village"]], {"project_001": [["Village"], ["Village"]]})
        args = {"evidence": {"table": "project_001"}, "answer": []}
        self.assertTrue(score(harness, "SELECT 'Village'", args, {"project_001"})[0])
        self.assertFalse(
            score(harness, "SELECT 'Village'", args, {"project_001"}, "strict-multiset")[0]
        )

    def test_score_rejects_broader_evidence_without_explicit_answer(self):
        harness = FakeHarness([["Aroostook"]], {"join_001": [[5, "Aroostook", "Village", 2]]})
        ok, pred, gold = score(
            harness,
            "SELECT 'Aroostook'",
            {"evidence": {"table": "join_001"}, "answer": []},
            {"join_001"},
        )
        self.assertFalse(ok)
        self.assertEqual(pred, [[5, "Aroostook", "Village", 2]])
        self.assertEqual(gold, [["Aroostook"]])

    def test_projected_row_candidates_only_permute_same_width_evidence(self):
        rows = [[1, "Town"], [4, "Village"]]
        self.assertIn([["Town", 1], ["Village", 4]], projected_row_candidates(rows, [["Town", 1], ["Village", 4]]))
        self.assertEqual(projected_row_candidates(rows, [["Town"], ["Village"]]), [])

    def test_score_accepts_evidence_columns_in_different_order(self):
        harness = FakeHarness(
            [["2017-08-03", 571], ["2017-10-21", 801]],
            {"setop_001": [[571, "2017-08-03"], [801, "2017-10-21"]]},
        )
        ok, pred, gold = score(
            harness,
            "SELECT date, id",
            {"evidence": {"table": "setop_001"}, "answer": []},
            {"setop_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [["2017-08-03", 571], ["2017-10-21", 801]])
        self.assertEqual(gold, [["2017-08-03", 571], ["2017-10-21", 801]])

    def test_normalize_step_id_table_refs(self):
        ctx = {"history": {"step_3": {"output": {"table": "filter_003"}}}}
        args = {
            "table": "step_3",
            "conditions": {"column": "id", "op": "in", "in_table": "step_3"},
            "value_ref": "step_3",
        }
        self.assertEqual(
            _normalize_table_refs(args, ctx),
            {
                "table": "filter_003",
                "conditions": {"column": "id", "op": "in", "in_table": "filter_003"},
                "value_ref": "step_3",
            },
        )

    def test_scalar_compute_resolves_grounded_value_refs_and_records_edges(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        ctx = new_ctx({"tables": [], "relations": []})
        ctx["history"] = {
            "step_1": {
                "tool": "group_aggregate",
                "arguments": {},
                "output": {
                    "table": "group_001",
                    "kind": "group",
                    "columns": ["part"],
                    "row_count": 1,
                    "rows": [[25]],
                },
                "references": [],
            },
            "step_2": {
                "tool": "group_aggregate",
                "arguments": {},
                "output": {
                    "table": "group_002",
                    "kind": "group",
                    "columns": ["whole"],
                    "row_count": 1,
                    "rows": [[100]],
                },
                "references": [],
            },
        }
        output, created = execute_tool(
            harness,
            "scalar_compute",
            {
                "operation": "percent",
                "operands": [
                    {"value_ref": "step_1"},
                    {"value_ref": "step_2"},
                ],
                "result_name": "percentage",
            },
            ctx,
            "step_3",
        )
        self.assertEqual(output["rows"], [[25.0]])
        self.assertEqual(created, output["table"])
        self.assertEqual(
            ctx["history"]["step_3"]["references"],
            [
                {
                    "type": "value",
                    "step": "step_1",
                    "role": "operand",
                    "target": {"operand_index": 0},
                },
                {
                    "type": "value",
                    "step": "step_2",
                    "role": "operand",
                    "target": {"operand_index": 1},
                },
            ],
        )

    def test_fact_only_derivation_covers_every_live_table_operator(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            "CREATE TABLE people(id INTEGER, first_name TEXT, last_name TEXT, team TEXT);"
            "INSERT INTO people VALUES "
            "(1, 'Ada', 'Lovelace', 'math'),"
            "(2, 'Grace', 'Hopper', 'navy');"
            "CREATE TABLE badges(person_id INTEGER, badge TEXT);"
            "INSERT INTO badges VALUES (1, 'founder');"
        )
        harness.register_sources()
        ctx = new_ctx({
            "tables": [
                {"table_name": "people", "num_rows": 2},
                {"table_name": "badges", "num_rows": 1},
            ],
            "relations": [],
        })

        project_args = {
            "table": "people",
            "expressions": ["first_name || ' ' || last_name AS full_name"],
        }
        projected, _ = execute_tool(
            harness, "project", project_args, ctx, "step_1"
        )
        self.assertEqual(project_args, {
            "table": "people",
            "expressions": ["first_name || ' ' || last_name AS full_name"],
        })
        self.assertEqual(
            projected["derivation"]["semantics"]["column_lineage"],
            [{
                "output": "full_name",
                "sources": ["first_name", "last_name"],
                "kind": "expression",
                "expression": "first_name || ' ' || last_name AS full_name",
            }],
        )

        filtered, _ = execute_tool(
            harness,
            "condition_filter",
            {
                "table": "people",
                "conditions": {"column": "team", "op": "=", "value": "math"},
            },
            ctx,
            "step_filter",
        )
        self.assertEqual(
            filtered["derivation"]["semantics"],
            {
                "row_operation": "filter",
                "predicate": {"column": "team", "op": "=", "value": "math"},
                "predicate_columns": ["team"],
                "column_operation": "preserve",
                "projected_columns": ["id", "first_name", "last_name", "team"],
            },
        )

        grouped, _ = execute_tool(
            harness,
            "group_aggregate",
            {
                "table": "people",
                "group_by": ["team"],
                "aggregations": [
                    {"op": "count", "column": "*", "as": "people_count"},
                ],
            },
            ctx,
            "step_2",
        )
        self.assertEqual(grouped["row_count"], 2)
        group_semantics = grouped["derivation"]["semantics"]
        self.assertEqual(group_semantics["row_operation"], "aggregate")
        self.assertEqual(group_semantics["row_grain"], ["team"])
        self.assertEqual(group_semantics["layout"], "rows")
        self.assertEqual(
            group_semantics["aggregations"],
            [{
                "output": "people_count",
                "op": "count",
                "source": "*",
            }],
        )

        joined, _ = execute_tool(
            harness,
            "join_tables",
            {
                "base": "people",
                "joins": [{
                    "table": "badges",
                    "type": "left",
                    "on": [{"left": "people.id", "right": "person_id"}],
                }],
            },
            ctx,
            "step_3",
        )
        self.assertEqual(joined["row_count"], 2)
        self.assertEqual(
            joined["derivation"]["semantics"]["edges"][0]["null_extended_output_rows"],
            1,
        )
        self.assertEqual(
            joined["derivation"]["semantics"]["edges"][0]["namespace"],
            "badges",
        )

        scalar, _ = execute_tool(
            harness,
            "scalar_compute",
            {
                "operation": "add",
                "operands": [{"value": 2}, {"value": 3}],
                "result_name": "total",
            },
            ctx,
            "step_scalar",
        )
        self.assertEqual(
            scalar["derivation"]["semantics"],
            {
                "row_operation": "scalar",
                "column_operation": "create",
                "operation": "add",
                "result_column": "total",
            },
        )
        grounded_filter, _ = execute_tool(
            harness,
            "condition_filter",
            {
                "table": "people",
                "conditions": {"column": "id", "op": ">=", "value_ref": "step_scalar"},
            },
            ctx,
            "step_grounded_filter",
        )
        self.assertEqual(
            grounded_filter["derivation"]["inputs"],
            [
                {"kind": "table", "role": "input", "ref": "people"},
                {"kind": "value", "role": "predicate_value", "ref": "step_scalar"},
            ],
        )
        self.assertTrue(
            any(
                reference.get("type") == "value"
                and reference.get("step") == "step_scalar"
                and reference.get("role") == "value_ref"
                for reference in ctx["history"]["step_grounded_filter"]["references"]
            )
        )

        top, _ = execute_tool(
            harness,
            "extreme_value_select",
            {
                "table": "people",
                "order_by": ["id DESC"],
                "top_k": 1,
                "return_columns": ["first_name"],
            },
            ctx,
            "step_top",
        )
        self.assertEqual(
            top["derivation"]["semantics"],
            {
                "row_operation": "ordered_prefix",
                "order_by": ["id DESC"],
                "top_k": 1,
                "column_operation": "project",
                "projected_columns": ["first_name"],
            },
        )

        people_names, _ = execute_tool(
            harness,
            "project",
            {"table": "people", "expressions": ["first_name AS value"]},
            ctx,
            "step_people_names",
        )
        badge_names, _ = execute_tool(
            harness,
            "project",
            {"table": "badges", "expressions": ["badge AS value"]},
            ctx,
            "step_badge_names",
        )
        combined, _ = execute_tool(
            harness,
            "set_op",
            {
                "left": people_names["table"],
                "right": badge_names["table"],
                "op": "union_all",
            },
            ctx,
            "step_set",
        )
        self.assertEqual(
            combined["derivation"]["semantics"],
            {
                "row_operation": "set",
                "column_operation": "align_by_position",
                "operation": "union_all",
                "duplicate_semantics": "preserve",
            },
        )
        self.assertNotIn("advice", json.dumps(ctx["environment"].snapshot()))
        self.assertNotIn("latest_structural_feedback", ctx["environment"].snapshot())


if __name__ == "__main__":
    unittest.main()
