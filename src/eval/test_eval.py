#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
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
from direct_sql_prompt import (  # noqa: E402
    CANONICAL_JSON_PROFILE,
    SQL_ASTRA_APPENDIX_PROFILE,
    SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE,
    build_direct_sql_messages,
    canonical_schema_prompt,
)
from protocol import ProtocolError  # noqa: E402
from rollout import (  # noqa: E402
    ChatAPIError,
    DEFAULT_FEWSHOT_IDS,
    _normalize_table_refs,
    answer_row_candidates,
    chat,
    error_limit_reached,
    execute_tool,
    fewshot_text,
    new_ctx,
    projected_row_candidates,
    protocol_failure_type,
    score,
    validate_tool_arguments_against_state,
)
from executor import Harness  # noqa: E402
from relation_derivation import SUPPORTED_TABLE_OPERATORS  # noqa: E402
from rollout_passk import (  # noqa: E402
    ToolExecutionTimeoutError,
    append_runner_error,
    bounded_harness_execution,
    error_limit_reached as passk_error_limit_reached,
)
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
    def test_no_progress_error_uses_action_budget_instead_of_per_type_abort(self):
        self.assertIs(passk_error_limit_reached, error_limit_reached)
        self.assertFalse(error_limit_reached("no_progress_error", 3, 3))
        self.assertFalse(error_limit_reached("no_progress_error", 30, 3))
        self.assertFalse(error_limit_reached("argument_validation_error", 2, 3))
        self.assertTrue(error_limit_reached("argument_validation_error", 3, 3))
        self.assertTrue(error_limit_reached("protocol_error", 4, 3))

    def test_preexecution_validation_reports_column_on_the_wrong_table(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            'CREATE TABLE schools("School Code" TEXT, "District" TEXT);'
            'CREATE TABLE frpm("School Code" TEXT, "School Type" TEXT);'
            'INSERT INTO schools VALUES ("1", "North");'
            'INSERT INTO frpm VALUES ("1", "Public");'
        )
        harness.register_sources()
        ctx = new_ctx({"tables": [], "relations": []})

        valid, created = execute_tool(
            harness,
            "inspect_column",
            {"table": "frpm", "column": "School Type"},
            ctx,
            "step_1",
        )
        self.assertIsNone(created)
        self.assertEqual(valid["column"], "School Type")

        sequence_before = harness._n
        with self.assertRaises(ProtocolError) as raised:
            execute_tool(
                harness,
                "inspect_column",
                {"table": "schools", "column": "School Type"},
                ctx,
                "step_2",
            )
        error = raised.exception
        self.assertEqual(error.code, "unknown_column")
        self.assertEqual(error.failure_type, "argument_validation_error")
        self.assertEqual(error.attempted_tool, "inspect_column")
        self.assertEqual(error.details["requested_table"], "schools")
        self.assertEqual(error.details["requested_column"], "School Type")
        self.assertEqual(error.details["available_columns"], ["School Code", "District"])
        self.assertIn("is not a column of 'schools'", str(error))
        self.assertIn("Choose the correct table or column from the observed schemas", str(error))
        self.assertEqual(harness._n, sequence_before)
        self.assertNotIn("step_2", ctx["history"])

    def test_preexecution_validation_checks_predicate_shape_and_columns(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            "CREATE TABLE people(id INTEGER, team TEXT);"
            "INSERT INTO people VALUES (1, 'math');"
        )
        harness.register_sources()
        ctx = new_ctx({"tables": [], "relations": []})

        with self.assertRaises(ProtocolError) as missing_value:
            execute_tool(
                harness,
                "condition_filter",
                {
                    "table": "people",
                    "conditions": {"column": "team", "op": "="},
                },
                ctx,
                "step_1",
            )
        self.assertEqual(missing_value.exception.code, "invalid_condition")
        self.assertIn("value, column_value, or value_ref", str(missing_value.exception))

        with self.assertRaises(ProtocolError) as wrong_column:
            execute_tool(
                harness,
                "condition_filter",
                {
                    "table": "people",
                    "conditions": {"column": "department", "op": "=", "value": "math"},
                },
                ctx,
                "step_2",
            )
        self.assertEqual(wrong_column.exception.code, "unknown_column")
        self.assertEqual(
            wrong_column.exception.details["available_columns"],
            ["id", "team"],
        )
        self.assertEqual(harness._n, 0)

        with self.assertRaises(ProtocolError) as bad_project:
            execute_tool(
                harness,
                "project",
                {
                    "table": "people",
                    "expressions": ["id", "missing_score + 1 AS adjusted_score"],
                },
                ctx,
                "step_3",
            )
        self.assertEqual(bad_project.exception.code, "unknown_column")
        self.assertEqual(
            bad_project.exception.details["argument_path"],
            "project.expressions",
        )
        self.assertEqual(
            bad_project.exception.details["requested_column"],
            "missing_score",
        )
        self.assertIn(
            "Choose the correct table or column from the observed schemas",
            str(bad_project.exception),
        )
        self.assertEqual(harness._n, 0)

        valid_project, created = execute_tool(
            harness,
            "project",
            {"table": "people", "expressions": ["id + 1 AS next_id", "team"]},
            ctx,
            "step_4",
        )
        self.assertEqual(created, valid_project["table"])
        self.assertEqual(valid_project["columns"], ["next_id", "team"])

        with self.assertRaises(ProtocolError) as no_op_aggregate:
            execute_tool(
                harness,
                "group_aggregate",
                {"table": "people", "group_by": [], "aggregations": []},
                ctx,
                "step_5",
            )
        self.assertEqual(no_op_aggregate.exception.code, "no_op_aggregation")
        self.assertIn("no-op", str(no_op_aggregate.exception))
        self.assertEqual(harness._n, 1)

    def test_preexecution_validation_checks_terminal_evidence_handle(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.execute("CREATE TABLE people(id INTEGER)")
        harness.register_sources()

        validate_tool_arguments_against_state(
            harness,
            "answer_from_context",
            {"evidence": {"table": "people"}},
        )
        with self.assertRaises(ProtocolError) as raised:
            validate_tool_arguments_against_state(
                harness,
                "answer_from_context",
                {"evidence": {"table": "missing_result"}},
            )
        self.assertEqual(raised.exception.code, "unknown_table")
        self.assertEqual(raised.exception.failure_type, "argument_validation_error")

    def test_row_addressed_read_and_typed_date_project_use_shared_live_path(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            """
            CREATE TABLE events(id INTEGER, started_at TEXT, ended_at TEXT);
            INSERT INTO events VALUES
              (1, '2024-01-31 09:00:00', '2024-02-02 09:00:00'),
              (2, '2024-01-31 18:00:00', '2024-02-01 06:00:00'),
              (3, '2024-02-01', '2024-02-04');
            """
        )
        harness.register_sources()
        ctx = new_ctx({"tables": [], "relations": []})

        read, created = execute_tool(
            harness,
            "read_subtable",
            {
                "table": "events",
                "columns": ["id", "started_at"],
                "conditions": {
                    "column": "started_at",
                    "op": "on_date",
                    "value": "2024-01-31",
                },
                "order_by": ["id DESC"],
                "offset": 1,
                "limit": 1,
            },
            ctx,
            "step_1",
        )
        self.assertIsNone(created)
        self.assertEqual(read["rows"], [[1, "2024-01-31 09:00:00"]])
        resident_read = ctx["environment"].snapshot()["tables"]["events"]["reads"][0]
        self.assertEqual(resident_read["offset"], 1)
        self.assertEqual(resident_read["order_by"], ["id DESC"])
        self.assertEqual(resident_read["conditions"]["op"], "on_date")

        projected, created = execute_tool(
            harness,
            "project",
            {
                "table": "events",
                "expressions": [
                    "id",
                    {
                        "op": "date_diff_days",
                        "operands": [
                            {"column": "started_at"},
                            {"column": "ended_at"},
                        ],
                        "as": "duration_days",
                    },
                ],
            },
            ctx,
            "step_2",
        )
        self.assertEqual(created, projected["table"])
        self.assertEqual(
            harness.rows(created),
            [(1, 2.0), (2, 0.5), (3, 3.0)],
        )
        self.assertEqual(
            projected["derivation"]["semantics"]["column_lineage"][1]["sources"],
            ["started_at", "ended_at"],
        )

        with self.assertRaises(ProtocolError) as wrong_date_column:
            execute_tool(
                harness,
                "project",
                {
                    "table": "events",
                    "expressions": [{
                        "op": "extract_year",
                        "operands": [{"column": "missing_date"}],
                        "as": "year",
                    }],
                },
                ctx,
                "step_3",
            )
        self.assertEqual(wrong_date_column.exception.code, "unknown_column")
        self.assertEqual(
            wrong_date_column.exception.details["argument_path"],
            "project.expressions[0].operands[0].column",
        )

    def test_bounded_harness_execution_interrupts_and_restores_handles(self):
        harness = Harness()
        harness.views["infinite"] = (
            "WITH RECURSIVE counter(value) AS "
            "(VALUES(0) UNION ALL SELECT value + 1 FROM counter) "
            "SELECT value FROM counter"
        )
        views_before = dict(harness.views)
        sequence_before = harness._n

        with self.assertRaisesRegex(ToolExecutionTimeoutError, "exceeded"):
            with bounded_harness_execution(harness, 0.01):
                harness._new("project", harness.views["infinite"])

        self.assertEqual(harness.views, views_before)
        self.assertEqual(harness._n, sequence_before)
        self.assertEqual(harness.conn.execute("SELECT 1").fetchone(), (1,))

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

    def test_direct_sql_canonical_prompt_profile_is_backward_compatible(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO people VALUES (1, 'Ada');"
        )
        harness.register_sources()
        example = {
            "db_id": "test",
            "db_path": ":memory:",
            "question": "Return the name.",
            "external_knowledge": "Use people.name.",
        }
        messages = build_direct_sql_messages(
            harness,
            example,
            profile=CANONICAL_JSON_PROFILE,
        )
        self.assertEqual(
            messages[1]["content"],
            canonical_schema_prompt(
                harness,
                example["question"],
                example["external_knowledge"],
            ),
        )
        self.assertNotIn("example:", messages[1]["content"])

    def test_sql_astra_prompt_profile_renders_comments_values_and_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "school.sqlite")
            harness = Harness(str(db_path))
            self.addCleanup(harness.conn.close)
            harness.conn.executescript(
                'CREATE TABLE schools("School ID" INTEGER PRIMARY KEY, name TEXT);'
                'CREATE TABLE scores("School ID" INTEGER, score REAL, '
                'FOREIGN KEY("School ID") REFERENCES schools("School ID"));'
                "INSERT INTO schools VALUES (1, 'Ada Academy'), (2, 'Turing School');"
                "INSERT INTO scores VALUES (1, 98.5), (2, 91.0);"
            )
            metadata_path = Path(tmp, "dev_tables.json")
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "db_id": "school",
                            "table_names_original": ["schools", "scores"],
                            "table_names": ["schools", "scores"],
                            "column_names_original": [
                                [-1, "*"],
                                [0, "School ID"],
                                [0, "name"],
                                [1, "School ID"],
                                [1, "score"],
                            ],
                            "column_names": [
                                [-1, "*"],
                                [0, "school identifier"],
                                [0, "school name"],
                                [1, "school identifier"],
                                [1, "test score"],
                            ],
                            "column_types": [
                                "text",
                                "integer",
                                "text",
                                "integer",
                                "real",
                            ],
                            "primary_keys": [1],
                            "foreign_keys": [[3, 1]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            messages = build_direct_sql_messages(
                harness,
                {
                    "db_id": "school",
                    "db_path": str(db_path),
                    "question": "Which school has the highest score?",
                    "external_knowledge": "Highest means maximum score.",
                },
                profile=SQL_ASTRA_APPENDIX_PROFILE,
                schema_value_count=2,
                schema_metadata_json=str(metadata_path),
            )

        prompt = messages[1]["content"]
        self.assertIn("CREATE TABLE schools", prompt)
        self.assertIn("`School ID` integer, -- school identifier, example: [1, 2]", prompt)
        self.assertIn("name text, -- school name, example: ['Ada Academy', 'Turing School']", prompt)
        self.assertIn("PRIMARY KEY (`School ID`)", prompt)
        self.assertIn(
            "FOREIGN KEY (`School ID`) REFERENCES schools (`School ID`)",
            prompt,
        )
        self.assertIn(
            "Highest means maximum score.\nWhich school has the highest score?",
            prompt,
        )
        self.assertIn("<answer>SELECT ...</answer>", prompt)

    def test_sql_astra_disclosed_single_turn_profile_uses_paper_code_block_carrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "school.sqlite")
            harness = Harness(str(db_path))
            self.addCleanup(harness.conn.close)
            harness.conn.executescript(
                "CREATE TABLE schools(id INTEGER PRIMARY KEY, name TEXT);"
                "INSERT INTO schools VALUES (1, 'Ada Academy'), (2, 'Turing School');"
            )
            metadata_path = Path(tmp, "dev_tables.json")
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "db_id": "school",
                            "table_names_original": ["schools"],
                            "table_names": ["schools"],
                            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                            "column_names": [[-1, "*"], [0, "identifier"], [0, "school name"]],
                            "column_types": ["text", "integer", "text"],
                            "primary_keys": [1],
                            "foreign_keys": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            messages = build_direct_sql_messages(
                harness,
                {
                    "db_id": "school",
                    "db_path": str(db_path),
                    "question": "Return every school name.",
                    "external_knowledge": "Use schools.name.",
                },
                profile=SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE,
                schema_value_count=2,
                schema_metadata_json=str(metadata_path),
            )

        self.assertEqual(messages[0]["content"], "You are a helpful SQL assistant.")
        prompt = messages[1]["content"]
        self.assertIn("example: ['Ada Academy', 'Turing School']", prompt)
        self.assertIn("please enclose the generated SQL query in a code block", prompt)
        self.assertIn("```sql\n-- Your SQL query\n```", prompt)
        self.assertNotIn("<answer>", prompt)
        self.assertNotIn("run_sql_remote", prompt)

    def test_provider_transport_error_is_not_argument_validation(self):
        error = ProtocolError(
            'DeepSeek split-response transport error: visible content must contain only '
            '{"tool":"...","arguments":{...}}'
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
            extract_sql(
                "Select the required column from the schools table.\n\n"
                "```sql\nSELECT s.Zip FROM schools AS s;\n```"
            ),
            "SELECT s.Zip FROM schools AS s",
        )
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

    def test_passk_runner_error_is_audited_without_becoming_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, {"runner": "test"}, resume=False)
            try:
                raise RuntimeError("synthetic runner failure")
            except RuntimeError as exc:
                record = append_runner_error(
                    writer,
                    17,
                    {
                        "db_id": "db",
                        "question": "question",
                        "gold_sql": "must not be copied",
                    },
                    exc,
                )

            self.assertFalse(record["semantic_failure"])
            self.assertTrue(record["retry_required"])
            self.assertEqual(writer.completed, set())
            self.assertFalse(Path(tmp, "all.jsonl").exists())
            audited = json.loads(
                Path(tmp, "runner_errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(audited["example_index"], 17)
            self.assertNotIn("gold_sql", audited)
            self.assertIn("synthetic runner failure", audited["traceback"])

    def test_artifact_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ArtifactWriter(tmp, {"runner": "a"}, resume=False)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                ArtifactWriter(tmp, {"runner": "b"}, resume=True)

    def test_artifact_operational_resume_allows_only_named_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(
                tmp,
                {"runner": "test", "max_inflight_requests": 2},
                resume=False,
            )
            writer.append({"example_index": 0, "correct": True})

            resumed = ArtifactWriter(
                tmp,
                {"runner": "test", "max_inflight_requests": 8},
                resume=True,
                operational_resume_fields={"max_inflight_requests"},
                operational_resume_metadata={"workers": 8},
            )
            self.assertEqual(resumed.completed, {0})
            event = json.loads(
                Path(tmp, "operational_resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(event["completed_before_resume"], 1)
            self.assertEqual(
                event["differences"]["max_inflight_requests"],
                {"previous": 2, "requested": 8},
            )
            self.assertEqual(event["metadata"]["workers"], 8)

            with self.assertRaisesRegex(ValueError, "manifest differs"):
                ArtifactWriter(
                    tmp,
                    {
                        "runner": "different",
                        "max_inflight_requests": 8,
                    },
                    resume=True,
                    operational_resume_fields={"max_inflight_requests"},
                )

    def test_answer_row_candidates_normalize_scalar_shapes(self):
        self.assertIn([[151]], answer_row_candidates([151], [[151]]))
        self.assertIn([["Village"]], answer_row_candidates("Village", [["Village"]]))
        self.assertIn([["a"], ["b"]], answer_row_candidates(["a", "b"], [["a"], ["b"]]))

    def test_missing_explicit_answer_does_not_match_empty_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "score.sqlite")
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE items (value TEXT)")
            conn.execute("INSERT INTO items VALUES ('not-empty')")
            conn.commit()
            conn.close()

            harness = Harness(db)
            try:
                harness.views["evidence_001"] = "SELECT value FROM items"
                correct, predicted, gold = score(
                    harness,
                    "SELECT value FROM items WHERE 0",
                    {"evidence": {"table": "evidence_001"}},
                    {"evidence_001"},
                    denotation_comparison="bird-set",
                )
            finally:
                harness.conn.close()

        self.assertFalse(correct)
        self.assertEqual(predicted, [("not-empty",)])
        self.assertEqual(gold, [])

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

    def test_score_rejects_evidence_columns_in_different_order(self):
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
        self.assertFalse(ok)
        self.assertEqual(pred, [[571, "2017-08-03"], [801, "2017-10-21"]])
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
