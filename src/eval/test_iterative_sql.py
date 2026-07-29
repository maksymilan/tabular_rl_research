import sqlite3
import tempfile
import unittest
from pathlib import Path

from executor import Harness
from iterative_sql import (
    FULL_SCHEMA_SAMPLES_CONTEXT,
    LAZY_CATALOG_CONTEXT,
    build_database_context,
    build_system_prompt,
    execute_preview,
    parse_sql_action_strict,
    task_prompt,
    validate_read_only_sql,
)
from protocol import ProtocolError


class IterativeSqlProtocolTests(unittest.TestCase):
    def test_flash_prompt_contains_only_sql_interface_tools(self):
        prompt = build_system_prompt("deepseek-v4-flash")
        self.assertIn("execute_sql", prompt)
        self.assertIn("submit_sql", prompt)
        forbidden = (
            "describe_table", "inspect_column", "read_subtable", "condition_filter",
            "project", "join_tables", "group_aggregate", "extreme_value_select",
            "set_op", "answer_from_context", "add_to_memory", "refine_memory",
        )
        self.assertEqual([name for name in forbidden if name in prompt], [])
        self.assertNotIn("output exactly one non-empty <think> block", prompt)
        self.assertEqual(prompt.count("DEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT"), 1)
        self.assertIn(
            '{"tool":"execute_sql","arguments":'
            '{"sql":"PRAGMA table_info(Orders)"}}',
            prompt,
        )
        self.assertNotIn("<tool_call>", prompt)

    def test_non_split_prompt_retains_canonical_sql_envelope(self):
        prompt = build_system_prompt("gpt-5.6-sol")
        self.assertIn("output exactly one non-empty <think> block", prompt)
        self.assertNotIn("DEEPSEEK SPLIT-RESPONSE", prompt)

    def test_full_context_prompt_does_not_claim_columns_are_hidden(self):
        prompt = build_system_prompt(
            "deepseek-v4-flash",
            FULL_SCHEMA_SAMPLES_CONTEXT,
        )
        self.assertIn("complete live column schema", prompt)
        self.assertIn("example values for each column", prompt)
        self.assertNotIn("not full columns", prompt)

    def test_strict_action(self):
        text = (
            '<think>Inspect the schema.</think>\n'
            '{"tool":"execute_sql","arguments":'
            '{"sql":"PRAGMA table_info(people)"}}'
        )
        think, tool, args = parse_sql_action_strict(text)
        self.assertEqual(think, "Inspect the schema.")
        self.assertEqual(tool, "execute_sql")
        self.assertEqual(args, {"sql": "PRAGMA table_info(people)"})

    def test_rejects_multiple_calls_and_extra_arguments(self):
        call = '{"tool":"execute_sql","arguments":{"sql":"SELECT 1"}}'
        with self.assertRaises(ProtocolError):
            parse_sql_action_strict(f"<think>x</think>[{call},{call}]")
        with self.assertRaises(ProtocolError):
            parse_sql_action_strict(
                '<think>x</think>{"tool":"execute_sql","arguments":'
                '{"sql":"SELECT 1","repair":true}}'
            )
        with self.assertRaises(ProtocolError):
            parse_sql_action_strict(
                '<think>x</think><tool_call>{"tool":"execute_sql","arguments":'
                '{"sql":"SELECT 1"}}</tool_call>'
            )

    def test_read_only_validation(self):
        validate_read_only_sql("SELECT * FROM people")
        validate_read_only_sql("PRAGMA table_info(people)")
        with self.assertRaises(ValueError):
            validate_read_only_sql("DELETE FROM people")
        with self.assertRaises(ValueError):
            validate_read_only_sql("SELECT 1; SELECT 2")
        with self.assertRaises(ValueError):
            validate_read_only_sql("PRAGMA table_info(people)", terminal=True)


class IterativeSqlExecutionTests(unittest.TestCase):
    @staticmethod
    def _harness(directory: str) -> Harness:
        path = Path(directory) / "test.sqlite"
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE departments(id INTEGER PRIMARY KEY, label TEXT)"
        )
        connection.execute(
            "CREATE TABLE people("
            "id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER, "
            "FOREIGN KEY(department_id) REFERENCES departments(id))"
        )
        connection.executemany(
            "INSERT INTO departments VALUES (?, ?)",
            [(1, "engineering"), (2, "sales")],
        )
        connection.executemany(
            "INSERT INTO people VALUES (?, ?, ?)",
            [(1, "Ada", 1), (2, "Grace", 1), (3, "Linus", 2)],
        )
        connection.commit()
        connection.close()
        harness = Harness(str(path))
        harness.conn.execute("PRAGMA query_only = ON")
        return harness

    def test_full_database_context_contains_complete_schema_and_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory)
            catalog = {
                "tables": [
                    {"table_name": "departments", "num_rows": 2},
                    {"table_name": "people", "num_rows": 3},
                ],
                "relations": [
                    {"from": "people.department_id", "to": "departments.id"},
                ],
            }
            context = build_database_context(
                harness,
                catalog,
                context_profile=FULL_SCHEMA_SAMPLES_CONTEXT,
                schema_value_count=2,
            )
            people = next(
                table for table in context["tables"]
                if table["table_name"] == "people"
            )
            self.assertEqual(
                [column["name"] for column in people["columns"]],
                ["id", "name", "department_id"],
            )
            name = next(
                column for column in people["columns"]
                if column["name"] == "name"
            )
            self.assertEqual(name["example_values"], ["Ada", "Grace"])
            self.assertEqual(context["relations"], catalog["relations"])
            prompt = task_prompt(
                context,
                {"question": "Who?", "external_knowledge": None},
                context_profile=FULL_SCHEMA_SAMPLES_CONTEXT,
            )
            self.assertIn("DATABASE SCHEMA AND VALUE SAMPLES", prompt)
            self.assertIn('"example_values":["Ada","Grace"]', prompt)
            harness.conn.close()

    def test_lazy_database_context_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory)
            catalog = {
                "tables": [{"table_name": "people", "num_rows": 3}],
                "relations": [],
            }
            context = build_database_context(
                harness,
                catalog,
                context_profile=LAZY_CATALOG_CONTEXT,
                schema_value_count=2,
            )
            self.assertIs(context, catalog)
            self.assertNotIn("columns", context["tables"][0])
            harness.conn.close()

    def test_preview_is_bounded_and_reports_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE people(id INTEGER, name TEXT)")
            connection.executemany("INSERT INTO people VALUES (?, ?)", [(1, "a"), (2, "b")])
            connection.commit()
            connection.close()
            harness = Harness(str(path))
            harness.conn.execute("PRAGMA query_only = ON")
            output = execute_preview(
                harness, "SELECT id, name FROM people ORDER BY id",
                limit=1, timeout_seconds=1,
            )
            self.assertEqual(output["columns"], ["id", "name"])
            self.assertEqual(output["rows"], [[1, "a"]])
            self.assertTrue(output["rows_truncated"])


if __name__ == "__main__":
    unittest.main()
