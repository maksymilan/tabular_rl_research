import sqlite3
import tempfile
import unittest
from pathlib import Path

from executor import Harness
from iterative_sql import (
    build_system_prompt,
    execute_preview,
    parse_sql_action_strict,
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

    def test_non_split_prompt_retains_canonical_sql_envelope(self):
        prompt = build_system_prompt("gpt-5.6-sol")
        self.assertIn("output exactly one non-empty <think> block", prompt)
        self.assertNotIn("DEEPSEEK SPLIT-RESPONSE", prompt)

    def test_strict_action(self):
        text = (
            '<think>Inspect the schema.</think>\n'
            '<tool_call>{"tool":"execute_sql","arguments":'
            '{"sql":"PRAGMA table_info(people)"}}</tool_call>'
        )
        think, tool, args = parse_sql_action_strict(text)
        self.assertEqual(think, "Inspect the schema.")
        self.assertEqual(tool, "execute_sql")
        self.assertEqual(args, {"sql": "PRAGMA table_info(people)"})

    def test_rejects_multiple_calls_and_extra_arguments(self):
        call = '<tool_call>{"tool":"execute_sql","arguments":{"sql":"SELECT 1"}}</tool_call>'
        with self.assertRaises(ProtocolError):
            parse_sql_action_strict(f"<think>x</think>{call}{call}")
        with self.assertRaises(ProtocolError):
            parse_sql_action_strict(
                '<think>x</think><tool_call>{"tool":"execute_sql","arguments":'
                '{"sql":"SELECT 1","repair":true}}</tool_call>'
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
