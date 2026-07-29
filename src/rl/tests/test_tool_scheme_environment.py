#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from tool_environment import create_tool_use_env  # noqa: E402
from tool_schemes import ACTION_BLOCK_TOOL_SCHEME, ATOMIC_TOOL_SCHEME  # noqa: E402


class ToolSchemeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            "CREATE TABLE items(category TEXT, price INTEGER);"
            "INSERT INTO items VALUES ('a', 1), ('b', 2);"
        )
        connection.close()
        self.example = {
            "db_id": "test",
            "db_path": str(self.db_path),
            "question": "Return all categories.",
            "gold_sql": "SELECT category FROM items",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_factory_does_not_merge_visible_action_spaces(self):
        atomic = create_tool_use_env(
            self.example,
            tool_scheme=ATOMIC_TOOL_SCHEME,
        )
        block = create_tool_use_env(
            self.example,
            tool_scheme=ACTION_BLOCK_TOOL_SCHEME,
        )
        try:
            self.assertIn(
                "one typed tool action per turn",
                atomic.model_messages()[0]["content"],
            )
            self.assertNotIn(
                "Always emit action_block",
                atomic.model_messages()[0]["content"],
            )
            self.assertIn(
                "TWO TOP-LEVEL ACTIONS",
                block.model_messages()[0]["content"],
            )
            self.assertNotIn(
                "<tool_call>{",
                block.model_messages()[0]["content"],
            )
        finally:
            atomic.close()
            block.close()

    def test_action_block_scalar_cell_surface_matches_eval_runtime(self):
        env = create_tool_use_env(
            self.example,
            tool_scheme=ACTION_BLOCK_TOOL_SCHEME,
            max_steps=2,
        )
        try:
            first = (
                "<think>Project one row, then add one to its grounded cell.</think>\n"
                '{"tool":"action_block","arguments":{"calls":['
                '{"id":"one","tool":"condition_filter","arguments":'
                '{"table":"items","conditions":{"column":"price","op":"=","value":1}}},'
                '{"id":"cell","tool":"project","arguments":'
                '{"table":"$one","expressions":["price"]}},'
                '{"id":"plus","tool":"scalar_compute","arguments":'
                '{"operation":"add","operands":["$cell.price",{"value":1}],'
                '"result_name":"answer"}}]}}'
            )
            transition = env.apply_model_output(first)
            self.assertFalse(transition.done)
            self.assertIn('"status":"success"', transition.observation)
            self.assertNotIn("value_ref", transition.observation)
            scalar = transition.turn["batch_results"][-1]
            self.assertEqual(
                scalar["resolved_arguments"]["operands"],
                [
                    {"value_ref": "step_2", "column": "price"},
                    {"value": 1},
                ],
            )
            self.assertEqual(
                env.interface_resolution_events[0]["rule"],
                "same_block_one_cell",
            )
        finally:
            env.close()

    def test_action_block_student_model_can_execute_and_answer(self):
        env = create_tool_use_env(
            self.example,
            tool_scheme=ACTION_BLOCK_TOOL_SCHEME,
            max_steps=2,
        )
        try:
            first = (
                '<think>Create and inspect the exact answer table.</think>\n'
                '{"tool":"action_block","arguments":{"calls":['
                '{"id":"exact","tool":"project","arguments":'
                '{"table":"items","expressions":["category"]}},'
                '{"id":"rows","tool":"read_subtable","arguments":'
                '{"table":"$exact","limit":20}}]}}'
            )
            transition = env.apply_model_output(first)
            self.assertFalse(transition.done)
            self.assertIn("ACTION BLOCK RESULTS", transition.observation)
            self.assertEqual(env.atomic_actions, 2)

            second = (
                '<think>The resident projection has the exact answer column.</think>\n'
                '{"tool":"answer_from_context","arguments":{"evidence":'
                '{"table":"project_001"}}}'
            )
            terminal = env.apply_model_output(second)
            self.assertTrue(terminal.done)
            self.assertTrue(terminal.correct)
            record = env.record()
            self.assertEqual(record["tool_scheme"], ACTION_BLOCK_TOOL_SCHEME)
            self.assertEqual(record["model_turns"], 2)
            self.assertEqual(record["atomic_actions"], 3)
            self.assertEqual(record["action_blocks"], 2)
            self.assertEqual(record["submitted_calls"], 3)
        finally:
            env.close()

    def test_action_block_rejects_mixed_terminal_and_can_recover(self):
        env = create_tool_use_env(
            self.example,
            tool_scheme=ACTION_BLOCK_TOOL_SCHEME,
            max_steps=5,
        )
        try:
            output = (
                '<think>Derive the exact answer and terminate from its grounded result.</think>\n'
                '{"tool":"action_block","arguments":{"calls":['
                '{"id":"final","tool":"answer_from_context","arguments":{"evidence":'
                '{"table":"$exact"}}},'
                '{"id":"exact","tool":"project","arguments":'
                '{"table":"items","expressions":["category"]}}]}}'
            )
            rejected = env.apply_model_output(output)
            self.assertFalse(rejected.done)
            self.assertIn("standalone top-level terminal", rejected.observation)

            derive = (
                '<think>Derive the exact answer table first.</think>\n'
                '{"tool":"action_block","arguments":{"calls":['
                '{"id":"exact","tool":"project","arguments":'
                '{"table":"items","expressions":["category"]}}]}}'
            )
            intermediate = env.apply_model_output(derive)
            self.assertFalse(intermediate.done)

            answer = (
                '<think>Terminate from the grounded prior result.</think>\n'
                '{"tool":"answer_from_context","arguments":{"evidence":'
                '{"table":"project_001"}}}'
            )
            terminal = env.apply_model_output(answer)
            self.assertTrue(terminal.done)
            self.assertTrue(terminal.correct)
            record = env.record()
            self.assertEqual(record["model_turns"], 3)
            self.assertEqual(record["action_blocks"], 3)
            self.assertEqual(record["executed_action_blocks"], 1)
            self.assertEqual(record["atomic_actions"], 2)
            self.assertEqual(record["submitted_calls"], 2)
            self.assertEqual(record["errors"], 1)
            self.assertEqual(
                [event["tool"] for event in record["atomic_events"]],
                ["project", "answer_from_context"],
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
