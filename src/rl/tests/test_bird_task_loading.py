from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_environment import ToolUseEnv
from build_sft_task_set import numeric_example_index
from task_loader import load_rl_task_records
from protocol import student_runtime_system_prompt


class FakeState:
    def snapshot(self):
        return {"plan": [], "tables": {}, "values": {}}


class FakeHarness:
    def __init__(self, path):
        self.path = path
        self.conn = self

    def close(self):
        pass


class BirdTaskAdapterTests(unittest.TestCase):
    def test_sft_task_set_uses_stable_bird_numeric_suffix(self):
        self.assertEqual(
            numeric_example_index(
                {
                    "trajectory_id": "bird_train_00123",
                    "source": {"example_id": "bird_train_00123"},
                }
            ),
            123,
        )

    def test_tool_env_preserves_external_knowledge_on_every_turn(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT 1",
            "external_knowledge": "eligible rate = free / enrollment",
        }
        with patch("tool_environment.Harness", FakeHarness), patch("tool_environment.overview", return_value={"tables": []}), patch(
            "tool_environment.new_ctx", return_value={"environment": FakeState()}
        ):
            environment = ToolUseEnv(task, example_index=7)
            self.assertIn("EXTERNAL KNOWLEDGE", environment.initial_messages[1]["content"])
            self.assertIn(task["external_knowledge"], environment.model_messages()[1]["content"])

    def test_tool_env_can_render_bounded_legal_history(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT 1",
        }
        with patch("tool_environment.Harness", FakeHarness), patch("tool_environment.overview", return_value={"tables": []}), patch(
            "tool_environment.new_ctx", return_value={"environment": FakeState()}
        ):
            environment = ToolUseEnv(task, context_mode="rolling-legal-history", history_turns=1)
            environment.legal_history.append({
                "assistant": '<think>Inspect.</think>{"tool":"describe_table","arguments":{"tables":["T"]}}',
                "observation": '{"step_id":"step_1","status":"ok","output":{}}',
            })
            messages = environment.model_messages()
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertIn("describe_table", messages[2]["content"])

    def test_tool_env_rejects_only_an_identical_adjacent_action_without_execution(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Inspect the relevant table.",
            "gold_sql": "SELECT 1",
        }
        first = (
            "<think>Inspect the schema.</think>"
            '{"tool":"describe_table","arguments":{"tables":["items"]}}'
        )
        repeated = (
            "<think>Use a different reason but the same call.</think>"
            '{"arguments":{"tables":["items"]},"tool":"describe_table"}'
        )
        with patch("tool_environment.Harness", FakeHarness), patch(
            "tool_environment.overview", return_value={"tables": []}
        ), patch(
            "tool_environment.new_ctx", return_value={"environment": FakeState()}
        ), patch(
            "tool_environment.execute_tool",
            return_value=({"tables": [{"table": "items"}]}, None),
        ) as execute:
            environment = ToolUseEnv(task)
            first_transition = environment.apply_model_output(first)
            repeated_transition = environment.apply_model_output(repeated)

        self.assertIsNone(first_transition.failure_type)
        self.assertFalse(repeated_transition.done)
        self.assertEqual(repeated_transition.failure_type, "no_progress_error")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(
            repeated_transition.turn["execution_error_type"],
            "no_progress_error",
        )
        feedback = json.loads(repeated_transition.observation)
        self.assertEqual(feedback["error"]["code"], "adjacent_identical_action")
        self.assertEqual(
            feedback["attempted_action"],
            {"tool": "describe_table", "arguments": {"tables": ["items"]}},
        )
        self.assertEqual(
            repeated_transition.turn["error_event"]["state_before_hash"],
            repeated_transition.turn["error_event"]["state_after_hash"],
        )

    def test_failed_answer_scoring_does_not_mark_episode_legal(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT 1",
        }
        output = (
            '<think>Use the result.</think>'
            '{"tool":"answer_from_context","arguments":'
            '{"evidence":{"table":"missing"}}}'
        )
        with patch("tool_environment.Harness", FakeHarness), patch("tool_environment.overview", return_value={"tables": []}), patch(
            "tool_environment.new_ctx", return_value={"environment": FakeState()}
        ), patch("tool_environment.score", side_effect=RuntimeError("missing evidence table")):
            environment = ToolUseEnv(task, example_index=7)
            transition = environment.apply_model_output(output)

        self.assertFalse(transition.done)
        self.assertFalse(environment.record()["legal"])
        self.assertEqual(transition.failure_type, "execution_error")

    def test_task_data_uses_adapter_paths_without_prompting_gold(self):
        task = {
            "example_index": 9,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT secret_gold",
            "external_knowledge": "eligible rate = free / enrollment",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tasks.json")
            path.write_text(json.dumps({"examples": [task]}), encoding="utf-8")
            with patch("task_loader.Harness", FakeHarness), patch(
                "task_loader.build_catalog", return_value={"tables": []}
            ):
                records = load_rl_task_records(
                    Path(directory), split="train", examples_json=path,
                )
        self.assertEqual(records[0]["environment"]["db_path"], task["db_path"])
        self.assertEqual(records[0]["environment"]["gold_sql"], task["gold_sql"])
        self.assertEqual(records[0]["environment"]["task_id"], "spider_train_00009")
        prompt = records[0]["prompt"][1]["content"]
        self.assertIn(task["external_knowledge"], prompt)
        self.assertNotIn(task["gold_sql"], prompt)
        expected_system = student_runtime_system_prompt(
            context_mode="rolling-legal-history",
            compact=False,
        )
        self.assertEqual(records[0]["prompt"][0]["content"], expected_system)
        self.assertNotIn("CANONICAL CALLS", expected_system)

    def test_tool_env_default_matches_sft_eval_rl_student_runtime_prompt(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT 1",
        }
        with patch("tool_environment.Harness", FakeHarness), patch(
            "tool_environment.overview", return_value={"tables": []}
        ), patch("tool_environment.new_ctx", return_value={"environment": FakeState()}):
            environment = ToolUseEnv(task)
        self.assertEqual(
            environment.system_prompt,
            student_runtime_system_prompt(
                context_mode="rolling-legal-history",
                compact=False,
            ),
        )

    def test_cached_catalog_does_not_leak_another_database_path(self):
        tasks = [
            {
                "example_index": 1,
                "example_id": "task_a1",
                "db_id": "a",
                "db_path": "/tmp/a.sqlite",
                "question": "a1",
                "gold_sql": "SELECT 1",
            },
            {
                "example_index": 2,
                "example_id": "task_b",
                "db_id": "b",
                "db_path": "/tmp/b.sqlite",
                "question": "b",
                "gold_sql": "SELECT 1",
            },
            {
                "example_index": 3,
                "example_id": "task_a2",
                "db_id": "a",
                "db_path": "/tmp/a.sqlite",
                "question": "a2",
                "gold_sql": "SELECT 1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tasks.json")
            path.write_text(json.dumps({"examples": tasks}), encoding="utf-8")
            with patch("task_loader.Harness", FakeHarness), patch(
                "task_loader.build_catalog", return_value={"tables": []}
            ):
                records = load_rl_task_records(
                    Path(directory),
                    split="train",
                    examples_json=path,
                )

        self.assertEqual(
            [record["environment"]["db_path"] for record in records],
            ["/tmp/a.sqlite", "/tmp/b.sqlite", "/tmp/a.sqlite"],
        )
        self.assertEqual(
            [record["environment"]["task_id"] for record in records],
            ["task_a1", "task_b", "task_a2"],
        )


if __name__ == "__main__":
    unittest.main()
