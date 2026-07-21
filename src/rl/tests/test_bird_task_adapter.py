from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from env import ToolUseEnv
from task_data import load_result_only_task_records


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
    def test_tool_env_preserves_external_knowledge_on_every_turn(self):
        task = {
            "example_index": 7,
            "db_id": "bird_db",
            "db_path": "/tmp/bird.sqlite",
            "question": "Compute the eligible rate.",
            "gold_sql": "SELECT 1",
            "external_knowledge": "eligible rate = free / enrollment",
        }
        with patch("env.Harness", FakeHarness), patch("env.overview", return_value={"tables": []}), patch(
            "env.new_ctx", return_value={"environment": FakeState()}
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
        with patch("env.Harness", FakeHarness), patch("env.overview", return_value={"tables": []}), patch(
            "env.new_ctx", return_value={"environment": FakeState()}
        ):
            environment = ToolUseEnv(task, context_mode="rolling-legal-history", history_turns=1)
            environment.legal_history.append({
                "assistant": '<think>Inspect.</think><tool_call>{"tool":"describe_table","arguments":{"tables":["T"]}}</tool_call>',
                "observation": '{"step_id":"step_1","status":"ok","output":{}}',
            })
            messages = environment.model_messages()
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertIn("describe_table", messages[2]["content"])

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
            with patch("task_data.Harness", FakeHarness), patch(
                "task_data._catalog", return_value={"tables": []}
            ):
                records = load_result_only_task_records(
                    Path(directory), split="train", examples_json=path,
                )
        self.assertEqual(records[0]["environment"]["db_path"], task["db_path"])
        self.assertEqual(records[0]["environment"]["gold_sql"], task["gold_sql"])
        prompt = records[0]["prompt"][1]["content"]
        self.assertIn(task["external_knowledge"], prompt)
        self.assertNotIn(task["gold_sql"], prompt)


if __name__ == "__main__":
    unittest.main()
