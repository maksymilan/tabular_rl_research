#!/usr/bin/env python3
from __future__ import annotations

import collections
import sqlite3
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
sys.path[:0] = [
    str(SRC_DIR),
    str(SRC_DIR / "sft"),
    str(SRC_DIR / "eval"),
    str(SRC_DIR / "harness"),
]

from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    PLAN_POLICY_OPTIONAL,
    ResidentPlanPolicyTracker,
    execute_native_tool_bundle,
    run_rollout,
)
from rollout import new_ctx, overview  # noqa: E402
from tool_modules.native_tool_bundle.no_plan_protocol import (  # noqa: E402
    MODEL_ARG_SCHEMA as VERSION54_MODEL_ARG_SCHEMA,
    validate_model_action as validate_model_action_version54,
)


class NativeToolBundleExecutionTests(unittest.TestCase):
    @staticmethod
    def _provider_turn(calls: list[dict], reasoning: str, content=None):
        tool_calls = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["tool"],
                    "arguments": json.dumps(
                        call["arguments"], separators=(",", ":")
                    ),
                },
            }
            for call in calls
        ]
        assistant = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
        }
        return (
            json.dumps({"calls": calls}, separators=(",", ":")),
            {
                "api_finish_reason": "tool_calls",
                "provider_native_assistant_message": assistant,
                "provider_native_rejection_reason": None,
                "provider_native_bundle_calls": calls,
            },
            reasoning,
        )

    def test_later_call_cannot_consume_handle_created_in_same_bundle(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "bundle.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'alpha'), (2, 'beta');"
        )
        connection.commit()
        connection.close()
        harness = Harness(str(db_path))
        context = new_ctx(overview(harness))
        steps: list[dict] = []
        error_events: list[dict] = []
        result = execute_native_tool_bundle(
            h=harness,
            ctx=context,
            calls=[
                {
                    "id": "make_projection",
                    "tool": "project",
                    "arguments": {
                        "table": "items",
                        "expressions": ["name"],
                    },
                },
                {
                    "id": "read_unseen_projection",
                    "tool": "read_subtable",
                    "arguments": {"table": "project_001", "limit": 2},
                },
            ],
            reasoning="Project and then inspect.",
            model_turn_index=1,
            primitive_count=0,
            created=set(),
            steps=steps,
            error_events=error_events,
            error_counts=collections.Counter(),
            plan_tracker=ResidentPlanPolicyTracker(PLAN_POLICY_OPTIONAL),
            table_output_rows=0,
            database_context_profile="catalog-v1",
            ex={},
            denotation_comparison="bird-set",
            gold_sql="SELECT name FROM items",
        )
        self.assertEqual(
            [item["status"] for item in result["results"]],
            ["success", "error"],
        )
        self.assertEqual(result["results"][1]["error"]["type"], "argument_validation_error")
        self.assertEqual(result["added_errors"], 1)
        self.assertEqual(len(result["tool_messages"]), 2)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["model_turn_index"], 1)
        harness.conn.close()
        temp_dir.cleanup()

    def test_version54_hallucinated_plan_is_state_preserving_structured_error(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "no_plan.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'alpha');"
        )
        connection.commit()
        connection.close()
        harness = Harness(str(db_path))
        context = new_ctx(overview(harness))
        before = context["environment"].snapshot()
        steps: list[dict] = []
        error_events: list[dict] = []
        result = execute_native_tool_bundle(
            h=harness,
            ctx=context,
            calls=[{
                "id": "hallucinated_plan",
                "tool": "plan",
                "arguments": {
                    "ops": [{"op": "create", "id": "p1", "goal": "inspect"}],
                },
            }],
            reasoning="Create a plan.",
            model_turn_index=1,
            primitive_count=0,
            created=set(),
            steps=steps,
            error_events=error_events,
            error_counts=collections.Counter(),
            plan_tracker=ResidentPlanPolicyTracker(PLAN_POLICY_OPTIONAL),
            table_output_rows=0,
            database_context_profile="catalog-v1",
            ex={},
            denotation_comparison="bird-set",
            gold_sql="SELECT name FROM items",
            model_arg_schema=VERSION54_MODEL_ARG_SCHEMA,
            model_action_validator=validate_model_action_version54,
        )
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(result["results"][0]["error"]["code"], "unknown_tool")
        self.assertEqual(context["environment"].snapshot(), before)
        self.assertEqual(steps, [])
        self.assertEqual(len(error_events), 1)
        self.assertNotIn("plan", error_events[0]["details"]["legal_tools"])
        harness.conn.close()
        temp_dir.cleanup()

    def test_version54_rollout_initializes_no_plan_set_and_keeps_bundle_boundaries(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "episode.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'alpha'), (2, 'beta');"
        )
        connection.commit()
        connection.close()
        first_calls = [
            {
                "id": "schema",
                "tool": "describe_table",
                "arguments": {"tables": ["items"]},
            },
            {
                "id": "values",
                "tool": "inspect_column",
                "arguments": {"table": "items", "column": "name"},
            },
        ]
        project_calls = [{
            "id": "project",
            "tool": "project",
            "arguments": {"table": "items", "expressions": ["name"]},
        }]
        terminal_calls = [{
            "id": "answer",
            "tool": "answer_from_context",
            "arguments": {"evidence": {"table": "project_001"}},
        }]
        with patch(
            "generate_teacher_rollouts.chat_with_retries",
            side_effect=[
                self._provider_turn(
                    first_calls,
                    "Inspect schema and values.",
                    content="This prose is audit-only.",
                ),
                self._provider_turn(project_calls, "Project the answer column."),
                self._provider_turn(terminal_calls, "Cite the exact result."),
            ],
        ) as mocked:
            record = run_rollout(
                example_index=0,
                ex={
                    "db_id": "episode",
                    "db_path": str(db_path),
                    "question": "List the names.",
                    "gold_sql": "SELECT name FROM items",
                },
                split="dev",
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                system_prompt="system",
                max_steps=6,
                max_tokens=512,
                api_retries=1,
                api_timeout=30,
                max_errors_per_type=3,
                table_output_rows=0,
                context_mode="rolling-legal-history",
                history_turns=4,
                rolling_prompt_variant="full",
                plan_policy="optional",
                deepseek_carrier="native-tool-bundle",
                denotation_comparison="bird-set",
                diagnostic_only=True,
                database_context_profile="catalog-v1",
                atomic_protocol_version="version54",
            )
        self.assertTrue(record["correct"])
        self.assertEqual(record["tool_scheme"], "native-tool-bundle")
        self.assertEqual(record["model_turns"], 3)
        self.assertEqual(record["primitive_actions"], 4)
        self.assertEqual(len(record["trajectory"]["steps"]), 4)
        self.assertEqual(
            [step["model_turn_index"] for step in record["trajectory"]["steps"]],
            [1, 1, 2, 3],
        )
        second_input = mocked.call_args_list[1].kwargs["messages"]
        self.assertEqual(
            [message["role"] for message in second_input[-3:]],
            ["assistant", "tool", "tool"],
        )
        self.assertEqual(second_input[-3]["content"], "This prose is audit-only.")
        temp_dir.cleanup()

    def test_version53_preserves_error_feedback_before_corrected_target(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "recovery.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'alpha'), (2, 'beta');"
        )
        connection.commit()
        connection.close()
        bad_project = [{
            "id": "bad_project",
            "tool": "project",
            "arguments": {"table": "items", "expressions": ["missing_name"]},
        }]
        fixed_project = [{
            "id": "fixed_project",
            "tool": "project",
            "arguments": {"table": "items", "expressions": ["name"]},
        }]
        terminal = [{
            "id": "answer",
            "tool": "answer_from_context",
            "arguments": {"evidence": {"table": "project_001"}},
        }]
        with patch(
            "generate_teacher_rollouts.chat_with_retries",
            side_effect=[
                self._provider_turn(bad_project, "Try the requested output field."),
                self._provider_turn(fixed_project, "Use the observed valid field."),
                self._provider_turn(terminal, "Cite the corrected result."),
            ],
        ) as mocked:
            record = run_rollout(
                example_index=0,
                ex={
                    "db_id": "recovery",
                    "db_path": str(db_path),
                    "question": "List the names.",
                    "gold_sql": "SELECT name FROM items",
                },
                split="dev",
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
                system_prompt="system",
                max_steps=6,
                max_tokens=512,
                api_retries=1,
                api_timeout=30,
                max_errors_per_type=3,
                table_output_rows=0,
                context_mode="rolling-legal-history",
                history_turns=4,
                rolling_prompt_variant="full",
                plan_policy="optional",
                deepseek_carrier="native-tool-bundle",
                denotation_comparison="bird-set",
                diagnostic_only=True,
                database_context_profile="catalog-v1",
                atomic_protocol_version="version53",
            )
        self.assertTrue(record["correct"])
        self.assertEqual(record["errors"], 1)
        error_result = record["provider_native_history"][0]["tool_messages"][0]
        self.assertIn("argument_validation_error", error_result["content"])
        second_input = mocked.call_args_list[1].kwargs["messages"]
        self.assertEqual(second_input[-1]["tool_call_id"], "bad_project")
        self.assertIn("argument_validation_error", second_input[-1]["content"])
        stats = record["native_bundle_rl_statistics"]
        self.assertEqual(stats["counts"]["error_feedback_preserved"], 1)
        self.assertEqual(stats["counts"]["corrected_same_tool_next_turn"], 1)
        self.assertEqual(
            record["trajectory"]["rollout_generation"]["error_feedback_training_policy"],
            "preserve-error-turn-and-tool-feedback; later-corrected-bundle-is-target",
        )
        self.assertEqual(
            record["trajectory"]["schema_version"],
            "v7-native-tool-bundle-reviewed-prompt",
        )
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
