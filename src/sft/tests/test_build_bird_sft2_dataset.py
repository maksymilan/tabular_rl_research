import copy
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sft"))

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from build_bird_sft2_corrections import deterministic_diagnosis, semantic_action  # noqa: E402
from build_bird_sft2_dataset import replay_sample  # noqa: E402


class PrepareBirdSft2OnpolicyTest(unittest.TestCase):
    def test_replay_uses_recorded_denotation_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "tiny.sqlite"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE items(value TEXT)")
            connection.execute("INSERT INTO items VALUES ('x')")
            connection.commit()
            connection.close()
            trajectory = {
                "trajectory_id": "tiny_bird_set",
                "source": {
                    "db_id": "tiny",
                    "db_path": str(db_path),
                    "gold_sql": "SELECT value FROM items",
                },
                "initial_state": {"dataset_overview": None},
                "steps": [],
                "rollout_generation": {"denotation_comparison": "bird-set"},
            }
            from executor import Harness
            from rollout import new_ctx, overview

            harness = Harness(str(db_path))
            try:
                catalog = overview(harness)
                state = new_ctx(catalog)["environment"].snapshot()
            finally:
                harness.conn.close()
            trajectory["initial_state"]["dataset_overview"] = catalog
            trajectory["steps"].append({
                "step_id": "step_1",
                "tool_call": {
                    "tool": "answer_from_context",
                    "arguments": {"answer": ["x", "x"], "evidence": []},
                },
                "environment_state_before": state,
                "environment_state": state,
            })

            self.assertEqual((True, None), replay_success_trajectory(trajectory))
            self.assertEqual(
                (
                    False,
                    "replay_mismatch: denotation comparison override "
                    "'strict-multiset' conflicts with recorded 'bird-set'",
                ),
                replay_success_trajectory(
                    trajectory,
                    denotation_comparison="strict-multiset",
                ),
            )
            strict = copy.deepcopy(trajectory)
            strict["rollout_generation"]["denotation_comparison"] = "strict-multiset"
            self.assertEqual(
                (False, "replay_mismatch: final denotation"),
                replay_success_trajectory(strict),
            )

    def test_describe_table_order_is_semantically_equivalent(self):
        left = {"tool": "describe_table", "arguments": {"tables": ["B", "A"]}}
        right = {"tool": "describe_table", "arguments": {"tables": ["A", "B"]}}
        self.assertEqual(semantic_action(left), semantic_action(right))

    def test_extraneous_base_table_is_diagnosable(self):
        bad = {
            "tool_call": {"tool": "describe_table", "arguments": {"tables": ["Hours", "Users"]}},
            "tool_output": {},
        }
        teacher = [{
            "tool_call": {"tool": "describe_table", "arguments": {"tables": ["Business", "Hours"]}},
        }]
        self.assertEqual(
            "extraneous_base_table:Users",
            deterministic_diagnosis(bad, teacher, {"Business", "Hours", "Users"}),
        )

    def test_execution_error_replays_hidden_handle_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "tiny.sqlite"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE items(id INTEGER, name TEXT)")
            connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
            connection.commit()
            connection.close()
            bad_args = {
                "table": "items",
                "conditions": {"op": "=", "column": "missing", "value": 1},
            }
            good_args = {
                "table": "items",
                "conditions": {"op": "=", "column": "id", "value": 1},
            }
            sample = {
                "sample_index": 0,
                "failure_type": "max_steps",
                "turns": [
                    {
                        "turn_index": 0,
                        "parsed": {"think": "Try the requested filter.", "tool": "condition_filter", "arguments": bad_args},
                        "execution_error_type": "execution_error",
                        "execution_error": "no such column: missing",
                        "error_event": {
                            "action_index": 1,
                            "error_type": "execution_error",
                            "message": "no such column: missing",
                            "attempted_tool": "condition_filter",
                            "attempted_arguments": bad_args,
                        },
                    },
                    {
                        "turn_index": 1,
                        "parsed": {"think": "Use the existing id column.", "tool": "condition_filter", "arguments": good_args},
                        "feedback_recovery": True,
                    },
                ],
            }
            task = {
                "example_index": 1,
                "example_id": "tiny_1",
                "db_id": "tiny",
                "db_path": str(db_path),
                "gold_sql": "SELECT 1",
                "question": "Find item one.",
                "metadata": {"difficulty_proxy": "easy"},
            }
            trajectory = replay_sample(
                task, sample, require_correct=False, trajectory_id="tiny_recovery",
            )
            self.assertEqual("filter_002", trajectory["steps"][0]["tool_output"]["table"])
            self.assertTrue(trajectory["steps"][0]["feedback_recovery"])
            self.assertEqual((True, None), replay_success_trajectory(trajectory))


if __name__ == "__main__":
    unittest.main()
