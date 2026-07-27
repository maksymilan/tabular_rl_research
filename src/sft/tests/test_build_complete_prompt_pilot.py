import json
import tempfile
import unittest
from pathlib import Path

from build_complete_prompt_pilot import build, episode_has_terminal, parse_quota
from action_carrier import parse_action_carrier
from protocol import STUDENT_PROMPT_FORMAL, student_runtime_system_prompt


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class CompletePromptPilotTest(unittest.TestCase):
    def test_episode_gaps_from_rejected_actions_are_legal(self):
        items = [
            ({}, {"source_step_id": "step_1", "tool_name": "describe_table"}),
            ({}, {"source_step_id": "step_3", "tool_name": "condition_filter"}),
            ({}, {"source_step_id": "step_4", "tool_name": "answer_from_context"}),
        ]
        self.assertTrue(episode_has_terminal(items))

    def test_quota_parser_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate quota"):
            parse_quota(["easy=1", "easy=2"])

    def test_builder_excludes_token_incomplete_episode_and_migrates_carrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "source.jsonl"
            index_path = root / "source.index.jsonl"
            source_manifest_path = root / "source.manifest.json"
            out_path = root / "pilot.jsonl"
            index_out_path = root / "pilot.index.jsonl"

            records = []
            indexes = []
            for episode_id, steps in {
                "episode_a": [
                    ("step_1", "describe_table", False),
                    ("step_3", "condition_filter", True),
                    ("step_4", "answer_from_context", False),
                ],
                "episode_b": [
                    ("step_1", "describe_table", False),
                    ("step_2", "answer_from_context", False),
                ],
            }.items():
                for step_id, tool, recovery in steps:
                    record_id = f"{episode_id}_{step_id}"
                    arguments = {
                        "describe_table": {"tables": ["source"]},
                        "condition_filter": {
                            "table": "source",
                            "conditions": [{"column": "id", "op": "=", "value": 1}],
                        },
                        "answer_from_context": {"evidence": {"table": "result"}},
                    }[tool]
                    target = {
                        "from": "gpt",
                        "value": (
                            f"<think>{tool}</think><tool_call>"
                            + json.dumps(
                                {"tool": tool, "arguments": arguments},
                                separators=(",", ":"),
                            )
                            + "</tool_call>"
                        ),
                    }
                    records.append(
                        {
                            "system": "old student prompt",
                            "conversations": [
                                {"from": "human", "value": f"state for {record_id}"},
                                target,
                            ],
                            "metadata": {
                                "record_id": record_id,
                                "source_episode_id": episode_id,
                                "source_step_id": step_id,
                                "feedback_recovery": recovery,
                            },
                        }
                    )
                    indexes.append(
                        {
                            "record_id": record_id,
                            "source_episode_id": episode_id,
                            "source_step_id": step_id,
                            "source_difficulty": "easy",
                            "tool_name": tool,
                        }
                    )
            write_jsonl(input_path, records)
            write_jsonl(index_path, indexes)
            source_manifest_path.write_text(
                json.dumps({"dropped_record_ids": ["episode_b_step_1"]}),
                encoding="utf-8",
            )

            manifest = build(
                input_path,
                index_path,
                source_manifest_path,
                out_path,
                index_out_path,
                dataset_name="pilot",
                quotas={"easy": 1},
                seed=42,
            )

            output = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["selected_episodes"], 1)
            self.assertEqual(manifest["records"], 3)
            self.assertEqual(manifest["terminal_targets"], 1)
            self.assertEqual(manifest["feedback_recovery_targets"], 1)
            self.assertEqual(
                {row["metadata"]["source_episode_id"] for row in output},
                {"episode_a"},
            )
            self.assertTrue(
                all(
                    row["system"]
                    == student_runtime_system_prompt(context_mode="rolling-legal-history")
                    for row in output
                )
            )
            migrated_targets = [
                row["conversations"][-1]["value"]
                for row in output
            ]
            parsed = [
                parse_action_carrier(row["conversations"][-1]["value"])
                for row in output
            ]
            self.assertEqual(
                [(think, action["tool"]) for think, action in parsed],
                [
                    ("describe_table", "describe_table"),
                    ("condition_filter", "condition_filter"),
                    ("answer_from_context", "answer_from_context"),
                ],
            )
            self.assertTrue(all("<tool_call>" not in value for value in migrated_targets))

    def test_builder_can_render_formal_student_variant_without_changing_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "source.jsonl"
            index_path = root / "source.index.jsonl"
            source_manifest_path = root / "source.manifest.json"
            out_path = root / "pilot.jsonl"
            index_out_path = root / "pilot.index.jsonl"
            target = {
                "from": "gpt",
                "value": (
                    '<think>done</think><tool_call>{"tool":"answer_from_context",'
                    '"arguments":{"evidence":{"table":"result"}}}</tool_call>'
                ),
            }
            write_jsonl(
                input_path,
                [{
                    "system": "old",
                    "conversations": [{"from": "human", "value": "state"}, target],
                    "metadata": {
                        "record_id": "episode_a_step_1",
                        "source_episode_id": "episode_a",
                        "source_step_id": "step_1",
                    },
                }],
            )
            write_jsonl(
                index_path,
                [{
                    "record_id": "episode_a_step_1",
                    "source_episode_id": "episode_a",
                    "source_step_id": "step_1",
                    "source_difficulty": "hard",
                    "tool_name": "answer_from_context",
                }],
            )
            source_manifest_path.write_text(
                json.dumps({"dropped_record_ids": []}),
                encoding="utf-8",
            )

            manifest = build(
                input_path,
                index_path,
                source_manifest_path,
                out_path,
                index_out_path,
                dataset_name="formal-pilot",
                quotas={"hard": 1},
                seed=42,
                student_prompt_variant=STUDENT_PROMPT_FORMAL,
            )

            output = json.loads(out_path.read_text(encoding="utf-8").strip())
            expected = student_runtime_system_prompt(
                context_mode="rolling-legal-history",
                student_prompt_variant=STUDENT_PROMPT_FORMAL,
            )
            self.assertEqual(output["system"], expected)
            think, action = parse_action_carrier(output["conversations"][-1]["value"])
            self.assertEqual(think, "done")
            self.assertEqual(
                action,
                {
                    "tool": "answer_from_context",
                    "arguments": {"evidence": {"table": "result"}},
                },
            )
            self.assertEqual(manifest["student_prompt_variant"], STUDENT_PROMPT_FORMAL)


if __name__ == "__main__":
    unittest.main()
