from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_qwen3_history_view import _as_token_ids, project, project_row  # noqa: E402


def action(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, separators=(",", ":"))


def thought_action(reason: str, tool: str, **arguments) -> str:
    return f"<think>{reason}</think>\n{action(tool, **arguments)}"


class ProjectQwen3HistoryViewTest(unittest.TestCase):
    def test_token_id_normalizer_accepts_transformers_style_mapping(self):
        value = UserDict({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        self.assertEqual(_as_token_ids(value, location="test"), [1, 2, 3])

    def test_sharegpt_projection_changes_only_historical_assistant_content(self):
        row = {
            "system": "strict raw JSON protocol",
            "conversations": [
                {"from": "human", "value": "question", "kept": 1},
                {
                    "from": "gpt",
                    "value": thought_action("inspect first", "describe_table", table="orders"),
                    "provider": {"request_id": "req-1"},
                },
                {"from": "human", "value": "tool observation", "kind": "feedback"},
                {
                    "from": "gpt",
                    "value": thought_action("answer now", "answer_from_context", table="step_1"),
                    "provider": {"request_id": "req-2"},
                },
            ],
            "metadata": {
                "record_id": "record-sharegpt",
                "source": {"teacher": "external", "verified": True},
            },
            "extra_root_field": ["must", "stay"],
        }
        original = copy.deepcopy(row)

        projected, detail = project_row(row, line_number=1)

        self.assertEqual(row, original, "project_row mutated its input")
        self.assertEqual(
            projected["conversations"][1]["value"],
            action("describe_table", table="orders"),
        )
        self.assertEqual(projected["conversations"][-1], original["conversations"][-1])
        self.assertEqual(projected["conversations"][0], original["conversations"][0])
        self.assertEqual(projected["conversations"][2], original["conversations"][2])
        self.assertEqual(projected["metadata"], original["metadata"])
        self.assertEqual(projected["system"], original["system"])
        self.assertEqual(projected["extra_root_field"], original["extra_root_field"])
        self.assertEqual(projected["conversations"][1]["provider"], {"request_id": "req-1"})
        self.assertEqual(detail["historical_assistant_messages"], 1)
        self.assertEqual(detail["removed_separator_newlines"], 1)

    def test_role_content_schema_preserves_tool_and_final_target(self):
        final_target = {
            "role": "assistant",
            "content": thought_action("finish", "answer_from_context", table="step_2"),
            "loss": "target",
        }
        row = {
            "system": "system stays exact",
            "conversations": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": thought_action("read", "read_subtable", table="orders"),
                },
                {"role": "tool", "content": '{"rows":[1]}', "tool_call_id": "call-1"},
                final_target,
            ],
            "metadata": {"record_id": "record-role-content", "nested": {"x": 1}},
        }

        projected, detail = project_row(row, line_number=3)

        self.assertEqual(
            projected["conversations"][1]["content"],
            action("read_subtable", table="orders"),
        )
        self.assertEqual(projected["conversations"][2], row["conversations"][2])
        self.assertEqual(projected["conversations"][-1], final_target)
        self.assertEqual(projected["metadata"], row["metadata"])
        self.assertEqual(detail["message_schemas"], {"role-content": 4})

    def test_action_text_keeps_spaces_and_trailing_whitespace_exactly(self):
        retained = '  {"tool":"inspect_rows","arguments":{"limit":2}}  \n'
        row = {
            "system": "s",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": f"<think>r</think>\n\n{retained}"},
                {"from": "human", "value": "o"},
                {"from": "gpt", "value": thought_action("f", "answer_from_context", table="x")},
            ],
            "metadata": {"record_id": "whitespace"},
        }

        projected, detail = project_row(row, line_number=1)

        self.assertEqual(projected["conversations"][1]["value"], retained)
        self.assertEqual(detail["removed_separator_newlines"], 2)

    def test_row_without_historical_assistant_is_identical(self):
        row = {
            "system": "s",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": thought_action("r", "describe_table", table="t")},
            ],
            "metadata": {"record_id": "single-turn"},
        }

        projected, detail = project_row(row, line_number=1)

        self.assertEqual(projected, row)
        self.assertEqual(detail["historical_assistant_messages"], 0)

    def test_project_writes_metadata_preserving_view_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "canonical.jsonl"
            output = root / "derived.jsonl"
            rows = [
                {
                    "system": "s",
                    "conversations": [
                        {"from": "human", "value": "q"},
                        {"from": "gpt", "value": thought_action("h", "describe_table", table="t")},
                        {"from": "human", "value": "o"},
                        {
                            "from": "gpt",
                            "value": thought_action("f", "answer_from_context", table="t"),
                        },
                    ],
                    "metadata": {"record_id": "a", "lane": {"name": "sft1"}},
                },
                {
                    "system": "s",
                    "conversations": [
                        {"from": "human", "value": "q2"},
                        {"from": "gpt", "value": thought_action("f2", "describe_table", table="u")},
                    ],
                    "metadata": {"record_id": "b", "different": [1, 2, 3]},
                },
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            manifest = project(source, output)
            actual = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(actual[0]["metadata"], rows[0]["metadata"])
            self.assertEqual(actual[1]["metadata"], rows[1]["metadata"])
            self.assertEqual(actual[0]["conversations"][-1], rows[0]["conversations"][-1])
            self.assertEqual(actual[1], rows[1])
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(manifest["records_with_historical_assistant"], 1)
            self.assertEqual(manifest["records_without_historical_assistant"], 1)
            self.assertEqual(manifest["historical_assistant_messages_projected"], 1)
            self.assertEqual(manifest["final_targets_exactly_preserved"], 2)
            self.assertEqual(manifest["metadata_exactly_preserved"], 2)
            self.assertTrue(output.with_name(output.name + ".manifest.json").is_file())

    def test_rejects_missing_or_multiple_historical_think_blocks(self):
        base = {
            "system": "s",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": action("describe_table", table="t")},
                {"from": "human", "value": "o"},
                {"from": "gpt", "value": thought_action("f", "answer_from_context", table="t")},
            ],
            "metadata": {"record_id": "invalid-think"},
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            project_row(base, line_number=1)

        multiple = copy.deepcopy(base)
        multiple["conversations"][1]["value"] = (
            "<think>one</think><think>two</think>" + action("describe_table", table="t")
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            project_row(multiple, line_number=1)

    def test_rejects_nonleading_think_and_malformed_action(self):
        row = {
            "system": "s",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "prefix<think>x</think>\n{}"},
                {"from": "human", "value": "o"},
                {"from": "gpt", "value": thought_action("f", "answer_from_context", table="t")},
            ],
            "metadata": {"record_id": "invalid-action"},
        }
        with self.assertRaisesRegex(ValueError, "character zero"):
            project_row(row, line_number=1)

        row["conversations"][1]["value"] = "<think>x</think>\nnot-json"
        with self.assertRaisesRegex(ValueError, "not one JSON action"):
            project_row(row, line_number=1)

    def test_rejects_final_nonassistant_and_never_overwrites_source(self):
        row = {
            "system": "s",
            "conversations": [{"from": "human", "value": "q"}],
            "metadata": {"record_id": "bad-final"},
        }
        with self.assertRaisesRegex(ValueError, "assistant target"):
            project_row(row, line_number=1)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "same.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output must differ"):
                project(source, source)


if __name__ == "__main__":
    unittest.main()
