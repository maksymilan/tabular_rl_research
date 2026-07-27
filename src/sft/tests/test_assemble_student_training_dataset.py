#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from assemble_student_training_dataset import build  # noqa: E402
from action_carrier import (  # noqa: E402
    ACTIVE_ACTION_CARRIER,
    LEGACY_TAGGED_ACTION_CARRIER,
)
from protocol import (  # noqa: E402
    SYSTEM_PROMPT,
    parse_assistant_strict,
    student_runtime_system_prompt,
)


def action(tool: str, arguments: dict) -> str:
    return (
        f"<think>Use {tool}.</think>\n"
        f"<tool_call>{json.dumps({'tool': tool, 'arguments': arguments})}</tool_call>"
    )


def record(episode: str, step: int, tool: str, arguments: dict) -> tuple[dict, dict]:
    record_id = f"{episode}_step_{step}"
    metadata = {
        "record_id": record_id,
        "source_episode_id": episode,
        "source_step_id": f"step_{step}",
        "feedback_recovery": step == 2,
    }
    row = {
        "system": "teacher-only prompt",
        "conversations": [
            {"from": "human", "value": "CURRENT ENVIRONMENT STATE\n{}"},
            {"from": "gpt", "value": action(tool, arguments)},
        ],
        "metadata": metadata,
    }
    index = {
        "record_id": record_id,
        "source_episode_id": episode,
        "source_step_id": f"step_{step}",
        "tool_name": tool,
    }
    return row, index


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AssembleStudentTrainingDatasetTests(unittest.TestCase):
    def test_combines_complete_lanes_and_rerenders_only_prompt_and_carrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lane1 = [
                record("episode_a", 1, "describe_table", {"tables": ["t"]}),
                record(
                    "episode_a",
                    2,
                    "answer_from_context",
                    {"evidence": {"table": "project_001"}},
                ),
            ]
            lane2 = [
                record("episode_b", 1, "describe_table", {"tables": ["u"]}),
                record(
                    "episode_b",
                    2,
                    "answer_from_context",
                    {"evidence": {"table": "project_002"}},
                ),
            ]
            input_paths, index_paths = [], []
            for lane_number, lane in enumerate((lane1, lane2), start=1):
                input_path = root / f"lane{lane_number}.jsonl"
                index_path = root / f"lane{lane_number}.index.jsonl"
                write_jsonl(input_path, [item[0] for item in lane])
                write_jsonl(index_path, [item[1] for item in lane])
                input_paths.append(input_path)
                index_paths.append(index_path)

            out = root / "student.jsonl"
            index_out = root / "student.index.jsonl"
            manifest = build(
                input_paths,
                index_paths,
                out,
                index_out,
                dataset_name="student_complete",
            )
            output_rows = [
                json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
            ]
            source_rows = [item[0] for lane in (lane1, lane2) for item in lane]
            self.assertEqual(manifest["records"], 4)
            self.assertEqual(manifest["episodes"], 2)
            self.assertEqual(manifest["terminal_targets"], 2)
            self.assertEqual(manifest["feedback_recovery_targets"], 2)
            self.assertTrue(manifest["structured_actions_unchanged"])
            self.assertTrue(manifest["human_context_and_metadata_unchanged"])
            self.assertEqual(ACTIVE_ACTION_CARRIER, manifest["action_carrier"])
            self.assertEqual(
                {LEGACY_TAGGED_ACTION_CARRIER: 4},
                manifest["source_action_carriers"],
            )
            self.assertEqual(
                {row["system"] for row in output_rows},
                {student_runtime_system_prompt()},
            )
            self.assertTrue(
                all(
                    "<tool_call>" not in row["conversations"][-1]["value"]
                    for row in output_rows
                )
            )
            self.assertEqual(
                [
                    parse_assistant_strict(row["conversations"][-1]["value"])[1:]
                    for row in output_rows
                ],
                [
                    ("describe_table", {"tables": ["t"]}),
                    ("answer_from_context", {"evidence": {"table": "project_001"}}),
                    ("describe_table", {"tables": ["u"]}),
                    ("answer_from_context", {"evidence": {"table": "project_002"}}),
                ],
            )
            self.assertEqual(
                [row["metadata"] for row in output_rows],
                [row["metadata"] for row in source_rows],
            )
            self.assertEqual(SYSTEM_PROMPT in output_rows[0]["system"], True)

    def test_rejects_episode_without_final_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, index = record("episode_a", 1, "describe_table", {"tables": ["t"]})
            input_path = root / "lane.jsonl"
            index_path = root / "lane.index.jsonl"
            write_jsonl(input_path, [row])
            write_jsonl(index_path, [index])
            with self.assertRaisesRegex(ValueError, "final answer_from_context"):
                build(
                    [input_path],
                    [index_path],
                    root / "out.jsonl",
                    root / "out.index.jsonl",
                    dataset_name="student_complete",
                )

    def test_rejects_duplicate_record_ids_across_lanes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lane = [
                record(
                    "episode_a",
                    1,
                    "answer_from_context",
                    {"evidence": {"table": "project_001"}},
                )
            ]
            input_paths, index_paths = [], []
            for lane_number in (1, 2):
                input_path = root / f"lane{lane_number}.jsonl"
                index_path = root / f"lane{lane_number}.index.jsonl"
                write_jsonl(input_path, [lane[0][0]])
                write_jsonl(index_path, [lane[0][1]])
                input_paths.append(input_path)
                index_paths.append(index_path)
            with self.assertRaisesRegex(ValueError, "duplicate record_id"):
                build(
                    input_paths,
                    index_paths,
                    root / "out.jsonl",
                    root / "out.index.jsonl",
                    dataset_name="student_complete",
                )


if __name__ == "__main__":
    unittest.main()
