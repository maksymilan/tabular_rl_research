#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from build_sft_data import build, convert, write_dataset_info  # noqa: E402


def trajectory(trajectory_id: str = "traj_1") -> dict:
    return {
        "trajectory_id": trajectory_id,
        "label_status": "verified",
        "question": "How many rows are there?",
        "initial_state": {
            "dataset_overview": {
                "tables": [
                    {
                        "table_name": "items",
                        "num_rows": 2,
                        "columns": [{"name": "id", "type": "integer"}],
                    }
                ]
            }
        },
        "steps": [
            {
                "step_id": "step_1",
                "think": "Count all rows in items.",
                "tool_call": {
                    "tool": "aggregate",
                    "arguments": {"table": "items", "column": "*", "op": "count"},
                },
                "tool_output": {"result_sample": [[2]], "row_count": 1},
            },
            {
                "step_id": "step_2",
                "think": "Return the computed count.",
                "tool_call": {
                    "tool": "answer_from_context",
                    "arguments": {
                        "answer": [[2]],
                        "evidence": {"table": None},
                        "reason": "The aggregate returned two rows.",
                    },
                },
                "tool_output": {"final_answer": [[2]]},
            },
        ],
    }


class BuildSftDataTests(unittest.TestCase):
    def test_convert_preserves_multiturn_roles(self):
        record = convert(trajectory())
        self.assertEqual(
            [message["from"] for message in record["conversations"]],
            ["human", "gpt", "observation", "gpt"],
        )
        self.assertIn("<tool_call>", record["conversations"][1]["value"])

    def test_build_writes_record_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            output = root / "spider_v1_train.jsonl"
            source.write_text(json.dumps(trajectory()) + "\n", encoding="utf-8")

            manifest = build("train", 10000, source, output)

            self.assertEqual(manifest["source_trajectories"], 1)
            self.assertEqual(manifest["kept"], 1)
            self.assertEqual(manifest["dropped_overlong"], 0)
            self.assertEqual(manifest["dropped_trajectory_ids"], [])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue(output.with_suffix(".manifest.json").is_file())

    def test_build_records_overlong_trajectory_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            source.write_text(json.dumps(trajectory("too_long")) + "\n", encoding="utf-8")

            manifest = build("train", 1, source, root / "out.jsonl")

            self.assertEqual(manifest["kept"], 0)
            self.assertEqual(manifest["dropped_trajectory_ids"], ["too_long"])

    def test_build_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            row = json.dumps(trajectory())
            source.write_text(f"{row}\n{row}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate trajectory_id"):
                build("train", 10000, source, root / "out.jsonl")

    def test_build_rejects_empty_think(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            item = trajectory()
            item["steps"][0]["think"] = ""
            source.write_text(json.dumps(item) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty think"):
                build("train", 10000, source, root / "out.jsonl")

    def test_dataset_registry_preserves_existing_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            registry = out_dir / "dataset_info.json"
            registry.write_text(json.dumps({"existing": {"file_name": "old.jsonl"}}), encoding="utf-8")

            snippet, _ = write_dataset_info(out_dir, "spider_tools_v1", "spider_v1")
            merged = json.loads(registry.read_text(encoding="utf-8"))

            self.assertIn("existing", merged)
            self.assertEqual(
                merged["spider_tools_v1"]["file_name"],
                "spider_v1_train.jsonl",
            )
            self.assertTrue(snippet.is_file())


if __name__ == "__main__":
    unittest.main()
