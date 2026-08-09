#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from export_sft_dataset import build, convert, write_dataset_info  # noqa: E402


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
                    "tool": "group_aggregate",
                    "arguments": {
                        "table": "items",
                        "group_by": [],
                        "aggregations": [{"op": "count", "column": "*", "as": "count_1"}],
                    },
                },
                "tool_output": {"table": "group_001", "columns": ["count_1"], "rows": [[2]], "row_count": 1},
                "environment_state_before": {"plan": [], "tables": {}, "values": {}},
                "environment_state": {
                    "plan": [],
                    "tables": {
                        "group_001": {
                            "kind": "derived",
                            "created_by": "step_1",
                            "columns": ["count_1"],
                            "row_count": 1,
                            "reads": [{"from_step": "step_1", "rows": [[2]], "row_count": 1}],
                        }
                    },
                    "values": {},
                },
            },
            {
                "step_id": "step_2",
                "think": "Return the computed count.",
                "tool_call": {
                    "tool": "answer_from_context",
                    "arguments": {
                        "answer": [[2]],
                        "evidence": {"table": "group_001"},
                        "reason": "The aggregate evidence table contains the count.",
                    },
                },
                "tool_output": {"final_answer": [[2]]},
                "environment_state_before": {
                    "plan": [],
                    "tables": {
                        "group_001": {
                            "kind": "derived",
                            "created_by": "step_1",
                            "columns": ["count_1"],
                            "row_count": 1,
                            "reads": [{"from_step": "step_1", "rows": [[2]], "row_count": 1}],
                        }
                    },
                    "values": {},
                },
                "environment_state": {
                    "plan": [],
                    "tables": {
                        "group_001": {
                            "kind": "derived",
                            "created_by": "step_1",
                            "columns": ["count_1"],
                            "row_count": 1,
                            "reads": [{"from_step": "step_1", "rows": [[2]], "row_count": 1}],
                        }
                    },
                    "values": {},
                },
            },
        ],
    }


class ExportSftDatasetTests(unittest.TestCase):
    def test_convert_preserves_multiturn_roles(self):
        record = convert(trajectory())
        self.assertEqual(
            [message["from"] for message in record["conversations"]],
            ["human", "gpt", "human", "gpt"],
        )
        self.assertNotIn("<tool_call>", record["conversations"][1]["value"])
        self.assertIn(
            '{"tool":"group_aggregate","arguments":',
            record["conversations"][1]["value"],
        )
        self.assertNotIn("observation", [message["from"] for message in record["conversations"]])

    def test_convert_renders_state_only_between_assistant_turns(self):
        item = trajectory()
        item["steps"][0]["environment_state"] = {
            "plan": [{"id": "p1", "goal": "count rows", "status": "done"}],
            "tables": {"items": {"kind": "source", "row_count": 2}},
            "values": {},
        }

        record = convert(item)
        state_turn = record["conversations"][2]["value"]

        self.assertIn("CURRENT ENVIRONMENT STATE", state_turn)
        self.assertIn('"tables"', state_turn)
        self.assertNotIn('"output"', state_turn)

    def test_convert_does_not_leak_current_step_output_to_its_prompt(self):
        record = convert(trajectory())

        first_prompt = record["conversations"][0]["value"]
        second_prompt = record["conversations"][2]["value"]

        self.assertNotIn("group_001", first_prompt)
        self.assertIn("group_001", second_prompt)

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

    def test_legacy_export_drops_trajectory_with_empty_result_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            item = trajectory("empty_result")
            item["steps"][0]["tool_output"] = {
                "table": "group_001",
                "columns": ["count_1"],
                "rows": [],
                "row_count": 0,
            }
            source.write_text(json.dumps(item) + "\n", encoding="utf-8")

            output = root / "out.jsonl"
            manifest = build("train", 10000, source, output)

            self.assertEqual(manifest["kept"], 0)
            self.assertEqual(manifest["dropped_empty_result"], 1)
            self.assertEqual(
                manifest["dropped_empty_result_trajectory_ids"],
                ["empty_result"],
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "")

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

    def test_build_rejects_post_hoc_enrichment_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            item = trajectory()
            item["enrichment"] = {"quality_status": "ready"}
            source.write_text(json.dumps(item) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "post-hoc trajectory enrichment"):
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
