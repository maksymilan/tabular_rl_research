from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from merge_rl_task_artifacts import merge_artifacts


class MergeRlTaskArtifactsTests(unittest.TestCase):
    def test_merges_disjoint_training_examples_in_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.json")
            second = Path(directory, "second.json")
            first.write_text(json.dumps({"examples": [
                {"example_index": 3, "dataset_split": "train"},
            ]}), encoding="utf-8")
            second.write_text(json.dumps({"examples": [
                {"example_index": 8, "dataset_split": "train"},
            ]}), encoding="utf-8")
            rows, sources = merge_artifacts([first, second])
        self.assertEqual([row["example_index"] for row in rows], [3, 8])
        self.assertEqual([source["count"] for source in sources], [1, 1])
        self.assertTrue(all(len(source["sha256"]) == 64 for source in sources))

    def test_rejects_cross_artifact_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.json")
            second = Path(directory, "second.json")
            payload = {"examples": [{"example_index": 3, "dataset_split": "train"}]}
            first.write_text(json.dumps(payload), encoding="utf-8")
            second.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                merge_artifacts([first, second])


if __name__ == "__main__":
    unittest.main()
