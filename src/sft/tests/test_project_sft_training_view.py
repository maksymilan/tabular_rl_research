from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_sft_training_view import project  # noqa: E402


class ProjectSftTrainingViewTest(unittest.TestCase):
    def test_omits_heterogeneous_metadata_without_changing_training_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "canonical.jsonl"
            index = root / "canonical.index.jsonl"
            output = root / "training.jsonl"
            rows = [
                {
                    "system": "same prompt",
                    "conversations": [{"from": "gpt", "value": "first"}],
                    "metadata": {"record_id": "a", "old_field": "x"},
                },
                {
                    "system": "same prompt",
                    "conversations": [{"from": "gpt", "value": "second"}],
                    "metadata": {"record_id": "b", "new_field": {"nested": True}},
                },
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            index.write_text(
                "".join(json.dumps({"record_id": item}) + "\n" for item in ("a", "b")),
                encoding="utf-8",
            )

            manifest = project(source, index, output, "training_view")
            projected = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                projected,
                [
                    {"system": row["system"], "conversations": row["conversations"]}
                    for row in rows
                ],
            )
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(manifest["unique_record_ids"], 2)
            self.assertEqual(manifest["retained_fields"], ["system", "conversations"])
            self.assertTrue(output.with_suffix(".manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
