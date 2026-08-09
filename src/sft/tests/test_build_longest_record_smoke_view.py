from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_longest_record_smoke_view import build  # noqa: E402


class BuildLongestRecordSmokeViewTest(unittest.TestCase):
    def test_repeats_longest_row_and_registers_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.jsonl"
            index = root / "index.jsonl"
            audit = root / "audit.json"
            output = root / "smoke.jsonl"
            rows = [
                {"system": "s", "conversations": [{"from": "gpt", "value": "a"}]},
                {"system": "s", "conversations": [{"from": "gpt", "value": "b"}]},
            ]
            training.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            index.write_text(
                "".join(json.dumps({"record_id": item}) + "\n" for item in ("a", "b")),
                encoding="utf-8",
            )
            audit.write_text(
                json.dumps(
                    {
                        "records": 2,
                        "longest_records": [{"record_id": "b", "original_tokens": 9}],
                    }
                ),
                encoding="utf-8",
            )
            manifest = build(training, index, audit, output, "smoke", 4)
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()],
                [rows[1]] * 4,
            )
            self.assertEqual(manifest["selected_record_id"], "b")
            self.assertEqual(manifest["repeats"], 4)
            self.assertTrue((root / "dataset_info.json").is_file())


if __name__ == "__main__":
    unittest.main()
