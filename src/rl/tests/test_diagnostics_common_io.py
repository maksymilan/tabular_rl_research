from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from rl.diagnostics.common_io import (  # noqa: E402
    load_process_reward_config,
    percentile,
    read_jsonl,
    sha256_file,
)


class DiagnosticsCommonIoTests(unittest.TestCase):
    def test_jsonl_and_digest_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"id": 1}, {"id": 2}])
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_jsonl_reports_line_and_rejects_non_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('[1]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:1"):
                read_jsonl(path)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.0)

    def test_reward_config_loader_ignores_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            path.write_text(json.dumps({"_schema": "test", "penalty_cap": 0.7}), encoding="utf-8")
            self.assertEqual(load_process_reward_config(path).penalty_cap, 0.7)


if __name__ == "__main__":
    unittest.main()

