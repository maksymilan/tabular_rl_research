#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import ArtifactWriter  # noqa: E402
from rollout import DEFAULT_FEWSHOT_IDS, fewshot_text  # noqa: E402
from text2sql import extract_sql  # noqa: E402


class EvalTests(unittest.TestCase):
    def test_extract_sql(self):
        self.assertEqual(extract_sql("```sql\nSELECT * FROM t;\n```"), "SELECT * FROM t")
        self.assertEqual(
            extract_sql("Here is the query: WITH x AS (SELECT 1) SELECT * FROM x; trailing"),
            "WITH x AS (SELECT 1) SELECT * FROM x",
        )
        self.assertIsNone(extract_sql("I cannot answer this."))

    def test_fixed_fewshot_rendering(self):
        rendered = fewshot_text(DEFAULT_FEWSHOT_IDS)
        self.assertIn("How many heads of the departments are older than 56", rendered)
        self.assertIn("List the name, born state and age", rendered)

    def test_artifacts_split_and_resume(self):
        manifest = {"runner": "test", "model": "model"}
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, manifest, resume=False)
            writer.append({"example_index": 0, "correct": True})
            writer.append({"example_index": 1, "correct": False})
            summary = writer.summarize()
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(len(Path(tmp, "success.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len(Path(tmp, "failure.jsonl").read_text().splitlines()), 1)
            success_case = json.loads(Path(tmp, "success_cases", "q0000.json").read_text())
            failure_case = json.loads(Path(tmp, "failure_cases", "q0001.json").read_text())
            self.assertTrue(success_case["correct"])
            self.assertFalse(failure_case["correct"])

            resumed = ArtifactWriter(tmp, manifest, resume=True)
            self.assertEqual(resumed.completed, {0, 1})

    def test_artifact_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ArtifactWriter(tmp, {"runner": "a"}, resume=False)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                ArtifactWriter(tmp, {"runner": "b"}, resume=True)


if __name__ == "__main__":
    unittest.main()
