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
from rollout import (  # noqa: E402
    DEFAULT_FEWSHOT_IDS,
    _normalize_table_refs,
    answer_row_candidates,
    fewshot_text,
    projected_row_candidates,
    score,
)
from text2sql import extract_sql  # noqa: E402


class FakeHarness:
    def __init__(self, gold_rows, evidence_rows=None):
        self._gold_rows = gold_rows
        self._evidence_rows = evidence_rows or {}
        self.views = set(self._evidence_rows)

    def gold(self, _sql):
        return self._gold_rows

    def rows(self, table):
        return self._evidence_rows[table]


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

    def test_answer_row_candidates_normalize_scalar_shapes(self):
        self.assertIn([[151]], answer_row_candidates([151], [[151]]))
        self.assertIn([["Village"]], answer_row_candidates("Village", [["Village"]]))
        self.assertIn([["a"], ["b"]], answer_row_candidates(["a", "b"], [["a"], ["b"]]))

    def test_score_accepts_answer_when_evidence_table_is_broader(self):
        harness = FakeHarness([[151]], {"join_001": [[151, "Alice", "dept"]]})
        ok, pred, gold = score(
            harness,
            "SELECT 151",
            {"evidence": {"table": "join_001"}, "answer": [151]},
            {"join_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [[151]])
        self.assertEqual(gold, [[151]])

    def test_score_still_accepts_exact_evidence_table(self):
        harness = FakeHarness([["Village"]], {"project_001": [["Village"]]})
        ok, pred, gold = score(
            harness,
            "SELECT 'Village'",
            {"evidence": {"table": "project_001"}, "answer": []},
            {"project_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [["Village"]])
        self.assertEqual(gold, [["Village"]])

    def test_score_rejects_broader_evidence_without_explicit_answer(self):
        harness = FakeHarness([["Aroostook"]], {"join_001": [[5, "Aroostook", "Village", 2]]})
        ok, pred, gold = score(
            harness,
            "SELECT 'Aroostook'",
            {"evidence": {"table": "join_001"}, "answer": []},
            {"join_001"},
        )
        self.assertFalse(ok)
        self.assertEqual(pred, [[5, "Aroostook", "Village", 2]])
        self.assertEqual(gold, [["Aroostook"]])

    def test_projected_row_candidates_only_permute_same_width_evidence(self):
        rows = [[1, "Town"], [4, "Village"]]
        self.assertIn([["Town", 1], ["Village", 4]], projected_row_candidates(rows, [["Town", 1], ["Village", 4]]))
        self.assertEqual(projected_row_candidates(rows, [["Town"], ["Village"]]), [])

    def test_score_accepts_evidence_columns_in_different_order(self):
        harness = FakeHarness(
            [["2017-08-03", 571], ["2017-10-21", 801]],
            {"setop_001": [[571, "2017-08-03"], [801, "2017-10-21"]]},
        )
        ok, pred, gold = score(
            harness,
            "SELECT date, id",
            {"evidence": {"table": "setop_001"}, "answer": []},
            {"setop_001"},
        )
        self.assertTrue(ok)
        self.assertEqual(pred, [["2017-08-03", 571], ["2017-10-21", 801]])
        self.assertEqual(gold, [["2017-08-03", 571], ["2017-10-21", 801]])

    def test_normalize_step_id_table_refs(self):
        ctx = {"history": {"step_3": {"output": {"table": "filter_003"}}}}
        args = {
            "table": "step_3",
            "conditions": {"column": "id", "op": "in", "in_table": "step_3"},
            "value_ref": "step_3",
        }
        self.assertEqual(
            _normalize_table_refs(args, ctx),
            {
                "table": "filter_003",
                "conditions": {"column": "id", "op": "in", "in_table": "filter_003"},
                "value_ref": "step_3",
            },
        )


if __name__ == "__main__":
    unittest.main()
