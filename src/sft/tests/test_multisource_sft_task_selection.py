from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


SFT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SFT_DIR.parent / "harness"
for path in (SFT_DIR, HARNESS_DIR):
    sys.path.insert(0, str(path))

from sql_task_coverage_profile import profile_sql  # noqa: E402
from select_multisource_sft_tasks import iter_json_array, read_only_nonempty, weighted_interleave  # noqa: E402


class CoverageProfileTest(unittest.TestCase):
    def test_simple_medium_hard_are_shared_ast_labels(self) -> None:
        easy = profile_sql("SELECT name FROM people WHERE id = 1")
        medium = profile_sql(
            "SELECT d.name, COUNT(*) FROM people p JOIN dept d ON p.dept_id=d.id "
            "GROUP BY d.name ORDER BY COUNT(*) DESC LIMIT 3"
        )
        hard = profile_sql("SELECT id FROM a UNION SELECT id FROM b")
        self.assertEqual(easy.difficulty, "easy")
        self.assertEqual(medium.difficulty, "medium")
        self.assertEqual(hard.difficulty, "hard")
        self.assertIn("pair:aggregate+join", medium.feature_keys)
        self.assertTrue(easy.sampling_only)
        self.assertFalse(easy.executable_actions)

    def test_literals_do_not_change_template_identity(self) -> None:
        first = profile_sql("SELECT name FROM people WHERE age > 18")
        second = profile_sql("SELECT name FROM people WHERE age > 21")
        self.assertEqual(first.literal_masked_template_sha256, second.literal_masked_template_sha256)
        self.assertNotEqual(first.canonical_sql_sha256, second.canonical_sql_sha256)

    def test_window_is_not_double_counted_as_group_aggregate(self) -> None:
        profile = profile_sql("SELECT ROW_NUMBER() OVER (ORDER BY score) AS n FROM results")
        self.assertEqual(profile.counts["windows"], 1)
        self.assertEqual(profile.counts["aggregate_metrics"], 0)
        self.assertIn("primitive:window", profile.feature_keys)
        self.assertNotIn("primitive:aggregate", profile.feature_keys)

    def test_correlated_subquery_is_explicit(self) -> None:
        profile = profile_sql(
            "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.id = a.id)"
        )
        self.assertEqual(profile.counts["correlated_subqueries"], 1)
        self.assertIn("subquery_rewrite", profile.atomic_support)


class SelectionUtilityTest(unittest.TestCase):
    def test_streaming_json_array_handles_small_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            expected = [{"text": "中文" * 100}, {"value": 2}]
            path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(iter_json_array(path, chunk_size=31)), expected)

    def test_nonempty_gate_is_read_only_and_distinguishes_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE t(value INTEGER)")
            connection.execute("INSERT INTO t VALUES (1)")
            connection.commit()
            connection.close()
            ok, reason = read_only_nonempty(
                {"db_path": str(path), "gold_sql": "SELECT * FROM t"}, timeout_seconds=1.0
            )
            self.assertEqual((ok, reason), (True, "nonempty"))
            ok, reason = read_only_nonempty(
                {"db_path": str(path), "gold_sql": "SELECT * FROM t WHERE 0"}, timeout_seconds=1.0
            )
            self.assertEqual((ok, reason), (False, "empty"))

    def test_weighted_interleave_preserves_counts(self) -> None:
        groups = {"bird": [1] * 7, "spider": [2] * 6, "synsql": [3] * 7}
        out = weighted_interleave(groups, ("bird", "spider", "synsql"))
        self.assertEqual(len(out), 20)
        self.assertEqual({value: out.count(value) for value in (1, 2, 3)}, {1: 7, 2: 6, 3: 7})


if __name__ == "__main__":
    unittest.main()
