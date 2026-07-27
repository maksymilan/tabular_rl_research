import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_disjoint_rollout_cohort import build  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(item: int, sql: str, *, database: str | None = None) -> dict:
    return {
        "example_id": f"example_{item}",
        "example_index": item,
        "db_id": database or f"db_{item}",
        "gold_sql": sql,
        "metadata": {"tool_round_trip": "verified"},
    }


class BuildDisjointRolloutCohortTest(unittest.TestCase):
    def test_builds_exact_stratified_disjoint_cohort(self):
        easy = "SELECT name FROM people"
        medium = "SELECT COUNT(*) FROM a JOIN b ON a.id = b.id"
        hard = (
            "SELECT a.id FROM a JOIN b ON a.id=b.id "
            "JOIN c ON b.id=c.id JOIN d ON c.id=d.id"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            excluded = root / "excluded.jsonl"
            out = root / "cohort.jsonl"
            manifest_path = root / "manifest.json"
            rows = [
                row(1, easy),
                row(2, easy),
                row(3, medium),
                row(4, medium),
                row(5, hard),
                row(6, hard),
            ]
            write_jsonl(source, rows)
            write_jsonl(excluded, [rows[0]])

            manifest = build(
                source,
                [excluded],
                out,
                manifest_path,
                quotas={"easy": 1, "medium": 1, "hard": 1},
                seed=7,
            )

            selected = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(3, len(selected))
            self.assertNotIn("example_1", {item["example_id"] for item in selected})
            self.assertEqual(
                {"easy": 1, "hard": 1, "medium": 1},
                manifest["counts"],
            )
            self.assertEqual(0, manifest["overlap_with_exclusions"])
            self.assertFalse(manifest["difficulty_mapping"]["model_visible"])
            self.assertTrue(manifest_path.exists())

    def test_rejects_insufficient_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_jsonl(source, [row(1, "SELECT name FROM people")])
            with self.assertRaisesRegex(ValueError, "requested 1 rows"):
                build(
                    source,
                    [],
                    root / "out.jsonl",
                    root / "manifest.json",
                    quotas={"easy": 0, "medium": 1, "hard": 0},
                    seed=7,
                )


if __name__ == "__main__":
    unittest.main()
