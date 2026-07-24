import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_external_teacher_cohort import build  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(item: int, difficulty: str) -> dict:
    return {
        "example_id": f"example_{item}",
        "db_id": f"db_{item % 2}",
        "metadata": {"difficulty_proxy": difficulty},
    }


class BuildExternalTeacherCohortTest(unittest.TestCase):
    def test_builds_exact_disjoint_complement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / "pool.jsonl"
            core_path = root / "core.jsonl"
            additional_path = root / "additional.jsonl"
            combined_path = root / "combined.jsonl"
            pool = [
                row(1, "easy"),
                row(2, "easy"),
                row(3, "medium"),
                row(4, "hard"),
            ]
            core = [pool[2], pool[0]]
            write_jsonl(pool_path, pool)
            write_jsonl(core_path, core)

            manifest = build(
                pool_path,
                core_path,
                additional_path,
                combined_path,
                root / "buckets",
            )

            additional = [
                json.loads(line)
                for line in additional_path.read_text(encoding="utf-8").splitlines()
            ]
            combined = [
                json.loads(line)
                for line in combined_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["example_2", "example_4"], [item["example_id"] for item in additional])
            self.assertEqual(
                ["example_3", "example_1", "example_2", "example_4"],
                [item["example_id"] for item in combined],
            )
            self.assertEqual(0, manifest["overlap_fixed_additional"])
            self.assertEqual(
                {"easy": 1, "hard": 1},
                manifest["additional"]["difficulty"],
            )
            self.assertFalse(manifest["difficulty_is_teacher_visible"])

    def test_rejects_core_outside_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / "pool.jsonl"
            core_path = root / "core.jsonl"
            write_jsonl(pool_path, [row(1, "easy")])
            write_jsonl(core_path, [row(2, "easy")])

            with self.assertRaisesRegex(ValueError, "not a subset"):
                build(
                    pool_path,
                    core_path,
                    root / "additional.jsonl",
                    root / "combined.jsonl",
                    root / "buckets",
                )


if __name__ == "__main__":
    unittest.main()
