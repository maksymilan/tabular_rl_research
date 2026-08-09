import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


SFT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SFT_DIR.parent / "harness"
EVAL_DIR = SFT_DIR.parent / "eval"
for path in (SFT_DIR, HARNESS_DIR, EVAL_DIR):
    sys.path.insert(0, str(path))

from filter_nonempty_training_tasks import (  # noqa: E402
    EMPTY_STATUS,
    NONEMPTY_STATUS,
    build_filtered_outputs,
    certify_gold_result,
)
from select_representative_atomic_teacher_cohort import build_outputs  # noqa: E402


def task(index: int, database: Path, sql: str) -> dict:
    return {
        "example_id": f"task_{index}",
        "db_id": "toy",
        "db_path": str(database),
        "question": f"question {index}",
        "gold_sql": sql,
        "query": sql,
        "metadata": {"tool_round_trip": "verified"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class NonemptyTrainingTaskFilterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "toy.sqlite"
        connection = sqlite3.connect(self.database)
        connection.executescript(
            "CREATE TABLE items(value INTEGER); INSERT INTO items VALUES (1), (2);"
        )
        connection.close()

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_rows_are_excluded_but_one_row_scalar_zero_is_retained(self):
        empty = certify_gold_result(
            task(1, self.database, "SELECT value FROM items WHERE value > 99")
        )
        scalar_zero = certify_gold_result(
            task(2, self.database, "SELECT COUNT(*) FROM items WHERE value > 99")
        )
        self.assertEqual(EMPTY_STATUS, empty.status)
        self.assertEqual(NONEMPTY_STATUS, scalar_zero.status)

    def test_outputs_preserve_source_rows_and_private_audit_has_no_gold(self):
        rows = [
            task(1, self.database, "SELECT value FROM items WHERE value = 1"),
            task(2, self.database, "SELECT value FROM items WHERE value > 99"),
            task(3, self.database, "SELECT COUNT(*) FROM items WHERE value > 99"),
        ]
        reference = self.root / "reference.jsonl"
        eligible = self.root / "eligible.jsonl"
        reference_output = self.root / "reference.nonempty.jsonl"
        eligible_output = self.root / "eligible.nonempty.jsonl"
        private_audit = self.root / "private.jsonl"
        manifest = self.root / "manifest.json"
        write_jsonl(reference, rows)
        write_jsonl(eligible, rows)

        built = build_filtered_outputs(
            reference_path=reference,
            eligible_path=eligible,
            reference_output_path=reference_output,
            eligible_output_path=eligible_output,
            private_audit_path=private_audit,
            manifest_path=manifest,
            timeout_seconds=5.0,
        )

        kept = [json.loads(line) for line in reference_output.read_text().splitlines()]
        audits = [json.loads(line) for line in private_audit.read_text().splitlines()]
        self.assertEqual(["task_1", "task_3"], [row["example_id"] for row in kept])
        self.assertEqual(rows[0], kept[0])
        self.assertEqual(1, built["status_counts"]["reference"][EMPTY_STATUS])
        self.assertTrue(built["all_acceptance_gates_passed"])
        self.assertTrue(all(record["teacher_visible"] is False for record in audits))
        rendered = json.dumps(audits)
        self.assertNotIn("gold_sql", rendered)
        self.assertNotIn("gold_rows", rendered)
        self.assertNotIn("SELECT ", rendered)

    def test_teacher_cohort_is_hash_bound_and_preserves_certified_order(self):
        rows = [
            task(1, self.database, "SELECT value FROM items WHERE value = 1"),
            task(2, self.database, "SELECT COUNT(*) FROM items WHERE value > 99"),
        ]
        reference = self.root / "reference.jsonl"
        eligible = self.root / "eligible.jsonl"
        reference_output = self.root / "reference.nonempty.jsonl"
        eligible_output = self.root / "eligible.nonempty.jsonl"
        private_audit = self.root / "private.jsonl"
        filter_manifest = self.root / "filter.manifest.json"
        write_jsonl(reference, rows)
        write_jsonl(eligible, rows)
        build_filtered_outputs(
            reference_path=reference,
            eligible_path=eligible,
            reference_output_path=reference_output,
            eligible_output_path=eligible_output,
            private_audit_path=private_audit,
            manifest_path=filter_manifest,
            timeout_seconds=5.0,
        )

        cohort = self.root / "cohort.jsonl"
        built = build_outputs(
            reference_path=reference_output,
            eligible_path=eligible_output,
            output_path=cohort,
            projection_path=self.root / "cohort.visible.jsonl",
            profile_path=self.root / "cohort.profiles.jsonl",
            manifest_path=self.root / "cohort.manifest.json",
            count=2,
            seed="unused-for-preserved-cohort",
            max_swaps=0,
            nonempty_filter_manifest_path=filter_manifest,
            preserve_cohort_path=eligible_output,
        )

        self.assertEqual(
            ["task_1", "task_2"],
            [json.loads(line)["example_id"] for line in cohort.read_text().splitlines()],
        )
        self.assertTrue(built["task_admission"]["nonempty_gold_result_required"])
        self.assertTrue(built["selection"]["preserved_cohort"]["task_ids_and_order_preserved"])
        self.assertTrue(built["all_acceptance_gates_passed"])


if __name__ == "__main__":
    unittest.main()
