import json
import sys
import tempfile
import unittest
from pathlib import Path


SFT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SFT_DIR.parent / "harness"
EVAL_DIR = SFT_DIR.parent / "eval"
for path in (SFT_DIR, HARNESS_DIR, EVAL_DIR):
    sys.path.insert(0, str(path))

from select_representative_atomic_teacher_cohort import (  # noqa: E402
    FORBIDDEN_TEACHER_FIELDS,
    build_outputs,
    select_cohort,
    teacher_projection,
)
from sql_atomic_tool_profile import (  # noqa: E402
    NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS,
    profile_sql,
)


def task(index: int, *, sql: str, knowledge: bool = False) -> dict:
    return {
        "example_id": f"bird_train_{index:05d}",
        "example_index": index,
        "db_id": f"db_{index % 2}",
        "question": "q" * (10 + index * 3),
        "external_knowledge": "mapping" if knowledge else None,
        "gold_sql": sql,
        "query": sql,
        "gold_exec_results": [[index]],
        "metadata": {"tool_round_trip": "verified"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SqlAtomicToolProfileTest(unittest.TestCase):
    def test_profiles_relational_families_without_emitting_actions(self):
        profile = profile_sql(
            "SELECT a.name, COUNT(*) AS n "
            "FROM a JOIN b ON a.id=b.a_id "
            "WHERE b.score > 0 GROUP BY a.name HAVING COUNT(*) > 1 "
            "ORDER BY n DESC LIMIT 3"
        )
        self.assertEqual(1, profile.counts["join_tables"])
        self.assertEqual(2, profile.counts["condition_filter"])
        self.assertEqual(1, profile.counts["group_aggregate"])
        self.assertEqual(1, profile.counts["extreme_value_select"])
        self.assertEqual(1, profile.counts["project"])
        self.assertEqual(1, profile.counts["answer_from_context"])
        payload = profile.to_json()
        self.assertFalse(payload["executable_actions"])
        self.assertNotIn("actions", payload)
        self.assertEqual(
            {"plan", "describe_table", "inspect_column", "read_subtable"},
            set(NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS),
        )

    def test_counts_scalar_and_set_operations(self):
        scalar = profile_sql("SELECT SUM(x) * 100.0 / COUNT(*) FROM t")
        self.assertEqual(1, scalar.counts["scalar_compute"])
        union = profile_sql("SELECT x FROM a UNION SELECT x FROM b")
        self.assertEqual(1, union.counts["set_op"])
        self.assertEqual(2, union.counts["project"])


class RepresentativeAtomicTeacherCohortTest(unittest.TestCase):
    def setUp(self):
        sqls = (
            "SELECT x FROM t",
            "SELECT x FROM t WHERE y=1",
            "SELECT COUNT(*) FROM t",
            "SELECT a.x FROM a JOIN b ON a.id=b.id",
            "SELECT x FROM t ORDER BY y DESC LIMIT 2",
            "SELECT SUM(x)*100.0/COUNT(*) FROM t",
            "SELECT x FROM a UNION SELECT x FROM b",
        )
        self.reference = [
            task(index, sql=sqls[index % len(sqls)], knowledge=index % 3 == 0)
            for index in range(40)
        ]
        self.eligible = list(self.reference)

    def test_selection_is_deterministic_and_tool_objective_does_not_increase(self):
        first, _profiles, first_details = select_cohort(
            self.reference,
            self.eligible,
            count=20,
            seed="unit-test",
            max_swaps=200,
        )
        second, _profiles, _details = select_cohort(
            self.reference,
            self.eligible,
            count=20,
            seed="unit-test",
            max_swaps=200,
        )
        self.assertEqual(
            [row["example_id"] for row in first],
            [row["example_id"] for row in second],
        )
        optimization = first_details["tool_balance_optimization"]
        self.assertLessEqual(
            optimization["final_objective"],
            optimization["initial_objective"],
        )
        self.assertEqual({"db_0", "db_1"}, {row["db_id"] for row in first})

    def test_teacher_projection_excludes_gold_and_private_profile(self):
        projected = teacher_projection(self.reference[0])
        self.assertFalse(set(projected) & set(FORBIDDEN_TEACHER_FIELDS))
        self.assertEqual(self.reference[0]["question"], projected["question"])

    def test_build_outputs_keeps_private_profiles_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            eligible = root / "eligible.jsonl"
            output = root / "tasks.jsonl"
            projection = root / "projection.jsonl"
            profiles = root / "profiles.jsonl"
            manifest = root / "manifest.json"
            write_jsonl(reference, self.reference)
            write_jsonl(eligible, self.eligible)

            built = build_outputs(
                reference_path=reference,
                eligible_path=eligible,
                output_path=output,
                projection_path=projection,
                profile_path=profiles,
                manifest_path=manifest,
                count=20,
                seed="unit-test-output",
                max_swaps=200,
                allow_unfiltered_historical_reproduction=True,
            )

            visible_rows = [json.loads(line) for line in projection.read_text().splitlines()]
            private_rows = [json.loads(line) for line in profiles.read_text().splitlines()]
            self.assertEqual(20, len(visible_rows))
            self.assertEqual(20, len(private_rows))
            self.assertTrue(all(not (set(row) & set(FORBIDDEN_TEACHER_FIELDS)) for row in visible_rows))
            self.assertTrue(all(row["teacher_visible"] is False for row in private_rows))
            self.assertFalse(
                built["outputs"]["teacher_visible_projection"]["forbidden_fields_present"]
            )
            self.assertEqual(
                "pending_causal_rollout_replay_quality_and_protocol_gate",
                built["training_admission"],
            )

    def test_training_selection_rejects_missing_nonempty_filter_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            eligible = root / "eligible.jsonl"
            write_jsonl(reference, self.reference)
            write_jsonl(eligible, self.eligible)
            with self.assertRaisesRegex(ValueError, "nonempty-filter-manifest"):
                build_outputs(
                    reference_path=reference,
                    eligible_path=eligible,
                    output_path=root / "tasks.jsonl",
                    projection_path=root / "projection.jsonl",
                    profile_path=root / "profiles.jsonl",
                    manifest_path=root / "manifest.json",
                    count=20,
                    seed="unit-test-missing-gate",
                    max_swaps=20,
                )


if __name__ == "__main__":
    unittest.main()
