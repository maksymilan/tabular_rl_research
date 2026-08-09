from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

try:
    from . import score_bird_dev as scorer
except ImportError:  # Direct ``python test_score_bird_dev.py`` execution.
    import score_bird_dev as scorer


def _assistant_result(rollout_idx: int, answer: str) -> dict:
    return {
        "instance_id": "unused",
        "rollout_idx": rollout_idx,
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>SELECT 999</answer>"},
            {"role": "user", "content": "later feedback"},
            {"role": "assistant", "content": answer},
        ],
        "final_messages": [],
        "terminated": True,
    }


class ScoreBirdDevTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "official_outputs"
        self.databases = self.root / "dev_databases"
        self.outputs.mkdir()
        self.databases.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _database(self, db_id: str, statements: list[str]) -> Path:
        directory = self.databases / db_id
        directory.mkdir()
        path = directory / f"{db_id}.sqlite"
        connection = sqlite3.connect(path)
        try:
            for statement in statements:
                connection.execute(statement)
            connection.commit()
        finally:
            connection.close()
        return path

    def _dev_json(self, examples: list[dict]) -> Path:
        path = self.root / "dev.json"
        path.write_text(json.dumps(examples), encoding="utf-8")
        return path

    def _input_manifest(self, dev_json: Path, source_indices: list[int]) -> Path:
        records = json.loads(dev_json.read_text(encoding="utf-8"))
        path = self.root / "prepared-input.manifest.json"
        path.write_text(
            json.dumps(
                {
                    "source": {
                        "record_count": len(records),
                        "bird_dev_sha256": scorer.sha256_file(dev_json),
                    },
                    "selection": {"source_indices": source_indices},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_extracts_fenced_or_raw_sql_from_last_assistant_answer(self) -> None:
        fenced = _assistant_result(
            0,
            "<think>done</think><answer>```sql\nSELECT name FROM people;\n```</answer>",
        )
        parsed = scorer.parse_last_assistant_answer(fenced)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.sql, "SELECT name FROM people;")
        self.assertEqual(parsed.message_source, "conversation")

        released_fence = _assistant_result(
            0,
            "<answer>'''SQLite\nSELECT name FROM people\n'''</answer>",
        )
        released_parsed = scorer.parse_last_assistant_answer(released_fence)
        self.assertEqual(released_parsed.status, "ok")
        self.assertEqual(released_parsed.sql, "SELECT name FROM people")

        raw = _assistant_result(0, "<answer>WITH x AS (SELECT 1) SELECT * FROM x</answer>")
        self.assertEqual(
            scorer.parse_last_assistant_answer(raw).sql,
            "WITH x AS (SELECT 1) SELECT * FROM x",
        )

        missing = _assistant_result(0, "<action>confirm_answer</action>")
        missing["conversation"][1]["content"] = "<action>explore_schema</action>"
        self.assertEqual(
            scorer.parse_last_assistant_answer(missing).status,
            "missing_assistant_answer",
        )

    def test_bird_set_ignores_rows_and_duplicates_but_not_columns(self) -> None:
        self.assertTrue(scorer.bird_set_equal([(1, 2), (1, 2)], [(1, 2)]))
        self.assertTrue(scorer.bird_set_equal([(2,), (1,)], [(1,), (2,)]))
        self.assertFalse(scorer.bird_set_equal([(2, 1)], [(1, 2)]))

    def test_cli_accepts_launcher_aliases(self) -> None:
        args = scorer.parse_args(
            [
                "--results",
                str(self.outputs),
                "--bird-json",
                str(self.root / "dev.json"),
                "--input-manifest",
                str(self.root / "prepared-input.manifest.json"),
                "--database-root",
                str(self.databases),
                "--output",
                str(self.root / "score.json"),
            ]
        )
        self.assertEqual(args.official_outputs_dir, self.outputs)
        self.assertEqual(
            args.input_manifest, self.root / "prepared-input.manifest.json"
        )
        self.assertEqual(args.output_manifest, self.root / "score.json")

    def test_read_only_execution_and_progress_handler_timeout(self) -> None:
        path = self._database("db", ["CREATE TABLE t(x INTEGER)", "INSERT INTO t VALUES (1)"])
        mutation = scorer.execute_read_only_sql(path, "DELETE FROM t", 1.0)
        self.assertEqual(mutation.status, "execution_error")
        connection = sqlite3.connect(path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM t").fetchone()[0], 1)
        finally:
            connection.close()

        expensive = (
            "WITH RECURSIVE c(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM c "
            "WHERE x < 100000000) SELECT sum(x) FROM c"
        )
        timed = scorer.execute_read_only_sql(path, expensive, 0.000001)
        self.assertEqual(timed.status, "timeout")
        self.assertIn("interrupted", timed.error or "")

    def test_k1_manifest_scores_gold_and_preserves_author_output(self) -> None:
        self._database(
            "values_db",
            [
                "CREATE TABLE items(v TEXT)",
                "INSERT INTO items VALUES ('a'), ('a'), ('b')",
            ],
        )
        dev = self._dev_json(
            [
                {
                    "question_id": 0,
                    "db_id": "values_db",
                    "question": "values?",
                    "SQL": "SELECT v FROM items",
                }
            ]
        )
        output = self.outputs / "0.json"
        output.write_text(
            json.dumps(
                [
                    _assistant_result(
                        0,
                        "<answer>```sql\nSELECT DISTINCT v FROM items ORDER BY v DESC\n```</answer>",
                    )
                ]
            ),
            encoding="utf-8",
        )
        original_bytes = output.read_bytes()

        manifest = scorer.score_bird_dev(
            official_outputs_dir=self.outputs,
            bird_dev_json=dev,
            input_manifest=self._input_manifest(dev, [0]),
            dev_databases_root=self.databases,
            expected_rollouts=1,
            sql_timeout_seconds=1.0,
        )

        self.assertEqual(output.read_bytes(), original_bytes)
        self.assertFalse(manifest["source_outputs_modified"])
        self.assertEqual(manifest["evaluation"]["rollout_aggregation"], "none")
        self.assertEqual(manifest["summary"]["correct_expected_rollouts"], 1)
        self.assertEqual(manifest["summary"]["accuracy_over_expected_rollout_slots"], 1.0)
        self.assertEqual(manifest["summary"]["greedy_accuracy"], 1.0)
        record = manifest["tasks"][0]["rollouts"][0]
        self.assertEqual(record["rollout_idx"], 0)
        self.assertEqual(record["status"], "correct")
        self.assertTrue(record["correct"])

    def test_multiple_rollouts_are_independent_and_missing_is_explicit(self) -> None:
        self._database(
            "pairs",
            [
                "CREATE TABLE t(a INTEGER, b INTEGER)",
                "INSERT INTO t VALUES (1, 2)",
            ],
        )
        dev = self._dev_json(
            [
                {
                    "question_id": 7,
                    "db_id": "pairs",
                    "question": "pair?",
                    "SQL": "SELECT a, b FROM t",
                }
            ]
        )
        (self.outputs / "dev_7.json").write_text(
            json.dumps(
                [
                    _assistant_result(0, "<answer>SELECT a, b FROM t</answer>"),
                    _assistant_result(2, "<answer>SELECT b, a FROM t</answer>"),
                ]
            ),
            encoding="utf-8",
        )

        manifest = scorer.score_bird_dev(
            official_outputs_dir=self.outputs,
            bird_dev_json=dev,
            input_manifest=self._input_manifest(dev, [0]),
            dev_databases_root=self.databases,
            expected_rollouts=3,
            sql_timeout_seconds=1.0,
        )
        records = {record["rollout_idx"]: record for record in manifest["tasks"][0]["rollouts"]}
        self.assertEqual(records[0]["status"], "correct")
        self.assertEqual(records[1]["status"], "missing_rollout")
        self.assertEqual(records[2]["status"], "wrong_result")
        self.assertEqual(manifest["summary"]["missing_rollout_slots"], 1)
        self.assertEqual(manifest["summary"]["independently_scored_rollouts"], 2)
        self.assertNotIn("majority", manifest["evaluation"])

    def test_input_manifest_selection_is_authoritative_and_ignores_score_json(self) -> None:
        self._database(
            "selected",
            ["CREATE TABLE t(x INTEGER)", "INSERT INTO t VALUES (2)"],
        )
        dev = self._dev_json(
            [
                {"question_id": 10, "db_id": "selected", "SQL": "SELECT x FROM t"},
                {"question_id": 11, "db_id": "selected", "SQL": "SELECT x FROM t"},
            ]
        )
        (self.outputs / "dev_11.json").write_text(
            json.dumps([_assistant_result(0, "<answer>SELECT x FROM t</answer>")]),
            encoding="utf-8",
        )
        (self.outputs / "score.json").write_text(
            json.dumps({"schema_version": scorer.SCHEMA_VERSION}), encoding="utf-8"
        )

        manifest = scorer.score_bird_dev(
            official_outputs_dir=self.outputs,
            bird_dev_json=dev,
            input_manifest=self._input_manifest(dev, [1]),
            dev_databases_root=self.databases,
            expected_rollouts=1,
            sql_timeout_seconds=1.0,
        )

        self.assertEqual(manifest["summary"]["tasks"], 1)
        self.assertEqual(manifest["summary"]["greedy_accuracy"], 1.0)
        self.assertEqual(manifest["tasks"][0]["dataset_index"], 1)
        self.assertEqual(
            manifest["summary"]["ignored_non_task_json_files"],
            [str(self.outputs / "score.json")],
        )
        self.assertNotIn("unmatched_output_files", manifest["summary"])

    def test_input_manifest_rejects_duplicate_and_out_of_range_indices(self) -> None:
        dev = self._dev_json(
            [{"question_id": 0, "db_id": "unused", "SQL": "SELECT 1"}]
        )
        duplicate = self._input_manifest(dev, [0, 0])
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            scorer.score_bird_dev(
                official_outputs_dir=self.outputs,
                bird_dev_json=dev,
                input_manifest=duplicate,
                dev_databases_root=self.databases,
            )

        out_of_range = self._input_manifest(dev, [1])
        with self.assertRaisesRegex(ValueError, "outside BIRD dev range"):
            scorer.score_bird_dev(
                official_outputs_dir=self.outputs,
                bird_dev_json=dev,
                input_manifest=out_of_range,
                dev_databases_root=self.databases,
            )

    def test_parse_execution_and_missing_failures_are_counted(self) -> None:
        self._database("bad", ["CREATE TABLE t(x INTEGER)"])
        dev = self._dev_json(
            [
                {"question_id": 0, "db_id": "bad", "SQL": "SELECT x FROM t"},
                {"question_id": 1, "db_id": "bad", "SQL": "SELECT x FROM t"},
                {"question_id": 2, "db_id": "bad", "SQL": "SELECT x FROM t"},
            ]
        )
        (self.outputs / "0.json").write_text(
            json.dumps([_assistant_result(0, "<answer>not sql</answer>")]),
            encoding="utf-8",
        )
        (self.outputs / "1.json").write_text(
            json.dumps([_assistant_result(0, "<answer>SELECT missing FROM t</answer>")]),
            encoding="utf-8",
        )

        manifest = scorer.score_bird_dev(
            official_outputs_dir=self.outputs,
            bird_dev_json=dev,
            input_manifest=self._input_manifest(dev, [0, 1, 2]),
            dev_databases_root=self.databases,
            expected_rollouts=1,
            sql_timeout_seconds=1.0,
        )
        self.assertEqual(manifest["summary"]["parse_errors"], 1)
        self.assertEqual(manifest["summary"]["prediction_execution_errors"], 1)
        self.assertEqual(manifest["summary"]["missing_output_files"], 1)
        self.assertEqual(manifest["summary"]["missing_rollout_slots"], 1)
        self.assertEqual(
            manifest["summary"]["gold_execution_status_counts"],
            {"not_run": 1, "ok": 2},
        )


if __name__ == "__main__":
    unittest.main()
