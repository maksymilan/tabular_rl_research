from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rl.runtime.reference_result_filter import (
    audit_task_environment,
    classify_reference_rows,
    filter_training_records,
)


class ReferenceResultFilterTests(unittest.TestCase):
    def test_classifies_only_empty_and_scalar_zero_shapes_as_excluded(self) -> None:
        self.assertEqual(classify_reference_rows([]), "empty_rows")
        self.assertEqual(classify_reference_rows([(0,)]), "zero_scalar")
        self.assertEqual(classify_reference_rows([(0.0,)]), "zero_scalar")
        self.assertEqual(classify_reference_rows([(None,)]), "null_scalar")
        self.assertEqual(classify_reference_rows([(1,)]), "nonempty")
        self.assertEqual(classify_reference_rows([(0, "kept")]), "nonempty")
        self.assertEqual(classify_reference_rows([(0,), (0,)]), "nonempty")

    def test_audit_executes_reference_query_read_only_without_recording_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory, "items.sqlite")
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("CREATE TABLE items(value INTEGER)")
                connection.execute("INSERT INTO items VALUES (7)")
                connection.commit()
            finally:
                connection.close()
            audit = audit_task_environment({
                "task_id": "bird_train_00001",
                "example_index": 1,
                "db_id": "items",
                "db_path": str(db_path),
                "gold_sql": "SELECT value FROM items",
            })
            payload = audit.to_dict()
        self.assertEqual(audit.kind, "nonempty")
        self.assertFalse(audit.excluded)
        self.assertNotIn("rows", payload)
        self.assertNotIn("query", payload)
        self.assertEqual(len(payload["query_sha256"]), 64)

    def test_filter_removes_entire_empty_or_zero_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory, "items.sqlite")
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("CREATE TABLE items(value INTEGER)")
                connection.execute("INSERT INTO items VALUES (2)")
                connection.commit()
            finally:
                connection.close()

            def record(index: int, query: str) -> dict:
                return {
                    "environment": {
                        "task_id": f"bird_train_{index:05d}",
                        "example_index": index,
                        "db_id": "items",
                        "db_path": str(db_path),
                        "gold_sql": query,
                    }
                }

            records = [
                record(1, "SELECT value FROM items"),
                record(2, "SELECT value FROM items WHERE value = 99"),
                record(3, "SELECT COUNT(*) FROM items WHERE value = 99"),
            ]
            retained, audits = filter_training_records(records)
        self.assertEqual(
            [row["environment"]["example_index"] for row in retained],
            [1],
        )
        self.assertEqual(
            [audit.kind for audit in audits],
            ["nonempty", "empty_rows", "zero_scalar"],
        )


if __name__ == "__main__":
    unittest.main()
