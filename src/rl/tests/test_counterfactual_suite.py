#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from counterfactual_suite import (  # noqa: E402
    SCHEMA_VERSION,
    load_counterfactual_suite_manifest,
    sha256_file,
    sha256_text,
)


class CounterfactualSuiteManifestTests(unittest.TestCase):
    def test_manifest_binds_task_sql_source_and_database_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            for path, value in ((source, "a"), (alternate, "b")):
                connection = sqlite3.connect(path)
                connection.execute("CREATE TABLE items(value TEXT)")
                connection.execute("INSERT INTO items VALUES (?)", (value,))
                connection.commit()
                connection.close()
            gold_sql = "SELECT value FROM items"
            payload = {
                "schema_version": SCHEMA_VERSION,
                "denotation_comparison": "bird-set",
                "generator": {"name": "unit-test", "seed": 1},
                "quality_gate": {"status": "passed", "audit_sha256": "a" * 64},
                "tasks": {
                    "task_1": {
                        "source_db_sha256": sha256_file(source),
                        "gold_sql_sha256": sha256_text(gold_sql),
                        "min_informative_databases": 1,
                        "databases": [{
                            "path": "alternate.sqlite",
                            "sha256": sha256_file(alternate),
                        }],
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = load_counterfactual_suite_manifest(manifest_path)
            suite = manifest.suite_for({
                "task_id": "task_1",
                "db_path": str(source),
                "gold_sql": gold_sql,
            })

        self.assertEqual(suite.database_paths, (alternate.resolve(),))
        self.assertEqual(suite.min_informative_databases, 1)

    def test_modified_database_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            for path in (source, alternate):
                connection = sqlite3.connect(path)
                connection.execute("CREATE TABLE items(value TEXT)")
                connection.commit()
                connection.close()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "denotation_comparison": "bird-set",
                "generator": {"name": "unit-test"},
                "quality_gate": {"status": "passed", "audit_sha256": "a" * 64},
                "tasks": {
                    "task_1": {
                        "source_db_sha256": sha256_file(source),
                        "gold_sql_sha256": sha256_text("SELECT value FROM items"),
                        "databases": [{
                            "path": str(alternate),
                            "sha256": "0" * 64,
                        }],
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "database hash mismatch"):
                load_counterfactual_suite_manifest(manifest_path)

    def test_unreviewed_manifest_cannot_enable_process_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            alternate = root / "alternate.sqlite"
            for path in (source, alternate):
                connection = sqlite3.connect(path)
                connection.execute("CREATE TABLE items(value TEXT)")
                connection.commit()
                connection.close()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "denotation_comparison": "bird-set",
                "generator": {"name": "unit-test"},
                "quality_gate": {"status": "candidate", "audit_sha256": "a" * 64},
                "tasks": {
                    "task_1": {
                        "source_db_sha256": sha256_file(source),
                        "gold_sql_sha256": sha256_text("SELECT value FROM items"),
                        "databases": [{
                            "path": str(alternate),
                            "sha256": sha256_file(alternate),
                        }],
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "quality_gate.status"):
                load_counterfactual_suite_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
