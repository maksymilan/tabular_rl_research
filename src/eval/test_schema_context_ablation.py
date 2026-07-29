#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from catalog import build_catalog  # noqa: E402
from executor import Harness  # noqa: E402
from schema_context_ablation import (  # noqa: E402
    CONTEXT_RENDERER_VERSION,
    FULL_SCHEMA_PROFILE,
    FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE,
    FULL_SCHEMA_SEMANTIC_PROFILE,
    FULL_SCHEMA_SEMANTIC_VALUES_PROFILE,
    INITIAL_CONTEXT_PROFILES,
    LAZY_CATALOG_PROFILE,
    LAZY_SEMANTIC_DESCRIBE_PROFILE,
    align_base_system_prompt,
    align_tool_output,
    build_initial_context,
    context_contract_sha256,
    context_prompt_suffix,
)


class SchemaContextAblationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db_path = root / "toy.sqlite"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE departments(
                dept_id INTEGER PRIMARY KEY,
                dept_code TEXT
            );
            CREATE TABLE staff(
                employee_id INTEGER PRIMARY KEY,
                dept_id INTEGER,
                nm TEXT,
                FOREIGN KEY(dept_id) REFERENCES departments(dept_id)
            );
            INSERT INTO departments VALUES (1, 'ENG'), (2, 'SALES');
            INSERT INTO staff VALUES
                (10, 1, 'Ada'),
                (11, 1, 'Ada'),
                (12, 2, 'Grace');
            """
        )
        connection.close()
        self.schema_path = root / "train_tables.json"
        self.schema_path.write_text(
            json.dumps(
                [
                    {
                        "db_id": "toy",
                        "table_names_original": ["departments", "staff"],
                        "table_names": ["departments", "employee roster"],
                        "column_names_original": [
                            [-1, "*"],
                            [0, "dept_id"],
                            [0, "dept_code"],
                            [1, "employee_id"],
                            [1, "dept_id"],
                            [1, "nm"],
                        ],
                        "column_names": [
                            [-1, "*"],
                            [0, "department id"],
                            [0, "department code"],
                            [1, "employee id"],
                            [1, "department id"],
                            [1, "employee name"],
                        ],
                        "column_types": [
                            "text",
                            "number",
                            "text",
                            "number",
                            "number",
                            "text",
                        ],
                        "primary_keys": [1, 3],
                        "foreign_keys": [[4, 1]],
                    }
                ]
            ),
            encoding="utf-8",
        )
        description_dir = root / "database_description"
        description_dir.mkdir()
        with (description_dir / "staff.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "original_column_name",
                    "column_name",
                    "column_description",
                    "data_format",
                    "value_description",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "original_column_name": "nm",
                    "column_name": "employee name",
                    "column_description": "full name of the employee",
                    "data_format": "text",
                    "value_description": "not model-visible in this arm",
                }
            )
        self.example = {
            "db_id": "toy",
            "db_path": str(self.db_path),
            "question": "Who works here?",
        }
        self.harness = Harness(str(self.db_path))

    def tearDown(self):
        self.harness.conn.close()
        self.tmp.cleanup()

    def build(self, profile):
        return build_initial_context(
            self.harness,
            self.example,
            profile=profile,
            schema_metadata_json=str(self.schema_path),
            value_count=2,
        )

    def test_lazy_profiles_keep_historical_catalog_exact(self):
        expected = build_catalog(self.harness)
        self.assertEqual(expected, self.build(LAZY_CATALOG_PROFILE))
        self.assertEqual(expected, self.build(LAZY_SEMANTIC_DESCRIBE_PROFILE))
        self.assertEqual("", context_prompt_suffix(LAZY_CATALOG_PROFILE))

    def test_full_schema_arm_has_raw_schema_but_no_semantics_or_values(self):
        overview = self.build(FULL_SCHEMA_PROFILE)
        staff = next(table for table in overview["tables"] if table["table_name"] == "staff")
        name = next(column for column in staff["columns"] if column["name"] == "nm")
        self.assertEqual("text", name["type"])
        self.assertNotIn("semantic_name", name)
        self.assertNotIn("example_values", name)
        self.assertIn(
            {"from": "staff.dept_id", "to": "departments.dept_id"},
            overview["relations"],
        )

    def test_semantic_arm_keeps_raw_executable_name_and_adds_alias(self):
        overview = self.build(FULL_SCHEMA_SEMANTIC_PROFILE)
        staff = next(table for table in overview["tables"] if table["table_name"] == "staff")
        self.assertEqual("employee roster", staff["semantic_table_name"])
        name = next(column for column in staff["columns"] if column["name"] == "nm")
        self.assertEqual("employee name", name["semantic_name"])
        self.assertNotIn("example_values", name)

    def test_value_arm_adds_two_distinct_bounded_live_examples(self):
        overview = self.build(FULL_SCHEMA_SEMANTIC_VALUES_PROFILE)
        staff = next(table for table in overview["tables"] if table["table_name"] == "staff")
        name = next(column for column in staff["columns"] if column["name"] == "nm")
        self.assertEqual(["Ada", "Grace"], name["example_values"])

    def test_sql_astra_description_arm_adds_description_without_values(self):
        overview = self.build(FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE)
        staff = next(table for table in overview["tables"] if table["table_name"] == "staff")
        name = next(column for column in staff["columns"] if column["name"] == "nm")
        self.assertEqual("employee name", name["semantic_name"])
        self.assertEqual("full name of the employee", name["description"])
        self.assertNotIn("example_values", name)
        rendered = json.dumps(overview, ensure_ascii=False)
        self.assertNotIn("not model-visible in this arm", rendered)

    def test_describe_feedback_is_aligned_only_for_semantic_profiles(self):
        raw = self.harness.describe_table(["staff"])
        aligned = align_tool_output(
            self.harness,
            "describe_table",
            raw,
            self.example,
            profile=LAZY_SEMANTIC_DESCRIBE_PROFILE,
            schema_metadata_json=str(self.schema_path),
        )
        name = next(
            column
            for column in aligned["tables"][0]["columns"]
            if column["name"] == "nm"
        )
        self.assertEqual("employee name", name["semantic_name"])
        self.assertNotIn("semantic_name", raw["tables"][0]["columns"][2])

    def test_every_arm_has_unique_auditable_contract(self):
        hashes = {context_contract_sha256(profile) for profile in INITIAL_CONTEXT_PROFILES}
        self.assertEqual(len(INITIAL_CONTEXT_PROFILES), len(hashes))
        for profile in INITIAL_CONTEXT_PROFILES[1:]:
            self.assertIn(profile, context_prompt_suffix(profile))
        self.assertEqual("bird-tool-context-ablation-v1", CONTEXT_RENDERER_VERSION)

    def test_full_schema_prompts_replace_all_lazy_only_instructions(self):
        from protocol import get_system_prompt

        base = get_system_prompt()
        self.assertEqual(base, align_base_system_prompt(base, LAZY_CATALOG_PROFILE))
        self.assertEqual(
            base,
            align_base_system_prompt(base, LAZY_SEMANTIC_DESCRIBE_PROFILE),
        )
        for profile in (
            FULL_SCHEMA_PROFILE,
            FULL_SCHEMA_SEMANTIC_PROFILE,
            FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE,
            FULL_SCHEMA_SEMANTIC_VALUES_PROFILE,
        ):
            aligned = align_base_system_prompt(base, profile)
            self.assertNotIn("only (no columns)", aligned)
            self.assertNotIn("describe_table the needed tables first", aligned)
            self.assertNotIn(
                "opening overview lists only table names and relations",
                aligned,
            )
            self.assertIn("INITIAL CONTEXT PROFILE", aligned)


if __name__ == "__main__":
    unittest.main()
