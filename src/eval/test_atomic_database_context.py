from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from atomic_database_context import (
    CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE,
    CATALOG_CONTEXT_PROFILE,
    FULL_BIRD_CONTEXT_PROFILE,
    FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
    build_full_bird_database_context,
    build_full_context_student_prompt,
    build_full_context_teacher_prompt,
    catalog_profile_student_prompt,
    disabled_tools_for_profile,
    enabled_tools_for_profile,
    enrich_catalog_perception_output,
    model_visible_tool_schema_hash,
    validate_profile_tool,
)
from executor import Harness
from protocol import ProtocolError, tool_schema_hash
from rollout import overview


class AtomicDatabaseContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        db_dir = root / "toy"
        db_dir.mkdir()
        self.db_path = db_dir / "toy.sqlite"
        connection = sqlite3.connect(self.db_path)
        connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, city TEXT)")
        connection.executemany(
            "INSERT INTO people (id, city) VALUES (?, ?)",
            [(1, "Paris"), (2, "London"), (3, "Paris")],
        )
        connection.commit()
        connection.close()

        description_dir = db_dir / "database_description"
        description_dir.mkdir()
        (description_dir / "people.csv").write_text(
            "original_column_name,column_name,column_description,data_format,value_description\n"
            "id,PERSON ID,the unique person identifier,integer,\n"
            "city,CITY,the city where the person lives,text,\n",
            encoding="utf-8",
        )
        self.metadata_path = root / "train_tables.json"
        self.metadata_path.write_text(
            json.dumps([
                {
                    "db_id": "toy",
                    "table_names_original": ["people"],
                    "table_names": ["people"],
                    "column_names_original": [[-1, "*"], [0, "id"], [0, "city"]],
                    "column_names": [[-1, "*"], [0, "person id"], [0, "city"]],
                    "column_types": ["text", "number", "text"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_full_context_contains_schema_semantics_and_live_examples(self) -> None:
        harness = Harness(str(self.db_path))
        context, audit = build_full_bird_database_context(
            harness,
            {"db_id": "toy", "db_path": str(self.db_path)},
            overview(harness),
            value_count=2,
            schema_metadata_json=str(self.metadata_path),
        )
        self.assertEqual(context["schema_scope"], "complete_source_schema")
        self.assertFalse(context["example_values_are_exhaustive"])
        table = context["tables"][0]
        self.assertEqual(table["table_name"], "people")
        columns = {column["name"]: column for column in table["columns"]}
        self.assertEqual(columns["id"]["description"], "the unique person identifier")
        self.assertTrue(columns["id"]["primary_key"])
        self.assertEqual(columns["city"]["semantic_name"], "CITY")
        self.assertEqual(columns["city"]["example_values"], ["Paris", "London"])
        self.assertEqual(audit["bird_column_description_file_count"], 1)
        self.assertEqual(
            audit["disabled_model_tools"],
            ["describe_table", "inspect_column"],
        )

    def test_full_context_prompt_removes_schema_perception_calls(self) -> None:
        student = build_full_context_student_prompt()
        teacher = build_full_context_teacher_prompt(student)
        self.assertNotIn("describe_table(tables)", student)
        self.assertNotIn("inspect_column(table", student)
        self.assertIn("deliberately unavailable", teacher)
        self.assertNotIn("describe_table(tables) ->", teacher)
        self.assertNotIn("inspect_column(table, column) ->", teacher)
        self.assertNotIn(
            "describe_table",
            enabled_tools_for_profile(FULL_BIRD_CONTEXT_PROFILE),
        )
        self.assertEqual(
            disabled_tools_for_profile(FULL_BIRD_CONTEXT_PROFILE),
            {"describe_table", "inspect_column"},
        )

    def test_catalog_profile_hash_preserves_default_atomic_schema(self) -> None:
        self.assertEqual(
            model_visible_tool_schema_hash(CATALOG_CONTEXT_PROFILE),
            tool_schema_hash(),
        )
        self.assertNotEqual(
            model_visible_tool_schema_hash(FULL_BIRD_CONTEXT_PROFILE),
            tool_schema_hash(),
        )
        self.assertNotEqual(
            model_visible_tool_schema_hash(FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE),
            tool_schema_hash(),
        )
        self.assertNotEqual(
            model_visible_tool_schema_hash(FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE),
            model_visible_tool_schema_hash(FULL_BIRD_CONTEXT_PROFILE),
        )

    def test_semantic_perception_profile_preserves_atomic_tool_schema(self) -> None:
        self.assertEqual(
            model_visible_tool_schema_hash(
                CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE
            ),
            tool_schema_hash(),
        )
        self.assertEqual(
            disabled_tools_for_profile(
                CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE
            ),
            frozenset(),
        )

    def test_catalog_prompt_delta_is_profile_scoped(self) -> None:
        base = "BASE PROMPT"
        self.assertEqual(
            catalog_profile_student_prompt(base, CATALOG_CONTEXT_PROFILE),
            base,
        )
        enriched = catalog_profile_student_prompt(
            base,
            CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE,
        )
        self.assertIn("semantic_name", enriched)
        self.assertIn("column_description", enriched)
        self.assertIn("exact raw table and column names", enriched)

    def test_describe_adds_only_short_semantic_names_without_mutation(self) -> None:
        raw = {
            "tables": [{
                "table_name": "people",
                "row_count": 3,
                "columns": [
                    {"name": "id", "type": "integer", "pk": True},
                    {"name": "city", "type": "text", "pk": False},
                ],
                "foreign_keys": [],
            }]
        }
        visible, audit = enrich_catalog_perception_output(
            CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE,
            {"db_id": "toy", "db_path": str(self.db_path)},
            "describe_table",
            {"tables": ["people"]},
            raw,
        )
        columns = {column["name"]: column for column in visible["tables"][0]["columns"]}
        self.assertEqual(columns["id"]["semantic_name"], "PERSON ID")
        self.assertEqual(columns["city"]["semantic_name"], "CITY")
        self.assertNotIn("description", columns["city"])
        self.assertNotIn("column_description", columns["city"])
        self.assertNotIn("data_format", columns["city"])
        self.assertNotIn("semantic_name", raw["tables"][0]["columns"][0])
        self.assertEqual(audit["enriched_field_count"], 2)
        self.assertNotEqual(
            audit["canonical_output_sha256"],
            audit["model_visible_output_sha256"],
        )

    def test_inspect_adds_long_description_but_not_metadata_values(self) -> None:
        raw = {
            "column": "city",
            "distinct_count": 2,
            "has_null": False,
            "frequent_values": ["Paris", "London"],
            "truncated": False,
        }
        visible, audit = enrich_catalog_perception_output(
            CATALOG_BIRD_SEMANTIC_PERCEPTION_PROFILE,
            {"db_id": "toy", "db_path": str(self.db_path)},
            "inspect_column",
            {"table": "people", "column": "city"},
            raw,
        )
        self.assertEqual(
            visible["column_description"],
            "the city where the person lives",
        )
        self.assertEqual(visible["frequent_values"], ["Paris", "London"])
        self.assertNotIn("data_format", visible)
        self.assertNotIn("column_description", raw)
        self.assertEqual(audit["enriched_field_count"], 1)

    def test_default_catalog_perception_is_bit_identical(self) -> None:
        raw = {"column": "city", "frequent_values": ["Paris"]}
        visible, audit = enrich_catalog_perception_output(
            CATALOG_CONTEXT_PROFILE,
            {"db_id": "toy", "db_path": str(self.db_path)},
            "inspect_column",
            {"table": "people", "column": "city"},
            raw,
        )
        self.assertIs(visible, raw)
        self.assertIsNone(audit)

    def test_full_context_rejects_disabled_tool_after_global_parse(self) -> None:
        with self.assertRaises(ProtocolError) as captured:
            validate_profile_tool(
                FULL_BIRD_CONTEXT_PROFILE,
                "describe_table",
                {"tables": ["people"]},
            )
        self.assertEqual(captured.exception.code, "unknown_tool")
        self.assertNotIn(
            "describe_table",
            captured.exception.details["legal_tools"],
        )

    def test_with_inspect_profile_removes_only_describe_table(self) -> None:
        student = build_full_context_student_prompt(
            FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE
        )
        teacher = build_full_context_teacher_prompt(
            student,
            FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
        )
        self.assertNotIn("describe_table(tables)", student)
        self.assertIn("inspect_column(table, column", student)
        self.assertIn("inspect_column(table, column) ->", teacher)
        self.assertEqual(
            disabled_tools_for_profile(FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE),
            {"describe_table"},
        )
        self.assertIn(
            "inspect_column",
            enabled_tools_for_profile(FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE),
        )
        validate_profile_tool(
            FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
            "inspect_column",
            {"table": "people", "column": "city"},
        )
        with self.assertRaises(ProtocolError):
            validate_profile_tool(
                FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
                "describe_table",
                {"tables": ["people"]},
            )

    def test_context_audit_records_with_inspect_profile(self) -> None:
        harness = Harness(str(self.db_path))
        context, audit = build_full_bird_database_context(
            harness,
            {"db_id": "toy", "db_path": str(self.db_path)},
            overview(harness),
            value_count=1,
            schema_metadata_json=str(self.metadata_path),
            profile=FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
        )
        baseline_context, _ = build_full_bird_database_context(
            harness,
            {"db_id": "toy", "db_path": str(self.db_path)},
            overview(harness),
            value_count=1,
            schema_metadata_json=str(self.metadata_path),
            profile=FULL_BIRD_CONTEXT_PROFILE,
        )
        self.assertEqual(context, baseline_context)
        self.assertEqual(
            context["context_profile"],
            FULL_BIRD_CONTEXT_PROFILE,
        )
        self.assertEqual(
            audit["context_profile"],
            FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
        )
        self.assertEqual(audit["disabled_model_tools"], ["describe_table"])


if __name__ == "__main__":
    unittest.main()
