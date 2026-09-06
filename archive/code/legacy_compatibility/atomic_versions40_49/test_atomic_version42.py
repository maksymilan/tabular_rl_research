#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version41 as version41  # noqa: E402
import atomic_version42 as version42  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    CATALOG_CONTEXT_PROFILE,
    POLICY_PROMPT_CANONICAL,
    run_rollout,
    selected_protocol_hash,
    selected_tool_schema_hash,
)
from protocol import ProtocolError  # noqa: E402


class AtomicVersion42Tests(unittest.TestCase):
    def test_only_terminal_tool_semantics_change(self):
        self.assertEqual(version42.PROTOCOL_VERSION, "version42")
        self.assertEqual(version42.TOOLS, version41.TOOLS)
        self.assertEqual(version42.MODEL_ARG_SCHEMA, version41.MODEL_ARG_SCHEMA)
        for tool in version42.TOOLS - {"answer_from_context"}:
            self.assertEqual(version42.TOOL_SPECS[tool], version41.TOOL_SPECS[tool])
        self.assertNotEqual(
            version42.TOOL_SPECS["answer_from_context"],
            version41.TOOL_SPECS["answer_from_context"],
        )
        self.assertNotEqual(
            version42.tool_schema_hash(),
            version41.tool_schema_hash(),
        )

    def test_prompt_requires_explicit_terminal_columns_once(self):
        prompt = version42.provider_system_prompt("deepseek-v4-flash")
        self.assertEqual(prompt.count("OUTPUT CONTRACT"), 1)
        self.assertIn(
            'evidence is exactly {"table":handle,"columns":[exact_column,...]}',
            prompt,
        )
        self.assertIn(
            '"evidence":{"table":"top_002","columns":["label"]}',
            prompt,
        )
        self.assertNotIn(
            '"evidence":{"table":"project_003"}',
            prompt,
        )
        self.assertNotIn("plan(", prompt)
        self.assertNotIn("read_subtable", prompt)

    def test_parser_requires_nonempty_unique_terminal_columns(self):
        valid = (
            "<think>Select only the requested label column at termination.</think>"
            '{"tool":"answer_from_context","arguments":{"evidence":'
            '{"table":"top_002","columns":["label"]},"reason":"label only"}}'
        )
        _, tool, args = version42.parse_assistant_strict(valid)
        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"]["columns"], ["label"])

        invalid_evidence = (
            {"table": "top_002"},
            {"table": "top_002", "columns": []},
            {"table": "top_002", "columns": ["label", "LABEL"]},
        )
        for evidence in invalid_evidence:
            text = (
                "<think>Try an invalid terminal shape.</think>"
                '{"tool":"answer_from_context","arguments":{"evidence":'
                + json.dumps(evidence, separators=(",", ":"))
                + "}}"
            )
            with self.subTest(evidence=evidence), self.assertRaises(ProtocolError):
                version42.parse_assistant_strict(text)

    def test_terminal_lowering_projects_exact_existing_columns_in_order(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            """
            CREATE TABLE ranked(label TEXT, review REAL, restaurant_id INTEGER);
            INSERT INTO ranked VALUES ('petalumas', 3.3, 7);
            """
        )
        harness.register_sources()
        lowered, audit = version42.lower_terminal_evidence(
            harness,
            {
                "evidence": {
                    "table": "ranked",
                    "columns": ["restaurant_id", "label"],
                },
                "reason": "Return the requested identifier and label.",
            },
        )
        projected = lowered["evidence"]["table"]
        self.assertEqual(harness.rows(projected), [(7, "petalumas")])
        self.assertEqual(audit["resolved_columns"], ["restaurant_id", "label"])
        self.assertEqual(audit["columns"], ["restaurant_id", "label"])
        self.assertEqual(audit["schema"], "terminal-column-projection-v1")

    def test_terminal_lowering_rejects_unknown_or_expression_columns(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            """
            CREATE TABLE ranked(label TEXT, review REAL);
            INSERT INTO ranked VALUES ('petalumas', 3.3);
            """
        )
        harness.register_sources()
        for column in ("missing", "review * 2"):
            with self.subTest(column=column), self.assertRaises(ProtocolError):
                version42.lower_terminal_evidence(
                    harness,
                    {
                        "evidence": {
                            "table": "ranked",
                            "columns": [column],
                        }
                    },
                )

    def test_teacher_rollout_registry_selects_version42_hashes(self):
        prompt = version42.provider_system_prompt("deepseek-v4-flash")
        self.assertIn("version42", ATOMIC_PROTOCOL_VERSIONS)
        self.assertEqual(
            selected_tool_schema_hash("version42"),
            version42.tool_schema_hash(),
        )
        self.assertEqual(
            selected_protocol_hash("version42", prompt),
            version42.protocol_hash(prompt),
        )

    def test_teacher_rollout_scores_the_deterministic_terminal_projection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "terminal.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE ranked(label TEXT, review REAL);
                INSERT INTO ranked VALUES ('petalumas', 3.3);
                """
            )
            connection.close()
            response = (
                "<think>The question requests label only, so select that terminal column.</think>"
                '{"tool":"answer_from_context","arguments":{"evidence":'
                '{"table":"ranked","columns":["label"]},"reason":"label only"}}'
            )
            harnesses = []

            def harness_factory(path):
                harness = Harness(path)
                harnesses.append(harness)
                return harness

            try:
                with (
                    patch(
                        "generate_teacher_rollouts.chat_with_retries",
                        return_value=(response, {}, None),
                    ),
                    patch(
                        "generate_teacher_rollouts.Harness",
                        side_effect=harness_factory,
                    ),
                ):
                    record = run_rollout(
                        example_index=1,
                        ex={
                            "db_id": "terminal_test",
                            "question": "Which label?",
                            "db_path": str(database),
                            "gold_sql": "SELECT label FROM ranked",
                        },
                        split="train",
                        base_url="https://unused.invalid",
                        api_key="unused",
                        model="unit-test-model",
                        system_prompt=version42.STUDENT_SYSTEM_PROMPT,
                        max_steps=3,
                        max_tokens=512,
                        api_retries=1,
                        api_timeout=1,
                        max_errors_per_type=3,
                        table_output_rows=0,
                        context_mode="rolling-legal-history",
                        history_turns=4,
                        rolling_prompt_variant="full",
                        policy_prompt_variant=POLICY_PROMPT_CANONICAL,
                        diagnostic_only=True,
                        database_context_profile=CATALOG_CONTEXT_PROFILE,
                        atomic_protocol_version="version42",
                    )
            finally:
                for harness in harnesses:
                    harness.conn.close()
        self.assertTrue(record["correct"])
        self.assertEqual(record["pred_sample"], [("petalumas",)])
        projection = record["turns"][0]["terminal_projection"]
        self.assertEqual(projection["source_table"], "ranked")
        self.assertEqual(projection["resolved_columns"], ["label"])
        self.assertEqual(
            record["terminal_evidence_policy"],
            "explicit-columns-deterministic-project-v1",
        )


if __name__ == "__main__":
    unittest.main()
