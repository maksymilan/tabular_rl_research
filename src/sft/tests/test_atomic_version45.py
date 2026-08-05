#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version44 as version44  # noqa: E402
import atomic_version45 as version45  # noqa: E402
import executor as executor_module  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    VERSION44_DATA_GENERATION_SUFFIX,
    selected_protocol_hash,
    selected_tool_schema_hash,
)
from protocol import rolling_system_prompt  # noqa: E402
from rollout import execute_tool, new_ctx, overview  # noqa: E402


class AtomicVersion45Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness(":memory:")
        self.harness.conn.execute("CREATE TABLE values_table(value TEXT, category TEXT)")
        self.harness.conn.executemany(
            "INSERT INTO values_table(value, category) VALUES (?, ?)",
            [
                ("Avangard Omsk", "team"),
                ("Avangrad Omsk", "team"),
                ("Dynamo Moscow", "team"),
            ],
        )
        self.harness.register_sources()

    def test_protocol_keeps_calls_but_audits_bounded_recall(self) -> None:
        self.assertEqual(version45.PROTOCOL_VERSION, "version45")
        self.assertEqual(version45.TOOLS, version44.TOOLS)
        self.assertEqual(version45.MODEL_ARG_SCHEMA, version44.MODEL_ARG_SCHEMA)
        self.assertNotEqual(version45.tool_schema_hash(), version44.tool_schema_hash())
        self.assertIn("candidate_truncated=true", version45.STUDENT_SYSTEM_PROMPT)
        self.assertIn("bounded candidate-pool", version45.STUDENT_SYSTEM_PROMPT)
        self.assertNotIn("plan(", version45.STUDENT_SYSTEM_PROMPT)
        self.assertNotIn("read_subtable", version45.STUDENT_SYSTEM_PROMPT)
        self.assertIn("version45", ATOMIC_PROTOCOL_VERSIONS)
        self.assertEqual(
            selected_tool_schema_hash("version45"),
            version45.tool_schema_hash(),
        )

        student = rolling_system_prompt(version45.STUDENT_SYSTEM_PROMPT, compact=False)
        teacher = (
            version45.teacher_system_prompt(student)
            + VERSION44_DATA_GENERATION_SUFFIX
        )
        provider = version45.provider_system_prompt("deepseek-v4-flash", teacher)
        self.assertIn("Similar candidates are alternatives", provider)
        self.assertEqual(
            selected_protocol_hash("version45", provider),
            version45.protocol_hash(provider),
        )

    def test_bounded_search_preserves_exact_and_typo_matches(self) -> None:
        result = self.harness.search_values_bounded(
            "values_table",
            "Avangrd Omsk",
            column="value",
        )
        self.assertFalse(result["candidate_truncated"])
        self.assertEqual(result["total_matches_scope"], "bounded_candidate_pool")
        self.assertEqual(result["matches"][0]["value"], "Avangard Omsk")
        self.assertEqual(result["matches"][0]["match_type"], "fuzzy")

        exact = self.harness.search_values_bounded(
            "values_table",
            "Avangard Omsk",
            column="value",
        )
        self.assertEqual(exact["matches"][0]["match_type"], "exact")
        self.assertEqual(
            [item["value"] for item in exact["matches"]],
            ["Avangard Omsk"],
        )

    def test_candidate_pool_is_bounded_and_pages_stably(self) -> None:
        self.harness.conn.executemany(
            "INSERT INTO values_table(value, category) VALUES (?, 'bulk')",
            [(f"Alpha Team {index:05d}",) for index in range(5_000)],
        )
        with patch.object(
            executor_module,
            "_value_search_match",
            wraps=executor_module._value_search_match,
        ) as matcher:
            first = self.harness.search_values_bounded(
                "values_table",
                "Alpha Team",
                column="value",
                limit=20,
            )
        self.assertTrue(first["candidate_truncated"])
        self.assertEqual(first["candidate_count"], 4_096)
        self.assertLessEqual(matcher.call_count, 4_096)
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_offset"], 20)

        second = self.harness.search_values_bounded(
            "values_table",
            "Alpha Team",
            column="value",
            limit=20,
            offset=20,
        )
        repeated = self.harness.search_values_bounded(
            "values_table",
            "Alpha Team",
            column="value",
            limit=20,
            offset=20,
        )
        first_values = [item["value"] for item in first["matches"]]
        second_values = [item["value"] for item in second["matches"]]
        self.assertFalse(set(first_values) & set(second_values))
        self.assertEqual(second_values, [item["value"] for item in repeated["matches"]])

    def test_execution_mode_does_not_change_frozen_version44_default(self) -> None:
        exhaustive_context = new_ctx(overview(self.harness))
        with (
            patch.object(
                self.harness,
                "search_values",
                wraps=self.harness.search_values,
            ) as exhaustive,
            patch.object(
                self.harness,
                "search_values_bounded",
                wraps=self.harness.search_values_bounded,
            ) as bounded,
        ):
            execute_tool(
                self.harness,
                "search_values",
                {"table": "values_table", "query": "Avangard Omsk"},
                exhaustive_context,
                "step_1",
            )
            exhaustive.assert_called_once()
            bounded.assert_not_called()

        bounded_context = new_ctx(overview(self.harness))
        bounded_context["search_values_mode"] = "bounded-v1"
        with (
            patch.object(
                self.harness,
                "search_values",
                wraps=self.harness.search_values,
            ) as exhaustive,
            patch.object(
                self.harness,
                "search_values_bounded",
                wraps=self.harness.search_values_bounded,
            ) as bounded,
        ):
            output, _ = execute_tool(
                self.harness,
                "search_values",
                {"table": "values_table", "query": "Avangard Omsk"},
                bounded_context,
                "step_1",
            )
            exhaustive.assert_not_called()
            bounded.assert_called_once()
            self.assertEqual(output["total_matches_scope"], "bounded_candidate_pool")


if __name__ == "__main__":
    unittest.main()
