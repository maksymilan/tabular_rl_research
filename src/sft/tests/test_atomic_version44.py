#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version44 as version44  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    VERSION44_DATA_GENERATION_SUFFIX,
    selected_protocol_hash,
    selected_tool_schema_hash,
)
from protocol import ProtocolError, rolling_system_prompt  # noqa: E402
from public_tool_contract import (  # noqa: E402
    PUBLIC_TOOL_CONTRACTS,
    VERSION44_PUBLIC_TOOL_CONTRACTS,
)
from rollout import execute_tool, new_ctx, overview  # noqa: E402


class AtomicVersion44Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness(":memory:")
        self.harness.conn.executescript(
            """
            CREATE TABLE teams(id INTEGER, name TEXT, city TEXT);
            INSERT INTO teams VALUES
              (1, 'Avangard Omsk', 'Omsk'),
              (2, 'Avangard Omsk', 'Omsk'),
              (3, 'Avangrad Omsk', 'Omsk'),
              (4, 'Dynamo Moscow', 'Moscow'),
              (5, 'Omsk Wings', 'Omsk');
            """
        )
        self.harness.register_sources()

    def test_public_surface_has_only_approved_deltas(self) -> None:
        self.assertEqual(version44.PROTOCOL_VERSION, "version44")
        self.assertNotIn("plan", version44.TOOLS)
        self.assertNotIn("read_subtable", version44.TOOLS)
        self.assertIn("inspect_rows", version44.TOOLS)
        self.assertIn("search_values", version44.TOOLS)
        self.assertEqual(
            VERSION44_PUBLIC_TOOL_CONTRACTS["join_tables"],
            PUBLIC_TOOL_CONTRACTS["join_tables"],
        )
        self.assertEqual(
            VERSION44_PUBLIC_TOOL_CONTRACTS["answer_from_context"],
            PUBLIC_TOOL_CONTRACTS["answer_from_context"],
        )

    def test_parser_validates_search_scope_and_pagination(self) -> None:
        valid = (
            "<think>Find the stored team spelling in this table.</think>\n"
            '{"tool":"search_values","arguments":'
            '{"table":"teams","query":"Avangard Omsk","limit":20}}'
        )
        _, tool, arguments = version44.parse_assistant_strict(valid)
        self.assertEqual(tool, "search_values")
        self.assertEqual(arguments["table"], "teams")
        for bad_arguments in (
            {"query": "Avangard"},
            {"table": "teams", "query": ""},
            {"table": "teams", "query": "Avangard", "column": ""},
            {"table": "teams", "query": "Avangard", "limit": 21},
            {"table": "teams", "query": "Avangard", "offset": -1},
        ):
            text = (
                "<think>Try a malformed search.</think>\n"
                '{"tool":"search_values","arguments":'
                + json.dumps(bad_arguments, separators=(",", ":"))
                + "}"
            )
            with self.assertRaises(ProtocolError):
                version44.parse_assistant_strict(text)

    def test_parser_rejects_retired_public_names(self) -> None:
        for tool, arguments in (
            ("plan", {"ops": []}),
            ("read_subtable", {"table": "teams", "limit": 5}),
        ):
            text = (
                "<think>Try a retired public name.</think>\n"
                f'{{"tool":"{tool}","arguments":'
                + json.dumps(arguments, separators=(",", ":"))
                + "}"
            )
            with self.assertRaises(ProtocolError):
                version44.parse_assistant_strict(text)

    def test_search_values_is_fuzzy_column_scoped_and_paginated(self) -> None:
        first = self.harness.search_values(
            "teams",
            "Avangard Omsk",
            limit=1,
        )
        self.assertEqual(first["matches"][0]["value"], "Avangard Omsk")
        self.assertEqual(first["matches"][0]["frequency"], 2)
        self.assertEqual(first["matches"][0]["match_type"], "exact")
        self.assertEqual(first["next_offset"], 1)

        second = self.harness.search_values(
            "teams",
            "Avangard Omsk",
            limit=1,
            offset=first["next_offset"],
        )
        self.assertEqual(second["matches"][0]["value"], "Avangrad Omsk")
        self.assertEqual(second["matches"][0]["match_type"], "fuzzy")
        self.assertFalse(second["has_more"])

        city_only = self.harness.search_values(
            "teams",
            "Omsk",
            column="city",
        )
        self.assertEqual(city_only["searched_columns"], ["city"])
        self.assertEqual(
            {(item["column"], item["value"]) for item in city_only["matches"]},
            {("city", "Omsk")},
        )

    def test_search_values_caps_each_page_at_twenty(self) -> None:
        self.harness.conn.executemany(
            "INSERT INTO teams(id, name, city) VALUES (?, ?, ?)",
            [
                (100 + index, f"Alpha Team {index:02d}", "Elsewhere")
                for index in range(25)
            ],
        )
        first = self.harness.search_values("teams", "Alpha Team")
        self.assertEqual(len(first["matches"]), 20)
        self.assertEqual(first["total_matches"], 25)
        self.assertEqual(first["next_offset"], 20)
        second = self.harness.search_values(
            "teams",
            "Alpha Team",
            offset=first["next_offset"],
        )
        self.assertEqual(len(second["matches"]), 5)
        self.assertFalse(second["has_more"])

    def test_search_is_resident_and_grounds_later_filter_literal(self) -> None:
        context = new_ctx(overview(self.harness))
        execute_tool(
            self.harness,
            "search_values",
            {"table": "teams", "query": "Avangard Omsk"},
            context,
            "step_1",
        )
        output, _ = execute_tool(
            self.harness,
            "condition_filter",
            {
                "table": "teams",
                "conditions": {
                    "column": "name",
                    "op": "=",
                    "value": "Avangard Omsk",
                },
            },
            context,
            "step_2",
        )
        self.assertEqual(output["row_count"], 2)
        references = context["history"]["step_2"]["references"]
        grounding = next(
            reference
            for reference in references
            if reference.get("role") == "domain_observation"
        )
        self.assertEqual(grounding["step"], "step_1")
        self.assertEqual(grounding["target"]["column"], "name")
        state = context["environment"].snapshot()
        self.assertEqual(
            state["tables"]["teams"]["value_searches"][0]["matches"][0]["value"],
            "Avangard Omsk",
        )

    def test_teacher_prompt_and_registry_are_version44_specific(self) -> None:
        student = rolling_system_prompt(
            version44.STUDENT_SYSTEM_PROMPT,
            compact=False,
        )
        canonical = version44.teacher_system_prompt(student) + VERSION44_DATA_GENERATION_SUFFIX
        provider_prompt = version44.provider_system_prompt(
            "deepseek-v4-flash",
            canonical,
        )
        self.assertIn(
            "search_values(table, query, column?, limit?, offset?)",
            provider_prompt,
        )
        self.assertIn(
            "inspect_rows(table, limit?, columns?, conditions?, order_by?, offset?)",
            provider_prompt,
        )
        self.assertIn("Its table is mandatory", provider_prompt)
        self.assertIn("omitted column means every column", provider_prompt)
        self.assertIn("cannot exceed 20", provider_prompt)
        self.assertIn("same deterministic relevance ordering", provider_prompt)
        self.assertIn(
            "describe_table returns only raw executable schema identifiers",
            provider_prompt,
        )
        self.assertIn(
            "inspect_column may additionally return BIRD semantic_name and "
            "column_description",
            provider_prompt,
        )
        self.assertIn("not executable aliases", provider_prompt)
        self.assertIn("never filters rows or creates a relation", provider_prompt)
        self.assertNotIn("read_subtable", provider_prompt)
        self.assertNotIn("plan(", provider_prompt)
        self.assertEqual(provider_prompt.count("ROLLING LEGAL HISTORY"), 1)
        self.assertEqual(provider_prompt.count("ONE REQUEST = ONE ACTION"), 1)
        self.assertIn("JSON Output", provider_prompt)
        self.assertIn("version44", ATOMIC_PROTOCOL_VERSIONS)
        self.assertEqual(
            selected_tool_schema_hash("version44"),
            version44.tool_schema_hash(),
        )
        self.assertEqual(
            selected_protocol_hash("version44", provider_prompt),
            version44.protocol_hash(provider_prompt),
        )


if __name__ == "__main__":
    unittest.main()
