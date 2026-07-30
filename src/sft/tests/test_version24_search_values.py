from __future__ import annotations

import unittest

from executor import Harness
from protocol import (
    PROTOCOL_VERSION,
    TOOLS,
    ProtocolError,
    parse_assistant_strict,
    validate_model_arguments,
)
from rollout import execute_tool, new_ctx, overview


class Version24SearchValuesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness(":memory:")
        self.harness.conn.executescript(
            """
            CREATE TABLE teams(team TEXT, city TEXT);
            INSERT INTO teams VALUES
              ('Avangard Omsk', 'Omsk'),
              ('Avangard Omsk', 'Omsk'),
              ('Avangard Omsk-2', 'Omsk'),
              ('Ak Bars Kazan', 'Kazan');
            """
        )
        self.harness.register_sources()

    def tearDown(self) -> None:
        self.harness.conn.close()

    def test_surface_keeps_version24_tools_and_adds_only_search(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, "version24-search-values-v1")
        self.assertIn("plan", TOOLS)
        self.assertIn("read_subtable", TOOLS)
        self.assertIn("search_values", TOOLS)
        self.assertEqual(len(TOOLS), 13)
        think, tool, arguments = parse_assistant_strict(
            "<think>Find the exact stored spelling.</think>\n"
            '<tool_call>{"tool":"search_values","arguments":'
            '{"table":"teams","query":"Avangrd Omsk","column":"team","limit":20}}'
            "</tool_call>"
        )
        self.assertTrue(think)
        self.assertEqual(tool, "search_values")
        self.assertEqual(arguments["column"], "team")

    def test_argument_validation_is_closed_and_bounded(self) -> None:
        validate_model_arguments(
            "search_values",
            {"table": "teams", "query": "Omsk", "offset": 0, "limit": 20},
        )
        invalid = (
            {"table": "", "query": "Omsk"},
            {"table": "teams", "query": ""},
            {"table": "teams", "query": "Omsk", "limit": 21},
            {"table": "teams", "query": "Omsk", "offset": -1},
            {"table": "teams", "query": "Omsk", "extra": True},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ProtocolError):
                    validate_model_arguments("search_values", arguments)

    def test_exact_suppresses_broader_candidates_and_typo_is_recalled(self) -> None:
        exact = self.harness.search_values(
            "teams",
            "Avangard Omsk",
            column="team",
        )
        self.assertEqual([item["value"] for item in exact["matches"]], ["Avangard Omsk"])
        self.assertEqual(exact["matches"][0]["frequency"], 2)
        self.assertEqual(exact["matches"][0]["match_type"], "exact")

        fuzzy = self.harness.search_values(
            "teams",
            "Avangrd Omsk",
            column="team",
        )
        self.assertEqual(fuzzy["matches"][0]["value"], "Avangard Omsk")
        self.assertEqual(fuzzy["matches"][0]["match_type"], "fuzzy")
        self.assertEqual(fuzzy["total_matches_scope"], "bounded_candidate_pool")
        self.assertFalse(fuzzy["candidate_truncated"])

    def test_execute_tool_keeps_search_read_only_and_resident(self) -> None:
        context = new_ctx(overview(self.harness))
        output, created = execute_tool(
            self.harness,
            "search_values",
            {
                "table": "teams",
                "query": "Avangrd Omsk",
                "column": "team",
                "limit": 20,
            },
            context,
            "step_1",
        )
        self.assertIsNone(created)
        self.assertEqual(output["matches"][0]["value"], "Avangard Omsk")
        snapshot = context["environment"].snapshot()
        searches = snapshot["tables"]["teams"]["value_searches"]
        self.assertEqual(searches[0]["from_step"], "step_1")
        self.assertEqual(searches[0]["matches"][0]["value"], "Avangard Omsk")

        _, created = execute_tool(
            self.harness,
            "condition_filter",
            {
                "table": "teams",
                "conditions": {
                    "column": "team",
                    "op": "=",
                    "value": "Avangard Omsk",
                },
            },
            context,
            "step_2",
        )
        self.assertIsNotNone(created)
        references = context["history"]["step_2"]["references"]
        self.assertTrue(any(
            reference.get("type") == "grounding"
            and reference.get("step") == "step_1"
            and reference.get("role") == "domain_observation"
            for reference in references
        ))


if __name__ == "__main__":
    unittest.main()
