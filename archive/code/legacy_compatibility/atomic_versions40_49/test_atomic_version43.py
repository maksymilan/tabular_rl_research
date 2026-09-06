#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version42 as version42  # noqa: E402
import atomic_version43 as version43  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    selected_protocol_hash,
    selected_tool_schema_hash,
)
from protocol import ProtocolError  # noqa: E402


class AtomicVersion43Tests(unittest.TestCase):
    def setUp(self):
        self.harness = Harness(":memory:")
        self.addCleanup(self.harness.conn.close)
        self.harness.conn.executescript(
            """
            CREATE TABLE players(playerID TEXT, teamID TEXT);
            CREATE TABLE appearances(playerID TEXT, games INTEGER);
            INSERT INTO players VALUES ('p1', 't1');
            INSERT INTO appearances VALUES ('p1', 10);
            """
        )
        self.harness.register_sources()

    def _joined_table(self) -> str:
        return self.harness.join_tables(
            base="players",
            joins=[{
                "table": "appearances",
                "on": [{"left": "players.playerID", "right": "playerID"}],
            }],
        )["table_name"]

    def test_only_terminal_resolution_semantics_change(self):
        self.assertEqual(version43.PROTOCOL_VERSION, "version43")
        self.assertEqual(version43.TOOLS, version42.TOOLS)
        self.assertEqual(version43.MODEL_ARG_SCHEMA, version42.MODEL_ARG_SCHEMA)
        for tool in version43.TOOLS - {"answer_from_context"}:
            self.assertEqual(version43.TOOL_SPECS[tool], version42.TOOL_SPECS[tool])
        self.assertNotEqual(
            version43.TOOL_SPECS["answer_from_context"],
            version42.TOOL_SPECS["answer_from_context"],
        )

    def test_prompt_states_unique_bare_resolution_once(self):
        prompt = version43.provider_system_prompt("deepseek-v4-flash")
        self.assertEqual(prompt.count("OUTPUT CONTRACT"), 1)
        self.assertEqual(
            prompt.count("bare name that matches exactly one dotted logical column by suffix"),
            1,
        )
        self.assertNotIn("[exact_column,...]", prompt)
        self.assertNotIn("plan(", prompt)
        self.assertNotIn("read_subtable", prompt)

    def test_terminal_lowering_prefers_exact_name(self):
        lowered, audit = version43.lower_terminal_evidence(
            self.harness,
            {
                "evidence": {
                    "table": "players",
                    "columns": ["playerID"],
                }
            },
        )
        self.assertEqual(
            self.harness.rows(lowered["evidence"]["table"]),
            [("p1",)],
        )
        self.assertEqual(audit["resolved_columns"], ["playerID"])
        self.assertEqual(audit["column_resolution"][0]["mode"], "exact")

    def test_terminal_lowering_resolves_one_unique_bare_suffix(self):
        joined = self._joined_table()
        lowered, audit = version43.lower_terminal_evidence(
            self.harness,
            {
                "evidence": {
                    "table": joined,
                    "columns": ["games"],
                }
            },
        )
        self.assertEqual(
            self.harness.rows(lowered["evidence"]["table"]),
            [(10,)],
        )
        self.assertEqual(audit["resolved_columns"], ["appearances.games"])
        self.assertEqual(audit["column_resolution"][0], {
            "requested": "games",
            "resolved": "appearances.games",
            "mode": "unique-bare",
        })
        self.assertEqual(audit["schema"], "terminal-column-projection-v2")

    def test_terminal_lowering_rejects_ambiguous_bare_suffix(self):
        joined = self._joined_table()
        with self.assertRaises(ProtocolError) as raised:
            version43.lower_terminal_evidence(
                self.harness,
                {
                    "evidence": {
                        "table": joined,
                        "columns": ["playerID"],
                    }
                },
            )
        self.assertEqual(
            raised.exception.details["candidate_columns"],
            ["players.playerID", "appearances.playerID"],
        )

    def test_terminal_lowering_does_not_repair_wrong_qualified_name(self):
        joined = self._joined_table()
        with self.assertRaises(ProtocolError):
            version43.lower_terminal_evidence(
                self.harness,
                {
                    "evidence": {
                        "table": joined,
                        "columns": ["wrong.games"],
                    }
                },
            )

    def test_teacher_rollout_registry_selects_version43_hashes(self):
        prompt = version43.provider_system_prompt("deepseek-v4-flash")
        self.assertIn("version43", ATOMIC_PROTOCOL_VERSIONS)
        self.assertEqual(
            selected_tool_schema_hash("version43"),
            version43.tool_schema_hash(),
        )
        self.assertEqual(
            selected_protocol_hash("version43", prompt),
            version43.protocol_hash(prompt),
        )


if __name__ == "__main__":
    unittest.main()
