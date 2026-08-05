#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version46 as version46  # noqa: E402
import atomic_version47 as version47  # noqa: E402
import atomic_version48 as version48  # noqa: E402
import atomic_version49 as version49  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    selected_protocol_hash,
    selected_resident_context,
    selected_tool_schema_hash,
)
from protocol import (  # noqa: E402
    RESIDENT_STATE_PROFILE_HANDLE_CARDS,
    RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS,
    RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
    model_visible_environment_state,
    rolling_legal_history_messages,
    tool_output_message,
    tool_schema_hash,
)


def _state() -> dict:
    return {
        "plan": [],
        "tables": {
            "people": {
                "kind": "source",
                "row_count": 2,
                "schema": {
                    "from_step": "step_1",
                    "columns": [
                        {"name": "id", "type": "integer"},
                        {"name": "name", "type": "text"},
                    ],
                    "foreign_keys": [],
                },
                "reads": [{
                    "from_step": "step_2",
                    "columns": ["id", "name"],
                    "limit": 2,
                    "row_count": 2,
                    "rows": [
                        {"id": 1, "name": "Alice"},
                        {"id": 2, "name": "Bob"},
                    ],
                }],
            },
            "filter_001": {
                "kind": "derived",
                "created_by": "step_3",
                "row_count": 1,
                "columns": ["id", "name"],
                "reads": [{
                    "from_step": "step_3",
                    "row_count": 1,
                    "rows": [{"id": 1, "name": "Alice"}],
                    "note": "inline table output preview",
                }],
                "derivation": {
                    "schema": "relation-derivation-v1",
                    "operator": "condition_filter",
                    "inputs": [{
                        "kind": "table",
                        "role": "input",
                        "ref": "people",
                    }],
                    "semantics": {
                        "row_operation": "filter",
                        "predicate": {"column": "id", "op": "=", "value": 1},
                        "predicate_columns": ["id"],
                        "column_operation": "preserve",
                        "projected_columns": ["id", "name"],
                    },
                },
            },
            "other": {
                "kind": "source",
                "row_count": 1,
                "reads": [{
                    "from_step": "step_4",
                    "columns": ["id", "label"],
                    "limit": 1,
                    "row_count": 1,
                    "rows": [{"id": 9, "label": "Inactive source cell"}],
                }],
            },
            "filter_002": {
                "kind": "derived",
                "created_by": "step_5",
                "row_count": 1,
                "columns": ["id", "label"],
                "reads": [{
                    "from_step": "step_5",
                    "row_count": 1,
                    "rows": [{"id": 9, "label": "Inactive derived cell"}],
                }],
                "derivation": {
                    "schema": "relation-derivation-v1",
                    "operator": "condition_filter",
                    "inputs": [{"kind": "table", "role": "input", "ref": "other"}],
                    "semantics": {
                        "row_operation": "filter",
                        "predicate": {"column": "id", "op": "=", "value": 9},
                        "predicate_columns": ["id"],
                        "column_operation": "preserve",
                        "projected_columns": ["id", "label"],
                    },
                },
            },
        },
        "values": {},
    }


class AtomicVersion46ContextTests(unittest.TestCase):
    def test_versions_are_registered_and_keep_version39_tools(self) -> None:
        for module in (version46, version47, version48, version49):
            self.assertIn(module.PROTOCOL_VERSION, ATOMIC_PROTOCOL_VERSIONS)
            self.assertEqual(selected_tool_schema_hash(module.PROTOCOL_VERSION), tool_schema_hash())
            self.assertEqual(
                selected_protocol_hash(module.PROTOCOL_VERSION, "prompt"),
                module.protocol_hash("prompt"),
            )
        self.assertIn("latest Harness output", version48.INTERPRET_BEFORE_ACT_SUFFIX)

    def test_handle_card_replaces_only_model_visible_derivation(self) -> None:
        canonical = _state()
        before = deepcopy(canonical)
        visible = model_visible_environment_state(
            canonical,
            RESIDENT_STATE_PROFILE_HANDLE_CARDS,
        )
        derived = visible["tables"]["filter_001"]
        self.assertNotIn("derivation", derived)
        self.assertIn("created by condition_filter", derived["handle_card"])
        self.assertIn('"ref":"people"', derived["handle_card"])
        self.assertIn('"column":"id"', derived["handle_card"])
        self.assertEqual(canonical, before)
        self.assertIn("derivation", canonical["tables"]["filter_001"])
        self.assertEqual(derived["reads"][0]["rows"][0]["name"], "Alice")

    def test_archived_read_profile_removes_cells_but_keeps_read_descriptor(self) -> None:
        visible = model_visible_environment_state(
            _state(),
            RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS,
        )
        read = visible["tables"]["people"]["reads"][0]
        self.assertNotIn("rows", read)
        self.assertEqual(read["returned_row_count"], 2)
        self.assertEqual(read["columns"], ["id", "name"])
        self.assertEqual(read["from_step"], "step_2")
        self.assertIn("observe", read["row_values"])

    def test_version47_keeps_only_latest_observation_full(self) -> None:
        assistant = (
            "<think>Inspect rows.</think>"
            '{"tool":"read_subtable","arguments":{"table":"people","limit":2}}'
        )
        history = [
            {
                "assistant": assistant,
                "observation": tool_output_message(
                    "step_1",
                    {"table": "people", "rows": [{"name": "Older"}], "row_count": 1},
                ),
            },
            {
                "assistant": assistant,
                "observation": tool_output_message(
                    "step_2",
                    {"table": "people", "rows": [{"name": "Latest"}], "row_count": 1},
                ),
            },
        ]
        profile, latest_full = selected_resident_context("version47")
        messages = rolling_legal_history_messages(
            "system",
            {"tables": []},
            "question",
            _state(),
            None,
            None,
            history,
            4,
            resident_state_profile=profile,
            latest_observation_full=latest_full,
        )
        older_observation = messages[-3]["content"]
        latest_observation = messages[-1]["content"]
        self.assertNotIn("Older", older_observation)
        self.assertIn("output_summary", older_observation)
        self.assertIn("Latest", latest_observation)
        resident = latest_observation.split("CURRENT ENVIRONMENT STATE", 1)[1]
        self.assertNotIn("Alice", resident)
        self.assertIn("row_values", resident)

    def test_version49_keeps_active_dependency_rows_and_archives_inactive_branches(self) -> None:
        visible = model_visible_environment_state(
            _state(),
            RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
            active_relation_refs={"filter_001"},
        )
        focus = visible["relation_focus"]
        self.assertEqual(
            focus["active_dependency_closure"],
            ["people", "filter_001"],
        )
        self.assertEqual(focus["archived_relations"], ["other", "filter_002"])
        self.assertEqual(
            visible["tables"]["people"]["reads"][0]["rows"][0]["name"],
            "Alice",
        )
        self.assertEqual(
            visible["tables"]["filter_001"]["reads"][0]["rows"][0]["name"],
            "Alice",
        )
        self.assertNotIn("rows", visible["tables"]["other"]["reads"][0])
        self.assertNotIn("rows", visible["tables"]["filter_002"]["reads"][0])

        assistant = (
            "<think>Continue from the active result.</think>"
            '{"tool":"read_subtable","arguments":{"table":"filter_001","limit":1}}'
        )
        messages = rolling_legal_history_messages(
            "system",
            {"tables": []},
            "question",
            _state(),
            None,
            None,
            [{
                "assistant": assistant,
                "observation": tool_output_message(
                    "step_6",
                    {"table": "filter_001", "rows": [{"id": 1, "name": "Alice"}]},
                ),
            }],
            4,
            resident_state_profile=RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
        )
        resident = messages[-1]["content"].split("CURRENT ENVIRONMENT STATE", 1)[1]
        self.assertIn('"active_dependency_closure":["people","filter_001"]', resident)
        self.assertIn("Alice", resident)
        self.assertNotIn("Inactive source cell", resident)

    def test_version49_accepts_provider_raw_json_actions_as_active_refs(self) -> None:
        messages = rolling_legal_history_messages(
            "system",
            {"tables": []},
            "question",
            _state(),
            None,
            None,
            [{
                "assistant": json.dumps({
                    "tool": "read_subtable",
                    "arguments": {"table": "filter_001", "limit": 1},
                }),
                "observation": tool_output_message(
                    "step_6",
                    {"rows": [{"id": 1, "name": "Alice"}], "row_count": 1},
                ),
            }],
            4,
            resident_state_profile=RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
        )
        resident = messages[-1]["content"].split("CURRENT ENVIRONMENT STATE", 1)[1]
        self.assertIn('"active_dependency_closure":["people","filter_001"]', resident)
        self.assertIn("Alice", resident)
        self.assertNotIn("Inactive source cell", resident)


if __name__ == "__main__":
    unittest.main()
