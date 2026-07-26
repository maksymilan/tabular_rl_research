#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "sft"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
]

from build_action_block_sft_data import (  # noqa: E402
    convert_turn,
    replay_episode,
    validate_source_episode,
)
from batch_plan_protocol import build_batch_plan_system_prompt  # noqa: E402
from tool_schemes import ACTION_BLOCK_TOOL_SCHEME  # noqa: E402


def source_record() -> dict:
    raw_block = (
        '{"tool":"action_block","arguments":{"calls":['
        '{"id":"exact","tool":"project","arguments":'
        '{"table":"items","expressions":["category"]}},'
        '{"id":"rows","tool":"read_subtable","arguments":'
        '{"table":"$exact","limit":20}}]}}'
    )
    raw_answer = (
        '{"tool":"answer_from_context","arguments":{"evidence":'
        '{"table":"project_001","columns":["category"]}}}'
    )
    system = build_batch_plan_system_prompt(8)
    return {
        "tool_scheme": ACTION_BLOCK_TOOL_SCHEME,
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "trajectory_id": "bird_train_00000",
        "example_index": 0,
        "db_id": "test",
        "question": "Return all categories.",
        "gold_sql": "SELECT category FROM items",
        "correct": True,
        "legal": True,
        "errors": 0,
        "error_events": [],
        "denotation_comparison": "bird-set",
        "sft_export_eligible": True,
        "turns": [
            {
                "model_input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "QUESTION\nReturn all categories."},
                ],
                "raw_model_output": raw_block,
                "canonical_model_output": (
                    "<think>Create and inspect the answer table.</think>\n"
                    + raw_block
                ),
                "provider_reasoning_content": "Create and inspect the answer table.",
                "parsed": json.loads(raw_block),
                "root_error_count": 0,
                "blocked_count": 0,
            },
            {
                "model_input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "QUESTION\nReturn all categories."},
                    {"role": "assistant", "content": raw_block},
                    {"role": "user", "content": "ACTION BLOCK RESULTS\n{}"},
                ],
                "raw_model_output": raw_answer,
                "canonical_model_output": (
                    "<think>Return the exact resident projection.</think>\n"
                    + raw_answer
                ),
                "provider_reasoning_content": "Return the exact resident projection.",
                "parsed": json.loads(raw_answer),
            },
        ],
        "atomic_events": [
            {
                "step_id": "step_1",
                "tool": "project",
                "status": "success",
                "arguments": {"table": "items", "expressions": ["category"]},
                "resolved_arguments": {
                    "table": "items",
                    "expressions": ["category"],
                },
            },
            {
                "step_id": "step_2",
                "tool": "read_subtable",
                "status": "success",
                "arguments": {"table": "$exact", "limit": 20},
                "resolved_arguments": {"table": "project_001", "limit": 20},
            },
            {
                "step_id": "step_3",
                "tool": "answer_from_context",
                "status": "success",
                "arguments": {
                    "evidence": {
                        "table": "project_001",
                        "columns": ["category"],
                    }
                },
            },
        ],
    }


class BuildActionBlockSftDataTests(unittest.TestCase):
    def test_conversion_translates_provider_history_to_student_carrier(self):
        record = source_record()
        validate_source_episode(record)
        converted, index = convert_turn(record, 1, max_batch_calls=8)
        self.assertEqual(converted["metadata"]["tool_scheme"], ACTION_BLOCK_TOOL_SCHEME)
        self.assertIn(
            "<think>Create and inspect the answer table.</think>",
            converted["conversations"][1]["value"],
        )
        self.assertIn(
            "<think>Return the exact resident projection.</think>",
            converted["conversations"][-1]["value"],
        )
        self.assertNotIn("<tool_call>", converted["conversations"][-1]["value"])
        self.assertEqual(index["tool_name"], "answer_from_context")

    def test_export_gate_rejects_unpromoted_or_error_episodes(self):
        record = source_record()
        record["sft_export_eligible"] = False
        with self.assertRaisesRegex(ValueError, "not explicitly promoted"):
            validate_source_episode(record)
        record = source_record()
        record["errors"] = 1
        with self.assertRaisesRegex(ValueError, "not SFT targets"):
            validate_source_episode(record)

    def test_fresh_replay_verifies_terminal_denotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                "CREATE TABLE items(category TEXT, price INTEGER);"
                "INSERT INTO items VALUES ('a', 1), ('b', 2);"
            )
            connection.close()
            replay_episode(
                source_record(),
                {
                    "db_id": "test",
                    "db_path": str(db_path),
                    "question": "Return all categories.",
                    "gold_sql": "SELECT category FROM items",
                },
            )


if __name__ == "__main__":
    unittest.main()
