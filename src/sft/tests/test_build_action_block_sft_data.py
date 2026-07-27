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
    is_sft_target_turn,
    replay_episode,
    validate_source_episode,
)
from batch_plan_protocol import (  # noqa: E402
    UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
    build_batch_plan_system_prompt,
)
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
        '{"table":"project_001"}}}'
    )
    system = build_batch_plan_system_prompt(5)
    return {
        "tool_scheme": ACTION_BLOCK_TOOL_SCHEME,
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "protocol_version": UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
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
                    }
                },
            },
        ],
    }


def mixed_terminal_source_record() -> dict:
    record = source_record()
    raw_answer = (
        '{"tool":"action_block","arguments":{"calls":['
        '{"id":"final","tool":"answer_from_context","arguments":{"evidence":'
        '{"table":"$exact"}}},'
        '{"id":"exact","tool":"project","arguments":'
        '{"table":"items","expressions":["category"]}}]}}'
    )
    record["turns"] = [{
        "model_input": record["turns"][0]["model_input"],
        "raw_model_output": raw_answer,
        "canonical_model_output": (
            "<think>Derive and return the exact answer.</think>\n" + raw_answer
        ),
        "provider_reasoning_content": "Derive and return the exact answer.",
        "parsed": json.loads(raw_answer),
        "root_error_count": 0,
        "blocked_count": 0,
    }]
    record["atomic_events"] = [
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
            "tool": "answer_from_context",
            "status": "success",
            "arguments": {
                "evidence": {
                    "table": "$exact",
                }
            },
            "resolved_evidence_arguments": {
                "evidence": {
                    "table": "project_001",
                }
            },
        },
    ]
    return record


class BuildActionBlockSftDataTests(unittest.TestCase):
    def test_conversion_translates_provider_history_to_student_carrier(self):
        record = source_record()
        validate_source_episode(record)
        converted, index = convert_turn(record, 1, max_batch_calls=5)
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
        self.assertEqual(index["atomic_calls"], 1)

    def test_export_gate_rejects_unpromoted_but_allows_recovery_episodes(self):
        record = source_record()
        record["sft_export_eligible"] = False
        with self.assertRaisesRegex(ValueError, "not explicitly promoted"):
            validate_source_episode(record)
        record = source_record()
        record["errors"] = 1
        record["error_events"] = [{"error_type": "argument_validation_error"}]
        validate_source_episode(record)

        failed_turn = {
            "parsed": {"tool": "action_block", "arguments": {"calls": []}},
            "provider_reasoning_content": "A failed attempt.",
            "root_error_count": 1,
            "blocked_count": 1,
        }
        self.assertFalse(is_sft_target_turn(failed_turn))
        self.assertTrue(is_sft_target_turn(source_record()["turns"][0]))

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

    def test_mixed_terminal_sink_is_not_exportable(self):
        record = mixed_terminal_source_record()
        with self.assertRaisesRegex(ValueError, "standalone terminal action"):
            validate_source_episode(record)


if __name__ == "__main__":
    unittest.main()
