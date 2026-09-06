#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

import atomic_version40 as version40  # noqa: E402
import atomic_version41 as version41  # noqa: E402
from atomic_version41_prompt import (  # noqa: E402
    ERROR_CORRECTION_ACTIONS,
    OUTPUT_CONTRACT,
    correction_action,
)
from generate_teacher_rollouts import (  # noqa: E402
    ATOMIC_PROTOCOL_VERSIONS,
    selected_protocol_hash,
    selected_tool_schema_hash,
)


class AtomicVersion41Tests(unittest.TestCase):
    def test_version41_is_prompt_only_over_version40(self):
        self.assertEqual(version41.PROTOCOL_VERSION, "version41")
        self.assertEqual(version41.TOOLS, version40.TOOLS)
        self.assertEqual(version41.TOOL_SPECS, version40.TOOL_SPECS)
        self.assertEqual(version41.MODEL_ARG_SCHEMA, version40.MODEL_ARG_SCHEMA)
        self.assertEqual(version41.tool_schema_hash(), version40.tool_schema_hash())

    def test_output_contract_is_consolidated_and_preserves_old_discriminators(self):
        prompt = version41.provider_system_prompt("deepseek-v4-flash")
        self.assertEqual(prompt.count("OUTPUT CONTRACT"), 1)
        self.assertEqual(prompt.count(OUTPUT_CONTRACT), 1)
        self.assertIn("Keep separate source fields in separate columns", prompt)
        self.assertIn("Do not concatenate names", prompt)
        self.assertIn("replace an ID/code with a label/name", prompt)
        self.assertIn("return_columns", prompt)
        self.assertIn("exact terminal columns and order", prompt)
        self.assertNotIn(
            "The terminal evidence table must itself contain exactly the requested",
            prompt,
        )
        self.assertNotIn(
            "remove helper columns before citing terminal evidence",
            prompt,
        )
        self.assertNotIn("plan(", prompt)
        self.assertNotIn("read_subtable", prompt)

    def test_gate50_error_tools_have_valid_correction_examples(self):
        covered = {action["tool"] for action in ERROR_CORRECTION_ACTIONS}
        self.assertEqual(
            covered,
            {
                "condition_filter",
                "extreme_value_select",
                "group_aggregate",
                "scalar_compute",
                "inspect_rows",
            },
        )
        prompt = version41.provider_system_prompt("deepseek-v4-flash")
        self.assertEqual(prompt.count("ERROR-CORRECTION EXAMPLES"), 1)
        for action in ERROR_CORRECTION_ACTIONS:
            model_action = correction_action(action)
            version41.parse_assistant_strict(
                "<think>Revise the rejected argument shape from structured feedback.</think>"
                + json.dumps(model_action, separators=(",", ":"))
            )
            self.assertIn(action["error"], prompt)

    def test_prompt_remains_bounded_and_protocol_hash_changes(self):
        version40_prompt = version40.provider_system_prompt("deepseek-v4-flash")
        version41_prompt = version41.provider_system_prompt("deepseek-v4-flash")
        self.assertGreater(len(version41_prompt), len(version40_prompt))
        self.assertLess(len(version41_prompt), 11000)
        self.assertNotEqual(
            version41.protocol_hash(version41_prompt),
            version40.protocol_hash(version40_prompt),
        )

    def test_teacher_rollout_registry_selects_version41_hashes(self):
        prompt = version41.provider_system_prompt("deepseek-v4-flash")
        self.assertIn("version41", ATOMIC_PROTOCOL_VERSIONS)
        self.assertEqual(
            selected_tool_schema_hash("version41"),
            version41.tool_schema_hash(),
        )
        self.assertEqual(
            selected_protocol_hash("version41", prompt),
            version41.protocol_hash(prompt),
        )


if __name__ == "__main__":
    unittest.main()
