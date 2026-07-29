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
from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    parse_assistant_strict as parse_version39,
    rolling_legal_history_messages,
)
from provider_adapter import provider_request_messages  # noqa: E402
from public_tool_contract import (  # noqa: E402
    PUBLIC_TOOL_CONTRACTS,
    VERSION40_PUBLIC_TOOL_CONTRACTS,
)
from rollout import execute_tool, new_ctx  # noqa: E402


class AtomicVersion40Tests(unittest.TestCase):
    def test_public_surface_removes_plan_renames_row_inspection_and_keeps_join(self):
        self.assertEqual(version40.PROTOCOL_VERSION, "version40")
        self.assertNotIn("plan", version40.TOOLS)
        self.assertNotIn("read_subtable", version40.TOOLS)
        self.assertIn("inspect_rows", version40.TOOLS)
        self.assertEqual(
            VERSION40_PUBLIC_TOOL_CONTRACTS["join_tables"],
            PUBLIC_TOOL_CONTRACTS["join_tables"],
        )

    def test_layered_prompt_is_substantive_without_old_duplicate_contracts(self):
        prompt = version40.provider_system_prompt("deepseek-v4-flash")
        headings = (
            "BACKGROUND",
            "TASK",
            "ENVIRONMENT",
            "TOOLS",
            "ARGUMENT-SHAPE EXAMPLES",
            "REASONING CONTINUITY",
            "RULES",
            "RESPONSE CONTRACT",
        )
        positions = [prompt.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(len(prompt), 6000)
        self.assertLess(len(prompt), 10000)
        self.assertNotIn("plan(", prompt)
        self.assertNotIn("read_subtable", prompt)
        self.assertEqual(prompt.count("EXTERNAL KNOWLEDGE"), 1)
        self.assertEqual(prompt.count("REASONING CONTINUITY"), 1)
        self.assertNotIn("under 120 words", prompt)
        self.assertIn("two to six substantive sentences", prompt)
        self.assertIn("at most the four most recent", prompt)

    def test_version40_parser_accepts_only_the_new_public_row_name(self):
        inspect_rows = (
            "<think>Inspect the next grounded rows.</think>\n"
            '{"tool":"inspect_rows","arguments":{"table":"items","limit":5}}'
        )
        _, tool, arguments = version40.parse_assistant_strict(inspect_rows)
        self.assertEqual(tool, "inspect_rows")
        self.assertEqual(arguments, {"table": "items", "limit": 5})
        with self.assertRaises(ProtocolError):
            parse_version39(inspect_rows)
        for retired_tool, arguments in (
            ("read_subtable", {"table": "items", "limit": 5}),
            ("plan", {"ops": []}),
        ):
            text = (
                "<think>Try an unavailable action.</think>\n"
                f'{{"tool":"{retired_tool}","arguments":'
                + json.dumps(arguments, separators=(",", ":"))
                + "}"
            )
            with self.assertRaises(ProtocolError):
                version40.parse_assistant_strict(text)

    def test_recent_four_preserves_complete_reasoning_calls_and_observations(self):
        history = []
        for index in range(5):
            history.append({
                "assistant": (
                    f"<think>complete reasoning {index} with an unfinished hypothesis</think>\n"
                    '{"tool":"describe_table","arguments":{"tables":["items"]}}'
                ),
                "observation": (
                    f'{{"step_id":"step_{index}","status":"success",'
                    f'"rows":[["full-row-{index}"]]}}'
                ),
            })
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            {},
            None,
            None,
            history,
            4,
            compact_observations=False,
        )
        rendered = provider_request_messages(
            "deepseek-v4-flash",
            messages,
            preserve_reasoning=True,
        )
        assistants = [item for item in rendered if item["role"] == "assistant"]
        observations = [item["content"] for item in rendered if item["role"] == "user"][1:]
        self.assertEqual(len(assistants), 4)
        self.assertEqual(
            [item["reasoning_content"] for item in assistants],
            [
                f"complete reasoning {index} with an unfinished hypothesis"
                for index in range(1, 5)
            ],
        )
        self.assertNotIn("<think>", assistants[0]["content"])
        for index in range(1, 5):
            self.assertIn(f"full-row-{index}", observations[index - 1])

    def test_inspect_rows_executes_same_read_only_operation_under_public_name(self):
        harness = Harness(":memory:")
        self.addCleanup(harness.conn.close)
        harness.conn.executescript(
            """
            CREATE TABLE items(id INTEGER, label TEXT);
            INSERT INTO items VALUES (1, 'a'), (2, 'b'), (3, 'c');
            """
        )
        harness.register_sources()
        context = new_ctx()
        output, created = execute_tool(
            harness,
            "inspect_rows",
            {
                "table": "items",
                "columns": ["id", "label"],
                "order_by": ["id DESC"],
                "limit": 2,
            },
            context,
            "step_1",
        )
        self.assertIsNone(created)
        self.assertEqual(output["rows"], [[3, "c"], [2, "b"]])
        self.assertEqual(context["history"]["step_1"]["tool"], "inspect_rows")
        reads = context["environment"].snapshot()["tables"]["items"]["reads"]
        self.assertEqual(reads[0]["from_step"], "step_1")
        self.assertEqual(reads[0]["order_by"], ["id DESC"])


if __name__ == "__main__":
    unittest.main()
