#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_DIR))

from protocol import ProtocolError, parse_assistant, parse_assistant_strict  # noqa: E402


class ProtocolParseTests(unittest.TestCase):
    def test_parse_standard_tool_call(self):
        think, tool, args = parse_assistant(
            '<think>Count rows.</think>\n'
            '<tool_call>{"tool":"group_aggregate","arguments":{"table":"items","group_by":[],"aggregations":[{"op":"count","column":"*","as":"count_1"}]}}</tool_call>'
        )

        self.assertEqual(think, "Count rows.")
        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["aggregations"][0]["op"], "count")

    def test_parse_answer_shorthand_missing_tool_key(self):
        _, tool, args = parse_assistant(
            '<tool_call>{"answer_from_context","arguments":{"answer":[[1]],'
            '"evidence":{"table":"project_001"},"reason":"done"}}</tool_call>'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["answer"], [[1]])

    def test_parse_answer_key_shorthand(self):
        _, tool, args = parse_assistant(
            '<tool_call>{"answer_from_context":{"answer":[[1]],'
            '"evidence":{"table":"project_001"},"reason":"done"}}</tool_call>'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"]["table"], "project_001")

    def test_parse_balanced_json_without_closing_tag(self):
        _, tool, args = parse_assistant(
            '<tool_call>{"tool":"group_aggregate","arguments":{"table":"items","group_by":[],"aggregations":[{"op":"count","column":"*","as":"count_1"}]}}'
        )

        self.assertEqual(tool, "group_aggregate")
        self.assertEqual(args["table"], "items")

    def test_legacy_aggregate_is_accepted_for_compatibility(self):
        _, tool, args = parse_assistant(
            '<tool_call>{"tool":"aggregate","arguments":{"table":"items","column":"*","op":"count"}}</tool_call>'
        )

        self.assertEqual(tool, "aggregate")
        self.assertEqual(args["op"], "count")

    def test_answer_allows_evidence_string_and_missing_answer(self):
        _, tool, args = parse_assistant(
            '<tool_call>{"tool":"answer_from_context","arguments":{"evidence":"project_001"}}</tool_call>'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["evidence"], {"table": "project_001"})
        self.assertEqual(args["answer"], [])

    def test_repairs_truncated_long_answer_with_cited_evidence(self):
        _, tool, args = parse_assistant(
            "<think>The evidence table project_002 contains the rows.</think>\n"
            '<tool_call>{"tool":"answer_from_context","arguments":{"answer":[[1],[2],[3]'
        )

        self.assertEqual(tool, "answer_from_context")
        self.assertEqual(args["answer"], [])
        self.assertEqual(args["evidence"]["table"], "project_002")

    def test_truncated_json_stays_protocol_error(self):
        with self.assertRaises(ProtocolError):
            parse_assistant(
                '<tool_call>{"tool":"answer_from_context","arguments":{"answer":[[1]],'
            )

    def test_strict_parser_rejects_replay_only_forms(self):
        with self.assertRaises(ProtocolError):
            parse_assistant_strict(
                '<think>Count.</think><tool_call>{"tool":"aggregate",'
                '"arguments":{"table":"items","column":"*","op":"count"}}</tool_call>'
            )
        with self.assertRaises(ProtocolError):
            parse_assistant_strict(
                '<think>Join.</think><tool_call>{"tool":"join_tables",'
                '"arguments":{"left":"a","right":"b","on":[{"left":"id","right":"id"}]}}</tool_call>'
            )

    def test_strict_parser_rejects_unbounded_read_limit(self):
        with self.assertRaisesRegex(ProtocolError, "integer from 1 to 20"):
            parse_assistant_strict(
                '<think>Read everything.</think><tool_call>{"tool":"read_subtable",'
                '"arguments":{"table":"items","limit":68}}</tool_call>'
            )
        _, tool, args = parse_assistant_strict(
            '<think>Read a bounded sample.</think><tool_call>{"tool":"read_subtable",'
            '"arguments":{"table":"items","limit":20}}</tool_call>'
        )
        self.assertEqual(tool, "read_subtable")
        self.assertEqual(args["limit"], 20)


if __name__ == "__main__":
    unittest.main()
