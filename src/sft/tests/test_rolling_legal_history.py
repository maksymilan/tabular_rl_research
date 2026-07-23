#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SFT_DIR.parents[0] / "harness"
sys.path[:0] = [str(SFT_DIR), str(HARNESS_DIR)]

from protocol import (  # noqa: E402
    SYSTEM_PROMPT,
    compact_resident_observation,
    rolling_legal_history_messages,
    rolling_system_prompt,
)


class RollingLegalHistoryTests(unittest.TestCase):
    def test_keeps_legal_assistant_actions_and_current_state(self):
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            {"tables": {"filter_001": {"row_count": 1}}, "plan": [], "values": {}},
            {"step_id": "step_3", "status": "error", "error": {"type": "protocol_error"}},
            None,
            [{"assistant": "<think>legal</think><tool_call>{}</tool_call>",
              "observation": '{"step_id":"step_1","status":"success"}'}],
            4,
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "<think>legal</think><tool_call>{}</tool_call>")
        self.assertIn('"step_id":"step_1"', messages[3]["content"])
        self.assertIn("CURRENT ENVIRONMENT STATE", messages[3]["content"])
        self.assertIn("LAST TOOL ERROR", messages[3]["content"])

    def test_history_window_retains_only_recent_legal_pairs(self):
        history = [
            {"assistant": "first", "observation": "first output"},
            {"assistant": "second", "observation": "second output"},
        ]
        messages = rolling_legal_history_messages(
            "system", {"tables": [], "relations": []}, "question", {}, None, None, history, 1,
        )
        contents = [message["content"] for message in messages]
        self.assertNotIn("first", contents)
        self.assertIn("second", contents)

    def test_rolling_system_prompt_explains_bounded_history(self):
        prompt = rolling_system_prompt("base")
        self.assertIn("bounded transcript", prompt)
        self.assertIn("CURRENT ENVIRONMENT STATE", prompt)

    def test_safe_compact_prompt_retains_critical_contract(self):
        full = rolling_system_prompt(SYSTEM_PROMPT)
        compact = rolling_system_prompt(SYSTEM_PROMPT, compact=True)
        self.assertLess(len(compact), len(full))
        for required in (
            "value_ref", "relation.column", "base_role", "LAST TOOL ERROR", "answer_from_context",
        ):
            self.assertIn(required, compact)

    def test_historical_rows_are_replaced_by_resident_state_pointer(self):
        state = {
            "plan": [],
            "tables": {
                "items": {
                    "reads": [{"from_step": "step_1", "rows": [[1, "large fact"]]}],
                },
            },
            "values": {},
        }
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            state,
            None,
            None,
            [{
                "assistant": "<think>read</think><tool_call>{}</tool_call>",
                "observation": (
                    '{"step_id":"step_1","status":"success","output":'
                    '{"table":"items","columns":["id","name"],"row_count":1,'
                    '"rows":[[1,"large fact"]]}}'
                ),
            }],
            4,
        )
        history_and_state = messages[-1]["content"]
        observation, current_state = history_and_state.split("\n\nCURRENT ENVIRONMENT STATE\n", 1)
        self.assertNotIn("large fact", observation)
        self.assertIn('"returned_row_count":1', observation)
        self.assertIn("resident_in_current_environment_state", observation)
        self.assertIn("large fact", current_state)

    def test_join_columns_are_compact_in_rolling_observation(self):
        observation = (
            '{"step_id":"step_2","status":"success","output":'
            '{"table":"join_001","kind":"join","columns":'
            '["orders.id","orders.customer_id","customers.id","customers.name"],'
            '"row_count":2}}'
        )
        compact = compact_resident_observation(observation)
        self.assertIn(
            '"column_namespaces":{"orders":["id","customer_id"],'
            '"customers":["id","name"]}',
            compact,
        )
        self.assertNotIn('"orders.id"', compact)

    def test_table_derivation_is_not_duplicated_in_compact_rolling_observation(self):
        observation = (
            '{"step_id":"step_2","status":"success","output":'
            '{"table":"group_001","kind":"group","columns":["team","n"],'
            '"row_count":2,"derivation":{"schema":"relation-derivation-v1",'
            '"operator":"group_aggregate","inputs":[{"kind":"table","role":"input",'
            '"ref":"items"}],"semantics":{"row_operation":"aggregate",'
            '"column_operation":"group_and_compute","row_grain":["team"],'
            '"layout":"rows","aggregations":[]}}}}'
        )
        compact = compact_resident_observation(observation)
        self.assertNotIn('"derivation"', compact)
        self.assertIn('"table":"group_001"', compact)
        self.assertIn("resident_in_current_environment_state", compact)

    def test_current_state_compacts_join_columns_without_changing_handle(self):
        state = {
            "plan": [],
            "tables": {
                "join_001": {
                    "kind": "join",
                    "created_by": "step_2",
                    "columns": [
                        "orders.id", "orders.customer_id", "customers.id", "customers.name",
                    ],
                    "row_count": 2,
                    "derivation": {
                        "schema": "relation-derivation-v1",
                        "operator": "join_tables",
                        "inputs": [
                            {"kind": "table", "role": "base", "ref": "orders"},
                            {"kind": "table", "role": "joined", "ref": "customers"},
                        ],
                        "semantics": {
                            "row_operation": "join",
                            "column_operation": "concatenate_namespaced",
                            "edges": [{
                                "index": 0,
                                "input": "customers",
                                "namespace": "customers",
                                "join_type": "inner",
                                "on": [{"left": "orders.customer_id", "right": "id"}],
                            }],
                        },
                    },
                },
            },
            "values": {},
        }
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            state,
            None,
            None,
            [],
            4,
        )
        rendered = messages[-1]["content"]
        self.assertIn('"join_001"', rendered)
        self.assertIn(
            '"column_namespaces":{"orders":["id","customer_id"],'
            '"customers":["id","name"]}',
            rendered,
        )
        self.assertNotIn('"orders.id"', rendered)
        self.assertIn('"schema":"relation-derivation-v1"', rendered)
        self.assertIn('"operator":"join_tables"', rendered)
        self.assertIn("columns", state["tables"]["join_001"])

    def test_plan_evidence_keeps_grounded_identity_without_duplicating_output(self):
        state = {
            "plan": [{
                "id": "inspect",
                "goal": "Inspect orders",
                "status": "done",
                "evidence": {
                    "step_id": "step_2",
                    "tool": "describe_table",
                    "output": {
                        "tables": [{
                            "table_name": "orders",
                            "columns": [{"name": "very_large_schema_column", "type": "TEXT"}],
                        }],
                    },
                },
            }],
            "tables": {},
            "values": {},
        }
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            state,
            None,
            None,
            [],
            4,
        )
        rendered = messages[-1]["content"]
        self.assertIn('"goal":"Inspect orders"', rendered)
        self.assertIn('"status":"done"', rendered)
        self.assertIn('"step_id":"step_2"', rendered)
        self.assertIn('"tool":"describe_table"', rendered)
        self.assertNotIn("very_large_schema_column", rendered)
        self.assertIn("very_large_schema_column", str(state))

    def test_full_observation_style_reproduces_pre_r2_history(self):
        observation = (
            '{"step_id":"step_1","status":"success","output":'
            '{"table":"items","rows":[[1,"historical fact"]]}}'
        )
        messages = rolling_legal_history_messages(
            "system",
            {"tables": [], "relations": []},
            "question",
            {"plan": [], "tables": {}, "values": {}},
            None,
            None,
            [{"assistant": "action", "observation": observation}],
            4,
            compact_observations=False,
        )
        self.assertIn('"historical fact"', messages[-1]["content"])
        self.assertNotIn("resident_in_current_environment_state", messages[-1]["content"])

    def test_schema_payload_is_summarized_without_losing_table_identity(self):
        observation = compact_resident_observation(
            '{"step_id":"step_2","status":"success","output":{"tables":['
            '{"table_name":"orders","row_count":12,"columns":['
            '{"name":"id","type":"INTEGER"},{"name":"amount","type":"REAL"}]}'
            ']}}'
        )
        self.assertIn('"table_name":"orders"', observation)
        self.assertIn('"column_count":2', observation)
        self.assertNotIn('"amount"', observation)

    def test_non_json_legacy_observation_is_not_repaired(self):
        self.assertEqual(compact_resident_observation("legacy output"), "legacy output")


if __name__ == "__main__":
    unittest.main()
