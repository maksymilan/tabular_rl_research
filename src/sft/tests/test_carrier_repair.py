from __future__ import annotations

import unittest

from carrier_repair import (
    assistant_tool_name,
    canonical_boundary_token_ids,
    encode_last_assistant_target,
    select_tool_balanced_records,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = []
        position = 0
        boundaries = {"<tool_call>": 90, "</tool_call>": 91}
        while position < len(text):
            match = next(
                (
                    (boundary, token_id)
                    for boundary, token_id in boundaries.items()
                    if text.startswith(boundary, position)
                ),
                None,
            )
            if match is None:
                ids.append(ord(text[position]))
                position += 1
            else:
                boundary, token_id = match
                ids.append(token_id)
                position += len(boundary)
        return ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize
        rendered = "".join(
            f"[{message['role']}]{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "[assistant]"
        return rendered


def record(tool: str, suffix: str = ""):
    return {
        "system": "system",
        "conversations": [
            {"from": "human", "value": f"question{suffix}"},
            {
                "from": "gpt",
                "value": (
                    "<think>x</think>\n"
                    f'<tool_call>{{"tool":"{tool}","arguments":{{}}}}</tool_call>'
                ),
            },
        ],
    }


class CarrierRepairTest(unittest.TestCase):
    def test_boundary_tokens_must_be_unique_single_tokens(self):
        self.assertEqual(
            canonical_boundary_token_ids(FakeTokenizer()),
            {"<tool_call>": 90, "</tool_call>": 91},
        )

    def test_tool_balanced_selection_round_robins(self):
        records = [
            record("project", "1"),
            record("project", "2"),
            record("project", "3"),
            record("join_tables", "1"),
        ]
        selected = select_tool_balanced_records(records, limit=2, seed=7)
        self.assertEqual(
            {assistant_tool_name(item) for item in selected},
            {"project", "join_tables"},
        )

    def test_only_last_assistant_turn_is_supervised(self):
        tokenizer = FakeTokenizer()
        encoded = encode_last_assistant_target(
            record("describe_table"),
            tokenizer,
            cutoff_len=1000,
            required_token_ids=(90, 91),
        )
        first_label = next(
            index for index, value in enumerate(encoded["labels"]) if value != -100
        )
        self.assertGreater(first_label, 0)
        self.assertEqual(encoded["labels"].count(90), 1)
        self.assertEqual(encoded["labels"].count(91), 1)


if __name__ == "__main__":
    unittest.main()
