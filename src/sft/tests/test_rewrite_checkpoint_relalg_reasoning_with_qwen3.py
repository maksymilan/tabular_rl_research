from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewrite_checkpoint_relalg_reasoning_with_qwen3 import (  # noqa: E402
    SYSTEM_PROMPT,
    decode_editor_response,
    request_payload,
    select_stratified,
    validate_cleaned_reasoning,
)


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"\w+|[^\w\s]", text))))


def _pair(record_id: str, tool: str, length: int, *, recovery: bool = False) -> tuple[dict, dict]:
    action = {"tool": tool, "arguments": {"table": f"table_{record_id}"}}
    row = {
        "conversations": [
            {"from": "human", "value": f"current context for {record_id}"},
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{'state ' * length}{tool} table_{record_id}\n</think>\n\n"
                    + json.dumps(action, separators=(",", ":"))
                ),
            },
        ],
        "metadata": {"record_id": record_id},
    }
    index = {
        "record_id": record_id,
        "source_episode_id": f"episode_{record_id}",
        "source_task_position": int(record_id.removeprefix("r")),
        "source_turn_index": 0,
        "tool_name": tool,
        "reasoning_characters": length,
        "feedback_recovery": recovery,
    }
    return row, index


def test_editor_prompt_has_adaptive_semantic_contract_without_sentence_limit() -> None:
    assert "Complex decisions may remain detailed" in SYSTEM_PROMPT
    assert "there is no sentence limit" in SYSTEM_PROMPT
    assert "begin from the established current state" in SYSTEM_PROMPT
    assert "restating the whole question" in SYSTEM_PROMPT
    assert '{"think":"cleaned reasoning"}' not in SYSTEM_PROMPT
    assert 'Never output a placeholder such as "cleaned reasoning"' in SYSTEM_PROMPT
    assert "full episode context is deliberately not included" in SYSTEM_PROMPT
    assert "1-3" not in SYSTEM_PROMPT
    assert "three sentences" not in SYSTEM_PROMPT.lower()


def test_decode_editor_response_is_strict_and_does_not_repair() -> None:
    assert decode_editor_response(' {"think":"Use filter_rows on people."} ') == "Use filter_rows on people."
    with pytest.raises(ValueError):
        decode_editor_response('prefix {"think":"Use filter_rows."}')
    with pytest.raises(ValueError):
        decode_editor_response('{"think":"Use filter_rows.","tool":"filter_rows"}')


def test_validation_allows_detailed_reasoning_when_grounded_and_nonrepetitive() -> None:
    source = " ".join(f"observed{i}" for i in range(180)) + " shape_rows source_table requested columns"
    cleaned = (
        "The current source_table already contains the filtered population. "
        "The requested columns are available, but their order still differs from the required output. "
        "Use shape_rows now to select and order those fields; no additional filtering is needed. "
        "This preserves the current row grain while making the next artifact answer-ready."
    )
    result = validate_cleaned_reasoning(
        cleaned,
        source,
        {"tool": "shape_rows", "arguments": {"table": "source_table"}},
        WordTokenizer(),
    )
    assert result["cleaned_tokens"] > 40
    assert result["cleaned_tokens"] < result["source_tokens"]
    assert result["action_anchor_present"] is True


def test_request_payload_preserves_exact_action_and_disables_model_thinking() -> None:
    action = {"tool": "answer", "arguments": {"table": "shape_004"}}
    payload = request_payload("local-qwen", "reasoning", action, 1024)
    quoted = json.loads(payload["messages"][-1]["content"])
    assert quoted["EXACT_NEXT_ACTION"] == action
    assert quoted["ORIGINAL_REASONING"] == "reasoning"
    assert "CURRENT_VISIBLE_CONTEXT" not in quoted
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {"type": "json_object"}


def test_stratified_selection_covers_tools_lengths_and_recovery() -> None:
    pairs = [
        _pair(f"r{offset}", tool, length, recovery=(offset % 7 == 0))
        for offset, (tool, length) in enumerate(
            [(tool, length) for tool in ("answer", "filter_rows", "shape_rows") for length in range(1, 13)]
        )
    ]
    rows = [pair[0] for pair in pairs]
    indexes = [pair[1] for pair in pairs]
    selected = select_stratified(rows, indexes, 12)
    assert len(selected) == 12
    assert {item[1]["tool_name"] for item in selected} == {"answer", "filter_rows", "shape_rows"}
    assert sum(item[1]["feedback_recovery"] for item in selected) >= 1
    assert len({item[1]["record_id"] for item in selected}) == 12
