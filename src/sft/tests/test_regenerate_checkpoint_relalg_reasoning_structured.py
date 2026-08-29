from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regenerate_checkpoint_relalg_reasoning_structured import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    decode_response,
    request_payload,
    split_causal_context,
    validate_reasoning,
)


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"\w+|[^\w\s]", text))))


def _reasoning() -> str:
    return (
        "filter_001 succeeded and contains the target population. The current decision is to retain "
        "rows whose score is above 5, so use filter_rows on filter_001 with score > 5."
    )


def test_teacher_and_student_prompts_are_explicitly_decoupled() -> None:
    assert "not the student's runtime prompt" in TEACHER_SYSTEM_PROMPT
    assert "original reasoning" in TEACHER_SYSTEM_PROMPT.lower()
    assert "three of these ideas in order" in TEACHER_SYSTEM_PROMPT
    assert "exactly one key, think" in TEACHER_SYSTEM_PROMPT
    assert "provider reasoning_content" not in TEACHER_SYSTEM_PROMPT


def test_request_omits_source_reasoning_and_disables_teacher_thinking() -> None:
    action = {
        "tool": "filter_rows",
        "arguments": {
            "table": "filter_001",
            "conditions": {"op": ">", "left": {"column": "score"}, "right": {"value": 5}},
        },
    }
    payload = request_payload(
        "deepseek-v4-flash",
        "QUESTION\nkeep high scores",
        action,
        1024,
        "disabled",
    )
    user = json.loads(payload["messages"][-1]["content"])
    assert set(user) == {
        "LATEST_HARNESS_FEEDBACK",
        "CURRENT_GOAL_AND_ENVIRONMENT",
        "CURRENT_STUDENT_TOOL_CONTRACT",
        "EXACT_NEXT_ACTION",
    }
    assert user["EXACT_NEXT_ACTION"] == action
    assert user["CURRENT_STUDENT_TOOL_CONTRACT"]["atomic_operator_profile"] == "atomic-v24-frozen-v1"
    assert user["CURRENT_STUDENT_TOOL_CONTRACT"]["exact_tool_definition"]["function"]["name"] == "filter_rows"
    assert "ORIGINAL_REASONING" not in user
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload
    assert "temperature" not in payload
    assert "tools" not in payload


def test_enabled_thinking_is_separate_from_formal_content() -> None:
    action = {
        "tool": "filter_rows",
        "arguments": {
            "table": "filter_001",
            "conditions": {"op": ">", "left": {"column": "score"}, "right": {"value": 5}},
        },
    }
    payload = request_payload(
        "deepseek-v4-flash",
        "QUESTION\nkeep high scores",
        action,
        1024,
        "enabled",
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload


def test_context_split_keeps_latest_feedback_separate() -> None:
    context = '{"type":"checkpoint_relalg_tool_result","result":{"status":"success"}}\n\nQUESTION\nQ\n\nCURRENT ENVIRONMENT STATE\nS'
    feedback, current = split_causal_context(context)
    assert "checkpoint_relalg_tool_result" in feedback
    assert current.startswith("QUESTION\nQ")
    first_feedback, first_current = split_causal_context("QUESTION\nQ")
    assert json.loads(first_feedback) == {"status": "no_previous_tool_feedback"}
    assert first_current == "QUESTION\nQ"


def test_strict_one_field_decode_returns_coherent_reasoning() -> None:
    raw = json.dumps({"think": _reasoning()})
    assert decode_response(raw) == _reasoning()
    with pytest.raises(ValueError):
        decode_response(json.dumps({"think": _reasoning(), "extra": "bad"}))


def test_quality_rejects_other_tools_and_accepts_grounded_three_parts() -> None:
    action = {
        "tool": "filter_rows",
        "arguments": {
            "table": "filter_001",
            "conditions": {"op": ">", "left": {"column": "score"}, "right": {"value": 5}},
        },
    }
    source = " ".join(["verbose"] * 300)
    context = "filter_001 target population score threshold 5"
    quality = validate_reasoning(_reasoning(), source, context, action, WordTokenizer())
    assert quality["regenerated_tokens"] < quality["source_tokens"]
    bad = _reasoning() + " Then use answer."
    with pytest.raises(ValueError, match="another tool"):
        validate_reasoning(bad, source, context, action, WordTokenizer())


def test_quality_rejects_three_part_template_narration() -> None:
    action = {
        "tool": "filter_rows",
        "arguments": {
            "table": "filter_001",
            "conditions": {"op": ">", "left": {"column": "score"}, "right": {"value": 5}},
        },
    }
    source = " ".join(["verbose"] * 300)
    templated = "The latest feedback indicates filter_001 succeeded. " + _reasoning()
    with pytest.raises(ValueError, match="templated"):
        validate_reasoning(templated, source, "filter_001 score 5", action, WordTokenizer())


def test_quality_rejects_invented_checkpoint_and_future_plan() -> None:
    action = {
        "tool": "filter_rows",
        "arguments": {
            "table": "filter_001",
            "conditions": {"op": ">", "left": {"column": "score"}, "right": {"value": 5}},
        },
    }
    source = " ".join(["verbose"] * 300)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_reasoning(
            _reasoning() + " The checkpoint confirms this.",
            source,
            "filter_001 score 5",
            action,
            WordTokenizer(),
        )
    with pytest.raises(ValueError, match="future"):
        validate_reasoning(
            _reasoning() + " A subsequent join can attach names.",
            source,
            "filter_001 score 5",
            action,
            WordTokenizer(),
        )
