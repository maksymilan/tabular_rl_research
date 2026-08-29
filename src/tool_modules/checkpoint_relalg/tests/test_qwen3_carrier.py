from __future__ import annotations

import pytest

from tool_modules.checkpoint_relalg.qwen3_carrier import (
    Qwen3ActionError,
    decode_qwen3_action,
    parse_qwen3_action,
    provider_student_system_prompt,
    qwen3_history_assistant_text,
    qwen3_student_system_prompt,
    render_qwen3_action,
)


def test_qwen3_prompt_changes_only_the_transport_contract() -> None:
    provider = provider_student_system_prompt()
    qwen3 = qwen3_student_system_prompt()

    assert "provider reasoning_content" in provider
    assert "provider reasoning_content" not in qwen3
    assert "QWEN3 INLINE ACTION CARRIER" in qwen3
    assert provider.count("describe_table") == qwen3.count("describe_table")
    assert provider.count("shape_rows") == qwen3.count("shape_rows")
    assert provider.count("commit_checkpoint") == qwen3.count("commit_checkpoint") == 0


def test_render_and_parse_qwen3_action_round_trip() -> None:
    text = render_qwen3_action(
        "The exact resident artifact already answers the question.",
        '{"tool":"answer","arguments":{"table":"shape_001"}}',
    )
    parsed = parse_qwen3_action(text)

    assert parsed["reasoning"].startswith("The exact resident artifact")
    assert parsed["tool"] == "answer"
    assert parsed["arguments"] == {"table": "shape_001"}
    assert parsed["assistant_text"] == text
    assert qwen3_history_assistant_text(text) == (
        '{"tool":"answer","arguments":{"table":"shape_001"}}'
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"tool":"answer","arguments":{"table":"x"}}',
        '<think></think>{"tool":"answer","arguments":{"table":"x"}}',
        '<think>one</think><think>two</think>{"tool":"answer","arguments":{"table":"x"}}',
        '<think>reason</think> prose {"tool":"answer","arguments":{"table":"x"}}',
        '<think>reason</think>```json\n{"tool":"answer","arguments":{"table":"x"}}\n```',
        '<think>reason</think>[{"tool":"answer","arguments":{"table":"x"}}]',
    ],
)
def test_qwen3_carrier_never_repairs_invalid_text(text: str) -> None:
    with pytest.raises(Qwen3ActionError):
        parse_qwen3_action(text)


def test_decode_qwen3_action_preserves_authored_invalid_arguments_for_harness() -> None:
    decoded = decode_qwen3_action(
        '<think>Try the authored call.</think>{"tool":"answer","arguments":{"bad":1}}'
    )
    assert decoded["raw_action"] == {"tool": "answer", "arguments": {"bad": 1}}
    with pytest.raises(Qwen3ActionError):
        parse_qwen3_action(decoded["assistant_text"])


def test_qwen3_parser_reuses_atomic_v24_action_validation() -> None:
    with pytest.raises(Qwen3ActionError) as error:
        parse_qwen3_action('<think>reason</think>{"tool":"execute_sql","arguments":{"sql":"select 1"}}')

    assert error.value.code == "tool_not_available_in_mode"

