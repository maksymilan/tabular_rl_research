from __future__ import annotations

import json

import pytest

from tool_modules.checkpoint_relalg.provider import (
    DeepSeekNativeClient,
    NativeAssistantResponse,
    ProviderCompletionTruncated,
    ProviderShapeError,
    _normalize_assistant_message,
    tool_result_message,
)


def _client() -> DeepSeekNativeClient:
    return DeepSeekNativeClient(
        api_key="test-only",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    )


def _response(*, calls: int = 1, finish_reason: str = "tool_calls", tokens: int = 7):
    raw_calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "describe_table", "arguments": '{"tables":["t"]}'},
        }
        for index in range(calls)
    ]
    message, normalized = _normalize_assistant_message(
        {
            "role": "assistant",
            "content": "audit-only",
            "reasoning_content": "inspect the schema",
            "tool_calls": raw_calls,
        }
    )
    return NativeAssistantResponse(
        message=message,
        calls=normalized,
        finish_reason=finish_reason,
        usage={"total_tokens": tokens},
    )


def test_official_deepseek_endpoint_is_mandatory():
    with pytest.raises(ValueError, match="api.deepseek.com"):
        DeepSeekNativeClient(
            api_key="test-only",
            base_url="https://example.invalid",
            model="renamed-model",
        )


def test_normalizer_preserves_reasoning_and_multiple_calls_for_runtime_rejection():
    response = _response(calls=2)
    assert response.message["reasoning_content"] == "inspect the schema"
    assert response.message["content"] == "audit-only"
    assert [call.call_id for call in response.calls] == ["call_0", "call_1"]


def test_argument_parser_never_repairs_non_object_json():
    message, calls = _normalize_assistant_message(
        {
            "role": "assistant",
            "reasoning_content": "submit the grounded table",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "answer", "arguments": "[]"},
                }
            ]
        }
    )
    assert message["tool_calls"]
    with pytest.raises(ProviderShapeError, match="object"):
        calls[0].parsed_arguments()


def test_bounded_zero_call_retry_accumulates_usage(monkeypatch):
    client = _client()
    responses = [_response(calls=0, tokens=3), _response(calls=1, tokens=5)]
    monkeypatch.setattr(client, "request_turn", lambda **_: responses.pop(0))
    result = client.request_turn_with_retries(
        messages=[], tools=[], max_tokens=128, retries=2
    )
    assert result.usage["total_tokens"] == 8
    assert result.retry_events == [{"type": "missing_tool_calls", "attempt": 1}]


def test_length_retry_is_bounded(monkeypatch):
    client = _client()
    monkeypatch.setattr(
        client,
        "request_turn",
        lambda **_: _response(calls=0, finish_reason="length"),
    )
    with pytest.raises(ProviderCompletionTruncated) as raised:
        client.request_turn_with_retries(
            messages=[], tools=[], max_tokens=128, retries=2, max_completion_tokens=256
        )
    assert raised.value.accumulated_usage["total_tokens"] == 14
    assert raised.value.retry_events


def test_exhausted_zero_call_retry_retains_usage_and_events(monkeypatch):
    client = _client()
    responses = [_response(calls=0, tokens=11), _response(calls=0, tokens=13)]
    monkeypatch.setattr(client, "request_turn", lambda **_: responses.pop(0))
    with pytest.raises(ProviderShapeError) as raised:
        client.request_turn_with_retries(
            messages=[], tools=[], max_tokens=128, retries=2
        )
    assert raised.value.accumulated_usage == {"total_tokens": 24}
    assert raised.value.retry_events == [
        {"type": "missing_tool_calls", "attempt": 1}
    ]


def test_normalizer_preserves_raw_role_type_and_extra_fields():
    raw = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "audit the call",
        "unexpected": "audit-me",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "answer", "arguments": '{"table":"x"}'},
            }
        ],
    }
    stored, _ = _normalize_assistant_message(raw)
    assert stored == raw


def test_shape_failure_before_normalization_retains_response_usage(monkeypatch):
    client = _client()
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "inspect",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "describe_table",
                                    "arguments": '{"tables":["t"]}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"total_tokens": 19},
        },
    )
    with pytest.raises(ProviderShapeError) as raised:
        client.request_turn_with_retries(
            messages=[], tools=[], max_tokens=128, retries=1
        )
    assert raised.value.accumulated_usage == {"total_tokens": 19}


@pytest.mark.parametrize(
    "patch",
    [
        {"role": "user"},
        {"reasoning_content": None},
        {"tool_calls": [{"id": "x", "type": None, "function": {"name": "answer", "arguments": "{}"}}]},
    ],
)
def test_transport_rejects_malformed_thinking_history_fields(patch):
    raw = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "answer", "arguments": '{"table":"x"}'},
            }
        ],
    }
    raw.update(patch)
    with pytest.raises(ProviderShapeError):
        _normalize_assistant_message(raw)


def test_tool_result_message_is_provider_valid_json():
    message = tool_result_message("call_7", {"status": "error", "value": None})
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_7"
    assert json.loads(message["content"])["status"] == "error"
