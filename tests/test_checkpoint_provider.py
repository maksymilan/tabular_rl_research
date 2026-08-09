from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from tool_modules.checkpoint_relalg.provider import (
    DeepSeekNativeClient,
    NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION,
    NativeAssistantResponse,
    ProviderCompletionTruncated,
    ProviderContextOverflow,
    ProviderError,
    ProviderShapeError,
    _normalize_assistant_message,
    provider_response_envelope_protocol_hash,
    provider_response_envelope_version,
    tool_result_message,
)
from tool_modules.checkpoint_relalg.protocol import (
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
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
            "index": index,
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
        usage={
            "prompt_tokens": tokens - 2,
            "completion_tokens": 2,
            "total_tokens": tokens,
        },
        response_metadata={"model": "deepseek-v4-flash"},
        raw_assistant_message=message,
    )


def test_official_deepseek_endpoint_is_mandatory():
    with pytest.raises(ValueError, match="api.deepseek.com"):
        DeepSeekNativeClient(
            api_key="test-only",
            base_url="https://example.invalid",
            model="renamed-model",
        )


def test_native_response_envelope_v2_is_carrier_specific_and_hash_bound():
    base_hash = "a" * 64
    assert (
        provider_response_envelope_version(CARRIER_NATIVE_TOOL_CALLS)
        == NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
    )
    assert provider_response_envelope_version(CARRIER_TEXT_JSON) is None
    assert (
        provider_response_envelope_protocol_hash(
            base_hash,
            CARRIER_NATIVE_TOOL_CALLS,
        )
        != base_hash
    )
    assert (
        provider_response_envelope_protocol_hash(base_hash, CARRIER_TEXT_JSON)
        == base_hash
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
    assert raised.value.accumulated_usage == {
        "completion_tokens": 4,
        "prompt_tokens": 20,
        "total_tokens": 24,
    }
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
            "model": "deepseek-v4-flash",
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
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 4,
                "total_tokens": 19,
            },
        },
    )
    with pytest.raises(ProviderShapeError) as raised:
        client.request_turn_with_retries(
            messages=[], tools=[], max_tokens=128, retries=1
        )
    assert raised.value.accumulated_usage == {
        "completion_tokens": 4,
        "prompt_tokens": 15,
        "total_tokens": 19,
    }


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


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (402, False),
        (403, False),
        (404, False),
        (422, False),
        (408, True),
        (409, True),
        (425, True),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_http_errors_expose_stable_status_and_retryability(
    monkeypatch,
    status,
    retryable,
):
    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            status,
            "test status",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"synthetic"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    request = urllib.request.Request("https://api.deepseek.com/chat/completions")
    with pytest.raises(ProviderError) as raised:
        _client()._request_json(request)
    assert raised.value.http_status == status
    assert raised.value.retryable is retryable
    assert "synthetic" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "expected_attempts", "retryable"),
    [
        (401, 1, False),
        (402, 1, False),
        (422, 1, False),
        (429, 4, True),
        (503, 4, True),
    ],
)
def test_http_retry_policy_is_bounded_and_deterministic_4xx_is_terminal(
    monkeypatch,
    status,
    expected_attempts,
    retryable,
):
    attempts = 0

    def fail(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            status,
            "test status",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"synthetic"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr(
        "tool_modules.checkpoint_relalg.provider.time.sleep",
        lambda _seconds: None,
    )
    with pytest.raises(ProviderError) as raised:
        _client().request_turn_with_retries(
            messages=[],
            tools=[],
            max_tokens=128,
            retries=4,
        )
    assert attempts == expected_attempts
    assert raised.value.provider_attempt_count == expected_attempts
    assert len(raised.value.retry_events) == expected_attempts - 1
    assert raised.value.http_status == status
    assert raised.value.retryable is retryable
    assert all(
        event["raw_assistant_message"] is None
        for event in raised.value.provider_attempt_events
    )
    assert "synthetic" not in str(raised.value)


def test_context_overflow_http_error_is_structured_and_never_retried(monkeypatch):
    attempts = 0

    def fail(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            400,
            "context overflow",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"maximum context length exceeded by max_tokens"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ProviderContextOverflow) as raised:
        _client().request_turn_with_retries(
            messages=[],
            tools=[],
            max_tokens=128,
            retries=4,
        )
    assert attempts == 1
    assert raised.value.provider_attempt_count == 1
    assert raised.value.retry_events == []
    assert raised.value.http_status == 400
    assert raised.value.retryable is False
