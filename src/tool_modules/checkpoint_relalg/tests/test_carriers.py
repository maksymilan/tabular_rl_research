from __future__ import annotations

import json
import sqlite3
import hashlib
from copy import deepcopy

import pytest

from src.tool_modules.checkpoint_relalg.protocol import (
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
    NATIVE_ASSISTANT_CARRIER,
    TEXT_JSON_ASSISTANT_CARRIER,
    capability_manifest,
    carrier_experiment_arm,
    get_system_prompt,
    prompt_hash,
    provider_tool_definitions,
)
from src.tool_modules.checkpoint_relalg.audit import (
    _manifest_binding_report,
    audit_record,
    audit_result_dir,
    provider_history_issues,
)
from src.tool_modules.checkpoint_relalg.provider import (
    DeepSeekNativeClient,
    NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION,
    NativeAssistantResponse,
    NativeToolCall,
    OFFICIAL_DEEPSEEK_BASE_URL,
    ProviderContentFiltered,
    ProviderInsufficientSystemResource,
    ProviderModelMismatch,
    ProviderShapeError,
    _normalize_assistant_message,
    provider_request_audit_options,
)
from tool_modules.checkpoint_relalg.provider import (
    ProviderModelMismatch as RunnerProviderModelMismatch,
)
from src.tool_modules.checkpoint_relalg.provider_tools import (
    NativeToolCallError,
    native_carrier_envelope_valid,
    validate_native_assistant_message,
)
from src.tool_modules.checkpoint_relalg.text_json_carrier import (
    TextJSONActionError,
    decode_text_json_result_message,
    text_json_carrier_envelope_valid,
    text_json_result_message,
    validate_text_json_assistant_message,
)
from src.tool_modules.registry import build_checkpoint_relalg_tool_scheme
from src.tool_modules.checkpoint_relalg.runner import run_episode
from src.tool_modules.checkpoint_relalg.runtime import RuntimeConfig


class _QueuedClient(DeepSeekNativeClient):
    def __init__(self, carrier: str, responses: list[dict]):
        super().__init__(
            api_key="test-key",
            base_url=OFFICIAL_DEEPSEEK_BASE_URL,
            model="deepseek-v4-flash",
            carrier=carrier,
        )
        self.responses = [deepcopy(item) for item in responses]
        self.payloads: list[dict] = []

    def _request_json(self, request, *, timeout=None):  # noqa: ANN001, ARG002
        self.payloads.append(json.loads(request.data))
        return self.responses.pop(0)


class _EpisodeClient:
    def __init__(self, carrier: str, message: dict):
        self.carrier = carrier
        self.message = message
        self.requests: list[list[dict]] = []
        self.request_audit_options = provider_request_audit_options(
            base_url=OFFICIAL_DEEPSEEK_BASE_URL,
            model="deepseek-v4-flash",
            carrier=carrier,
        )

    def request_turn_with_retries(self, **kwargs):  # noqa: ANN003, ARG002
        self.requests.append(deepcopy(kwargs["messages"]))
        calls: tuple[NativeToolCall, ...] = ()
        if self.carrier == CARRIER_NATIVE_TOOL_CALLS:
            raw = self.message["tool_calls"][0]
            calls = (NativeToolCall(
                raw["id"],
                raw["function"]["name"],
                raw["function"]["arguments"],
            ),)
        usage = {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
        envelope_hash = hashlib.sha256(json.dumps(
            self.message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")).hexdigest()
        return NativeAssistantResponse(
            message=deepcopy(self.message),
            calls=calls,
            finish_reason=("tool_calls" if calls else "stop"),
            usage=usage,
            response_metadata={"model": "deepseek-v4-flash"},
            raw_assistant_message=deepcopy(self.message),
            retry_events=[],
            provider_attempt_count=1,
            provider_elapsed_seconds=0.125,
            provider_attempt_events=[{
                "attempt_index": 1,
                "max_tokens": kwargs["max_tokens"],
                "finish_reason": "tool_calls" if calls else "stop",
                "usage": usage,
                "shape_category": "accepted_response",
                "response_envelope_sha256": envelope_hash,
                "response_model": "deepseek-v4-flash",
                "raw_assistant_message": deepcopy(self.message),
                "elapsed_seconds_from_request_start": 0.125,
            }],
        )


class _FailureClient:
    carrier = CARRIER_TEXT_JSON
    request_audit_options = provider_request_audit_options(
        base_url=OFFICIAL_DEEPSEEK_BASE_URL,
        model="deepseek-v4-flash",
        carrier=CARRIER_TEXT_JSON,
    )

    def request_turn_with_retries(self, **kwargs):  # noqa: ANN003, ARG002
        usage = {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19}
        raise RunnerProviderModelMismatch(
            "provider response model mismatch",
            accumulated_usage=usage,
            retry_events=[],
            provider_attempt_count=1,
            provider_elapsed_seconds=0.1,
            provider_attempt_events=[{
                "attempt_index": 1,
                "max_tokens": kwargs["max_tokens"],
                "finish_reason": None,
                "usage": usage,
                "shape_category": "model_mismatch",
                "response_envelope_sha256": None,
                "response_model": "deepseek-pro",
                "raw_assistant_message": None,
                "elapsed_seconds_from_request_start": 0.1,
            }],
            response_model="deepseek-pro",
        )


def _native_message(tool: str = "describe_table", arguments: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "I need the schema.",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "index": 0,
            "type": "function",
            "function": {
                "name": tool,
                "arguments": json.dumps(
                    arguments if arguments is not None else {"tables": ["people"]}
                ),
            },
        }],
    }


def _text_message(tool: str = "describe_table", arguments: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "I need the schema.",
        "content": json.dumps({
            "tool": tool,
            "arguments": arguments if arguments is not None else {"tables": ["people"]},
        }),
    }


def _response(message: dict, *, finish_reason: str = "tool_calls", tokens: int = 11) -> dict:
    return {
        "id": "resp_1",
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": tokens - 2,
            "completion_tokens": 2,
            "total_tokens": tokens,
        },
    }


def test_frozen_native_prompt_hashes_and_scheme_identity() -> None:
    expected = {
        "direct": (
            "87d47b6f4e461a2a53d0d216f5f1013a79222b87502a2f5013506d901a28d13a",
            "adc69b25324931f0cc812420d18907a04c2ca7080078dc588a119440e6de5e39",
            "de9f4b37c5a00f0c48505f6aa782dd20e72d733c84ec9424e124e0531d547fe0",
        ),
        "atomic": (
            "f96420049880be892156020872fab5cf9c40e33e36c8d9d5069797f707899b0a",
            "aa1310a40ee6e2051d1b789f7386c11506afa30621df10f6e77a19554ba257b0",
            "f9f55ffe3d49d7b77026a5ff2e0b423bc65be7368592534826801fc14eb05d07",
        ),
        "hybrid": (
            "4141bb734d9d6e3833f76d0cc98e4d0294ddcb4dad339da4af830cda46b0f033",
            "a56411e22781add41f3768ce2a5d5a1ef0ee937829157184aa8eb7ec0d2ad72f",
            "c4870af823f7ef6b4092493f7ab43991b4a3d4331ab48d4dd06d6f4f0cd9100f",
        ),
    }
    for mode, (student, teacher, protocol_identity) in expected.items():
        assert prompt_hash(mode) == student
        assert prompt_hash(mode, teacher=True) == teacher
        default = build_checkpoint_relalg_tool_scheme(mode=mode)
        explicit = build_checkpoint_relalg_tool_scheme(
            mode=mode,
            carrier=CARRIER_NATIVE_TOOL_CALLS,
        )
        assert default == explicit
        assert default.assistant_carrier == NATIVE_ASSISTANT_CARRIER
        assert default.protocol_hash == protocol_identity
        assert (
            default.provider_response_envelope_version
            == NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
        )
        assert (
            default.manifest_fields()["provider_response_envelope_version"]
            == NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
        )


def test_text_prompt_has_exact_parameter_schemas_without_function_wrapper() -> None:
    prompt = get_system_prompt("direct", teacher=True, carrier=CARRIER_TEXT_JSON)
    assert "TEXT-JSON CARRIER (diagnostic-only)" in prompt
    assert "Make exactly one native tool call per turn." not in prompt
    schema = json.loads(prompt.split("EXACT TOOL SCHEMAS FOR DIRECT MODE\n", 1)[1])
    expected = [item["function"] for item in provider_tool_definitions("direct")]
    assert schema == expected
    assert all(set(item) == {"name", "description", "parameters"} for item in schema)
    text_scheme = build_checkpoint_relalg_tool_scheme(
        mode="direct",
        carrier=CARRIER_TEXT_JSON,
    )
    assert text_scheme.assistant_carrier == TEXT_JSON_ASSISTANT_CARRIER
    assert text_scheme.provider_response_envelope_version is None
    assert "provider_response_envelope_version" not in text_scheme.manifest_fields()
    assert text_scheme.protocol_hash != build_checkpoint_relalg_tool_scheme(
        mode="direct"
    ).protocol_hash
    assert capability_manifest("direct", CARRIER_TEXT_JSON)["max_tool_calls_per_turn"] == 0
    assert carrier_experiment_arm(CARRIER_TEXT_JSON) == "A"
    assert carrier_experiment_arm(CARRIER_NATIVE_TOOL_CALLS) == "B"


def test_native_request_payload_is_unchanged_and_text_omits_native_controls() -> None:
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    tools = provider_tool_definitions("direct")
    native = _QueuedClient(CARRIER_NATIVE_TOOL_CALLS, [_response(_native_message())])
    native.request_turn(messages=messages, tools=tools, max_tokens=32)
    native_payload = native.payloads[0]
    assert list(native_payload) == [
        "model",
        "messages",
        "max_tokens",
        "tools",
        "tool_choice",
        "thinking",
        "reasoning_effort",
    ]
    assert native_payload["tools"] == tools
    assert native_payload["tool_choice"] == "auto"
    assert "response_format" not in native_payload

    text = _QueuedClient(
        CARRIER_TEXT_JSON,
        [_response(_text_message(), finish_reason="stop")],
    )
    text.request_turn(messages=messages, tools=tools, max_tokens=32)
    text_payload = text.payloads[0]
    assert list(text_payload) == [
        "model",
        "messages",
        "max_tokens",
        "response_format",
        "thinking",
        "reasoning_effort",
    ]
    assert text_payload["response_format"] == {"type": "json_object"}
    assert "tools" not in text_payload
    assert "tool_choice" not in text_payload


def test_response_model_mismatch_is_non_retryable_and_usage_audited() -> None:
    response = _response(_text_message(), finish_reason="stop")
    response["model"] = "deepseek-pro"
    client = _QueuedClient(CARRIER_TEXT_JSON, [response])
    with pytest.raises(ProviderModelMismatch) as caught:
        client.request_turn_with_retries(
            messages=[{"role": "user", "content": "x"}],
            tools=provider_tool_definitions("direct"),
            max_tokens=8,
            max_completion_tokens=16,
            retries=4,
        )
    assert len(client.payloads) == 1
    assert client.payloads[0]["model"] == "deepseek-v4-flash"
    assert caught.value.accumulated_usage["total_tokens"] == 11
    assert caught.value.provider_attempt_count == 1
    assert caught.value.provider_attempt_events[0]["shape_category"] == "model_mismatch"


def test_model_mismatch_precedes_choices_shape_and_keeps_usage() -> None:
    client = _QueuedClient(CARRIER_TEXT_JSON, [{
        "model": "deepseek-pro",
        "choices": [],
        "usage": {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19},
    }])
    with pytest.raises(ProviderModelMismatch) as caught:
        client.request_turn_with_retries(
            messages=[{"role": "user", "content": "x"}],
            tools=provider_tool_definitions("direct"),
            max_tokens=8,
            max_completion_tokens=16,
            retries=4,
        )
    assert len(client.payloads) == 1
    assert caught.value.accumulated_usage["total_tokens"] == 19
    assert caught.value.provider_attempt_events == [{
        "attempt_index": 1,
        "max_tokens": 8,
        "finish_reason": None,
        "usage": {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19},
        "shape_category": "model_mismatch",
        "response_envelope_sha256": None,
        "response_model": "deepseek-pro",
        "raw_assistant_message": None,
        "elapsed_seconds_from_request_start": caught.value.provider_attempt_events[0][
            "elapsed_seconds_from_request_start"
        ],
    }]

    missing_model = _response(_text_message(), finish_reason="stop")
    missing_model.pop("model")
    client = _QueuedClient(CARRIER_TEXT_JSON, [missing_model])
    with pytest.raises(ProviderShapeError) as missing:
        client.request_turn_with_retries(
            messages=[], tools=None, max_tokens=32, retries=1
        )
    assert missing.value.accumulated_usage["total_tokens"] == 11
    assert missing.value.provider_attempt_events[-1]["shape_category"] == "provider_shape"


def test_usage_preserves_cache_and_reasoning_counters_and_missing_usage_retries() -> None:
    response = _response(_text_message(), finish_reason="stop")
    response["usage"].update({
        "prompt_cache_hit_tokens": 5,
        "prompt_cache_miss_tokens": 4,
        "completion_tokens_details": {"reasoning_tokens": 1},
    })
    client = _QueuedClient(CARRIER_TEXT_JSON, [response])
    accepted = client.request_turn_with_retries(
        messages=[], tools=None, max_tokens=32, retries=1
    )
    assert accepted.usage["prompt_cache_hit_tokens"] == 5
    assert accepted.usage["prompt_cache_miss_tokens"] == 4
    assert accepted.usage["completion_tokens_details.reasoning_tokens"] == 1
    assert accepted.provider_attempt_events[0]["usage"] == accepted.usage

    missing = _response(_text_message(), finish_reason="stop")
    missing.pop("usage")
    client = _QueuedClient(CARRIER_TEXT_JSON, [missing])
    with pytest.raises(ProviderShapeError) as caught:
        client.request_turn_with_retries(
            messages=[], tools=None, max_tokens=32, retries=1
        )
    assert caught.value.provider_attempt_events[0]["shape_category"] == "provider_shape"
    assert caught.value.provider_attempt_events[0]["usage"] == {}


def test_finish_reason_policy_retries_resource_and_never_executes_filter() -> None:
    resource = _response(
        _native_message(),
        finish_reason="insufficient_system_resource",
        tokens=6,
    )
    accepted = _response(_native_message(), finish_reason="tool_calls", tokens=7)
    client = _QueuedClient(CARRIER_NATIVE_TOOL_CALLS, [resource, accepted])
    response = client.request_turn_with_retries(
        messages=[], tools=provider_tool_definitions("direct"), max_tokens=32, retries=2
    )
    assert response.retry_events == [{
        "type": "insufficient_system_resource",
        "attempt": 1,
    }]
    assert [event["shape_category"] for event in response.provider_attempt_events] == [
        "insufficient_system_resource",
        "accepted_response",
    ]
    assert response.usage["total_tokens"] == 13

    filtered = _response(_text_message(), finish_reason="content_filter", tokens=8)
    client = _QueuedClient(CARRIER_TEXT_JSON, [filtered])
    with pytest.raises(ProviderContentFiltered) as caught:
        client.request_turn_with_retries(
            messages=[], tools=None, max_tokens=32, retries=3
        )
    assert len(client.payloads) == 1
    assert caught.value.provider_attempt_events[-1]["shape_category"] == "content_filter"

    exhausted = _QueuedClient(CARRIER_NATIVE_TOOL_CALLS, [resource])
    with pytest.raises(ProviderInsufficientSystemResource):
        exhausted.request_turn_with_retries(
            messages=[], tools=provider_tool_definitions("direct"), max_tokens=32, retries=1
        )


@pytest.mark.parametrize(
    ("carrier", "message", "wrong_finish"),
    [
        (CARRIER_NATIVE_TOOL_CALLS, _native_message(), "stop"),
        (CARRIER_TEXT_JSON, _text_message(), "tool_calls"),
    ],
)
def test_valid_action_envelope_with_wrong_finish_reason_is_not_accepted(
    carrier: str,
    message: dict,
    wrong_finish: str,
) -> None:
    client = _QueuedClient(
        carrier,
        [_response(message, finish_reason=wrong_finish)],
    )
    with pytest.raises(ProviderShapeError) as caught:
        client.request_turn_with_retries(
            messages=[],
            tools=(
                provider_tool_definitions("direct")
                if carrier == CARRIER_NATIVE_TOOL_CALLS
                else None
            ),
            max_tokens=32,
            retries=1,
        )
    assert caught.value.provider_attempt_events[-1]["shape_category"] == "provider_shape"


def test_empty_text_has_distinct_retry_shape_category() -> None:
    empty = _text_message()
    empty["content"] = ""
    client = _QueuedClient(CARRIER_TEXT_JSON, [
        _response(empty, finish_reason="stop", tokens=19),
        _response(_text_message(), finish_reason="stop", tokens=7),
    ])
    response = client.request_turn_with_retries(
        messages=[{"role": "user", "content": "x"}],
        tools=provider_tool_definitions("direct"),
        max_tokens=8,
        max_completion_tokens=16,
        retries=2,
    )
    assert response.provider_attempt_events[0]["shape_category"] == "empty_text"
    assert response.retry_events[0]["type"] == "provider_shape"
    assert response.usage["total_tokens"] == 26


@pytest.mark.parametrize("carrier", [CARRIER_NATIVE_TOOL_CALLS, CARRIER_TEXT_JSON])
def test_length_with_empty_or_malformed_message_expands_before_shape_validation(
    carrier: str,
) -> None:
    valid_message = _native_message() if carrier == CARRIER_NATIVE_TOOL_CALLS else _text_message()
    valid_finish_reason = (
        "tool_calls" if carrier == CARRIER_NATIVE_TOOL_CALLS else "stop"
    )
    client = _QueuedClient(carrier, [
        _response({}, finish_reason="length", tokens=18),
        _response(valid_message, finish_reason=valid_finish_reason, tokens=7),
    ])
    response = client.request_turn_with_retries(
        messages=[{"role": "user", "content": "x"}],
        tools=provider_tool_definitions("direct"),
        max_tokens=8,
        max_completion_tokens=16,
        retries=2,
    )
    assert [payload["max_tokens"] for payload in client.payloads] == [8, 16]
    assert response.provider_attempt_count == 2
    assert response.usage["total_tokens"] == 25
    assert response.retry_events[0]["type"] == "completion_length"
    assert len(response.provider_attempt_events) == 2
    assert response.provider_attempt_events[0]["usage"]["total_tokens"] == 18
    assert response.provider_attempt_events[0]["response_envelope_sha256"]
    assert response.provider_attempt_events[0]["elapsed_seconds_from_request_start"] >= 0


@pytest.mark.parametrize(
    "content",
    [
        "Here is the answer",
        "```json\n{\"tool\":\"describe_table\",\"arguments\":{}}\n```",
        "[]",
        "{\"tool\":\"describe_table\",\"arguments\":{}} trailing",
    ],
)
def test_text_json_parser_never_repairs_visible_content(content: str) -> None:
    message = {
        "role": "assistant",
        "reasoning_content": "reason",
        "content": content,
    }
    with pytest.raises(TextJSONActionError):
        validate_text_json_assistant_message("direct", message)


@pytest.mark.parametrize(
    ("validator", "message", "envelope_valid"),
    [
        (
            validate_text_json_assistant_message,
            _text_message("unknown_tool", {}),
            text_json_carrier_envelope_valid,
        ),
        (
            validate_native_assistant_message,
            _native_message("unknown_tool", {}),
            native_carrier_envelope_valid,
        ),
    ],
)
def test_unknown_tool_is_action_validation_not_carrier_envelope_failure(
    validator,
    message,
    envelope_valid,
) -> None:
    assert envelope_valid(message) is True
    with pytest.raises((TextJSONActionError, NativeToolCallError)) as caught:
        validator("direct", message)
    assert caught.value.code == "unknown_tool"


@pytest.mark.parametrize(
    ("validator", "message", "envelope_valid"),
    [
        (
            validate_text_json_assistant_message,
            _text_message("describe_table", {}),
            text_json_carrier_envelope_valid,
        ),
        (
            validate_native_assistant_message,
            _native_message("describe_table", {}),
            native_carrier_envelope_valid,
        ),
    ],
)
def test_invalid_arguments_are_not_carrier_envelope_failures(
    validator,
    message,
    envelope_valid,
) -> None:
    assert envelope_valid(message) is True
    with pytest.raises((TextJSONActionError, NativeToolCallError)) as caught:
        validator("direct", message)
    assert caught.value.code == "missing_required_field"


@pytest.mark.parametrize("mutation", ["missing_type", "object_arguments"])
def test_native_raw_validator_requires_explicit_type_and_json_string(mutation: str) -> None:
    message = _native_message()
    call = message["tool_calls"][0]
    if mutation == "missing_type":
        del call["type"]
    else:
        call["function"]["arguments"] = {"tables": ["people"]}
    assert native_carrier_envelope_valid(message) is False
    with pytest.raises(NativeToolCallError):
        validate_native_assistant_message("direct", message)


def test_official_native_call_index_is_audited_but_not_lowered() -> None:
    message = _native_message()
    stored, calls = _normalize_assistant_message(
        message,
        carrier=CARRIER_NATIVE_TOOL_CALLS,
    )
    assert stored["tool_calls"][0]["index"] == 0
    assert len(calls) == 1
    assert native_carrier_envelope_valid(message) is True
    action = validate_native_assistant_message("direct", message)
    assert action["tool"] == "describe_table"
    assert action["tool_call_id"] == "call_1"
    assert "index" not in action

    legacy = deepcopy(message)
    del legacy["tool_calls"][0]["index"]
    assert native_carrier_envelope_valid(legacy) is True
    assert validate_native_assistant_message("direct", legacy)["tool"] == "describe_table"


@pytest.mark.parametrize("bad_index", [True, -1, 1, "0", None])
def test_invalid_single_native_call_index_is_rejected_as_provider_shape(
    bad_index,
) -> None:
    message = _native_message()
    message["tool_calls"][0]["index"] = bad_index
    with pytest.raises(ProviderShapeError):
        _normalize_assistant_message(message, carrier=CARRIER_NATIVE_TOOL_CALLS)
    assert native_carrier_envelope_valid(message) is False
    with pytest.raises(NativeToolCallError) as caught:
        validate_native_assistant_message("direct", message)
    assert caught.value.code == "invalid_tool_call_index"


def test_multi_call_indices_must_be_complete_and_match_provider_order() -> None:
    message = _native_message()
    second = deepcopy(message["tool_calls"][0])
    second["id"] = "call_2"
    second["index"] = 1
    message["tool_calls"].append(second)
    _, calls = _normalize_assistant_message(
        message,
        carrier=CARRIER_NATIVE_TOOL_CALLS,
    )
    assert [call.call_id for call in calls] == ["call_1", "call_2"]

    for indices in ([0, 0], [1, 0], [0, None]):
        changed = deepcopy(message)
        for position, value in enumerate(indices):
            if value is None:
                del changed["tool_calls"][position]["index"]
            else:
                changed["tool_calls"][position]["index"] = value
        with pytest.raises(ProviderShapeError):
            _normalize_assistant_message(
                changed,
                carrier=CARRIER_NATIVE_TOOL_CALLS,
            )

    legacy = deepcopy(message)
    for call in legacy["tool_calls"]:
        del call["index"]
    _, legacy_calls = _normalize_assistant_message(
        legacy,
        carrier=CARRIER_NATIVE_TOOL_CALLS,
    )
    assert len(legacy_calls) == 2


def test_text_result_feedback_is_causal_user_message() -> None:
    result = {"status": "success", "step_id": "step_001"}
    message = text_json_result_message(result)
    assert message["role"] == "user"
    assert decode_text_json_result_message(message) == result


def test_text_history_requires_exactly_one_feedback_per_assistant() -> None:
    assistant = _text_message()
    feedback = text_json_result_message({"status": "success"})
    assert provider_history_issues(
        [assistant, feedback],
        root="history",
        carrier=CARRIER_TEXT_JSON,
    ) == []
    assert provider_history_issues(
        [feedback],
        root="history",
        carrier=CARRIER_TEXT_JSON,
    )
    assert provider_history_issues(
        [assistant, feedback, feedback],
        root="history",
        carrier=CARRIER_TEXT_JSON,
    )
    assert provider_history_issues(
        [assistant],
        root="history",
        carrier=CARRIER_TEXT_JSON,
    )


@pytest.mark.parametrize("carrier", [CARRIER_NATIVE_TOOL_CALLS, CARRIER_TEXT_JSON])
def test_runner_writes_recomputable_carrier_metrics_and_turn_usage(
    tmp_path,
    carrier: str,
) -> None:
    db_path = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO people VALUES(1, 'Ada')")
    connection.commit()
    connection.close()
    message = _native_message() if carrier == CARRIER_NATIVE_TOOL_CALLS else _text_message()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect the table.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(carrier, message),
        carrier=carrier,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    metrics = record["turns"][0]["carrier_metrics"]
    assert metrics["provider_response_present"] is True
    assert metrics["authored_action_count"] == 1
    assert metrics["exact_single_action"] is True
    assert metrics["carrier_envelope_valid"] is True
    assert metrics["carrier_error_code"] is None
    assert metrics["action_validation_error_code"] is None
    assert record["turns"][0]["provider_usage"] == record["provider_usage"]
    if carrier == CARRIER_NATIVE_TOOL_CALLS:
        assert (
            record["provider_response_envelope_version"]
            == NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
        )
        for old_identity in (None, "deepseek-native-tool-call-response-envelope-v1"):
            changed = deepcopy(record)
            if old_identity is None:
                changed.pop("provider_response_envelope_version")
            else:
                changed["provider_response_envelope_version"] = old_identity
            identity_audit = audit_record(changed)
            assert identity_audit["passed"] is False
            assert any(
                "provider_response_envelope_version" in issue
                for issue in identity_audit["issues"]
            )
    else:
        # Frozen text-JSON A artifacts predate the native-only identity field.
        assert "provider_response_envelope_version" not in record
        changed = deepcopy(record)
        changed["provider_response_envelope_version"] = "forged-native-envelope"
        identity_audit = audit_record(changed)
        assert identity_audit["passed"] is False
        assert any(
            "provider_response_envelope_version" in issue
            for issue in identity_audit["issues"]
        )
    assert audit_record(record)["passed"] is True


def test_runner_classifies_unknown_tool_as_action_validation(tmp_path) -> None:
    db_path = tmp_path / "tiny.sqlite"
    sqlite3.connect(db_path).close()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_TEXT_JSON, _text_message("unknown_tool", {})),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    metrics = record["turns"][0]["carrier_metrics"]
    assert metrics["carrier_envelope_valid"] is True
    assert metrics["carrier_error_code"] is None
    assert metrics["action_validation_error_code"] == "unknown_tool"
    assert audit_record(record)["passed"] is True


@pytest.mark.parametrize(
    "old_identity",
    [None, "deepseek-native-tool-call-response-envelope-v1"],
)
def test_native_manifest_response_envelope_identity_fails_closed(
    tmp_path,
    old_identity: str | None,
) -> None:
    db_path = tmp_path / "native_identity.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY)")
    connection.close()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "native_identity",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_NATIVE_TOOL_CALLS, _native_message()),
        carrier=CARRIER_NATIVE_TOOL_CALLS,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    manifest = deepcopy(record)
    manifest["run_started_at_utc"] = "2026-08-09T00:00:00+00:00"
    assert _manifest_binding_report(manifest, [record])["passed"] is True

    changed_manifest = deepcopy(manifest)
    changed_record = deepcopy(record)
    if old_identity is None:
        changed_manifest.pop("provider_response_envelope_version")
        changed_record.pop("provider_response_envelope_version")
    else:
        changed_manifest["provider_response_envelope_version"] = old_identity
        changed_record["provider_response_envelope_version"] = old_identity
    report = _manifest_binding_report(changed_manifest, [changed_record])
    assert report["passed"] is False
    assert any(
        "manifest.provider_response_envelope_version" in issue
        for issue in report["issues"]
    )


def test_text_manifest_rejects_synchronized_response_envelope_forgery(
    tmp_path,
) -> None:
    db_path = tmp_path / "text_identity.sqlite"
    sqlite3.connect(db_path).close()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "text_identity",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_TEXT_JSON, _text_message()),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    manifest = deepcopy(record)
    manifest["run_started_at_utc"] = "2026-08-09T00:00:00+00:00"
    assert _manifest_binding_report(manifest, [record])["passed"] is True

    manifest["provider_response_envelope_version"] = "forged-native-envelope"
    record["provider_response_envelope_version"] = "forged-native-envelope"
    report = _manifest_binding_report(manifest, [record])
    assert report["passed"] is False
    assert any(
        "manifest.provider_response_envelope_version" in issue
        for issue in report["issues"]
    )


def test_text_history_preserves_split_assistant_and_uses_user_feedback(tmp_path) -> None:
    db_path = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY)")
    connection.close()
    client = _EpisodeClient(CARRIER_TEXT_JSON, _text_message())
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=client,
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=2,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    history = client.requests[1]
    assert [message["role"] for message in history] == [
        "system", "user", "assistant", "user", "user"
    ]
    assert history[2]["reasoning_content"] == _text_message()["reasoning_content"]
    assert history[2]["content"] == _text_message()["content"]
    assert decode_text_json_result_message(history[3]) == record["turns"][0]["result"]
    assert audit_record(record)["passed"] is True


def test_audit_binds_attempt_metrics_to_success_and_provider_failure(tmp_path) -> None:
    db_path = tmp_path / "tiny.sqlite"
    sqlite3.connect(db_path).close()
    task = {
        "db_path": str(db_path),
        "db_id": "tiny",
        "question": "Inspect.",
        "example_index": 0,
    }
    success = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_TEXT_JSON, _text_message()),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    changed = deepcopy(success)
    changed["turns"][0]["carrier_metrics"]["provider_attempt_count"] = 999
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(success)
    changed["turns"][0]["provider_attempt_events"][-1][
        "response_envelope_sha256"
    ] = "0" * 64
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(success)
    changed["turns"][0]["carrier_metrics"]["provider_elapsed_seconds"] = 0
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(success)
    changed["turns"][0]["finish_reason"] = "tool_calls"
    changed["turns"][0]["provider_attempt_events"][-1][
        "finish_reason"
    ] = "tool_calls"
    assert audit_record(changed)["passed"] is False

    failure = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=_FailureClient(),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    assert audit_record(failure)["passed"] is True
    changed = deepcopy(failure)
    del changed["turns"][0]["provider_error"]
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(failure)
    changed["turns"][0]["provider_attempt_events"][-1][
        "shape_category"
    ] = "transport"
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(failure)
    changed["turns"][0]["provider_attempt_events"][-1][
        "shape_category"
    ] = "transport"
    changed["turns"][0]["provider_error"]["type"] = "ProviderError"
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(failure)
    changed["failure_type"] = "context_length_exceeded"
    assert audit_record(changed)["passed"] is False
    changed = deepcopy(failure)
    changed["turns"].append(deepcopy(changed["turns"][0]))
    assert audit_record(changed)["passed"] is False


def test_audit_recomputes_usage_equations_across_all_aggregation_levels(
    tmp_path,
) -> None:
    db_path = tmp_path / "tiny.sqlite"
    sqlite3.connect(db_path).close()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_TEXT_JSON, _text_message()),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    assert audit_record(record)["passed"] is True

    for updates in (
        {"total_tokens": 12},
        {"prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 5},
        {"completion_tokens_details.reasoning_tokens": 3},
    ):
        changed = deepcopy(record)
        event_usage = changed["turns"][0]["provider_attempt_events"][0]["usage"]
        turn_usage = changed["turns"][0]["provider_usage"]
        record_usage = changed["provider_usage"]
        event_usage.update(updates)
        turn_usage.update(updates)
        record_usage.update(updates)
        assert audit_record(changed)["passed"] is False


def test_audit_recomputes_early_retry_shape_from_raw_assistant(tmp_path) -> None:
    db_path = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY)")
    connection.close()
    empty = _text_message()
    empty["content"] = ""
    client = _QueuedClient(CARRIER_TEXT_JSON, [
        _response(empty, finish_reason="stop", tokens=19),
        _response(_text_message(), finish_reason="stop", tokens=7),
    ])
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=client,
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    assert audit_record(record)["passed"] is True
    first = record["turns"][0]["provider_attempt_events"][0]
    assert first["shape_category"] == "empty_text"

    changed = deepcopy(record)
    changed["turns"][0]["provider_attempt_events"][0][
        "raw_assistant_message"
    ] = _text_message()
    assert audit_record(changed)["passed"] is False

    changed = deepcopy(record)
    first = changed["turns"][0]["provider_attempt_events"][0]
    first["raw_assistant_message"] = _text_message()
    first["response_envelope_sha256"] = hashlib.sha256(json.dumps(
        first["raw_assistant_message"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert audit_record(changed)["passed"] is False

    first["shape_category"] = "accepted_response"
    assert audit_record(changed)["passed"] is False


def test_audit_rejects_extra_native_tool_result_fields_even_if_history_matches(
    tmp_path,
) -> None:
    db_path = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY)")
    connection.close()
    record = run_episode(
        {
            "db_path": str(db_path),
            "db_id": "tiny",
            "question": "Inspect.",
            "example_index": 0,
        },
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_NATIVE_TOOL_CALLS, _native_message()),
        carrier=CARRIER_NATIVE_TOOL_CALLS,
        runtime_config=RuntimeConfig(),
        max_model_turns=2,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    assert audit_record(record)["passed"] is True
    changed = deepcopy(record)
    changed["turns"][0]["tool_result_messages"][0]["name"] = "describe_table"
    changed["turns"][1]["model_input"][3]["name"] = "describe_table"
    report = audit_record(changed)
    assert report["passed"] is False
    assert any("native tool result" in issue for issue in report["issues"])


def test_result_dir_audit_rejects_synchronized_manifest_identity_tamper(tmp_path) -> None:
    db_path = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY)")
    connection.close()
    task = {
        "db_path": str(db_path),
        "db_id": "tiny",
        "question": "Inspect.",
        "external_knowledge": None,
        "example_index": 0,
    }
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=_EpisodeClient(CARRIER_TEXT_JSON, _text_message()),
        carrier=CARRIER_TEXT_JSON,
        runtime_config=RuntimeConfig(),
        max_model_turns=1,
        max_tokens=32,
        max_completion_tokens=64,
        api_retries=2,
    )
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    bound_fields = (
        "tool_scheme_registry_version", "tool_scheme", "protocol_version",
        "protocol_hash", "assistant_carrier", "top_level_tools", "atomic_tools",
        "max_batch_calls", "mode", "tool_schema_sha256", "student_prompt_sha256",
        "teacher_prompt_sha256", "admission_status", "capability_manifest", "backend",
        "dialect", "environment_renderer_version", "checkpoint_policy_version",
        "executor_version", "tool_schema_hash", "carrier_ablation_protocol_version",
        "carrier_policy_version", "experiment_arm", "within_batch_order", "carrier",
        "prompt_hash", "provider_request_options", "runtime_config",
        "denotation_comparison", "strict_artifact_audit_version",
        "sft_export_eligible", "rl_admission_eligible",
    )
    manifest = {field: deepcopy(record.get(field)) for field in bound_fields}
    manifest.update({
        "dataset": str(tasks_path),
        "run_started_at_utc": "2026-08-09T00:00:00+00:00",
    })
    manifest_path = result_dir / "manifest.json"
    records_path = result_dir / "all.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["passed"] is True
    assert set(report["artifact_sha256"]) == {"manifest.json", "all.jsonl"}

    manifest["protocol_hash"] = "0" * 64
    record["protocol_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tampered = audit_result_dir(result_dir)
    assert tampered["manifest_binding"]["passed"] is True
    assert tampered["passed"] is False
    assert any(
        "protocol_hash does not match" in issue
        for detail in tampered["records_detail"]
        for issue in detail["issues"]
    )
