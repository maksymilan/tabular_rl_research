"""Official DeepSeek transport for checkpoint-relalg carrier diagnostics.

The default remains the native single-tool-call carrier.  The explicitly selected text-json
diagnostic sends the same task through JSON Output without a provider tool surface.  This module
does not validate relational semantics, execute tools, repair arguments, or inspect verifier data.
"""

from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .protocol import (
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
    DEFAULT_CARRIER,
    assistant_carrier_protocol,
    normalize_carrier,
)


OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION = (
    "deepseek-native-tool-call-response-envelope-index-aware-v2"
)


def provider_response_envelope_version(carrier: str = DEFAULT_CARRIER) -> str | None:
    """Return the carrier-specific response-envelope identity, when versioned.

    Text-JSON keeps its frozen artifact identity unchanged.  Native responses
    are versioned separately because the official transport includes an
    ``index`` member on each tool call; that raw member is audited but is not
    lowered into the canonical action.
    """

    active_carrier = normalize_carrier(carrier)
    if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
        return NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
    return None


def provider_response_envelope_protocol_hash(
    base_protocol_hash: str,
    carrier: str = DEFAULT_CARRIER,
) -> str:
    """Bind a versioned response envelope into the carrier protocol identity."""

    version = provider_response_envelope_version(carrier)
    if version is None:
        return base_protocol_hash
    payload = {
        "base_protocol_hash": base_protocol_hash,
        "provider_response_envelope_version": version,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provider_request_audit_options(
    *,
    base_url: str,
    model: str,
    carrier: str = DEFAULT_CARRIER,
) -> dict[str, Any]:
    """Return auditable controls without duplicating full tool schemas."""

    active_carrier = normalize_carrier(carrier)
    options: dict[str, Any] = {
        "endpoint": base_url.rstrip("/") + "/chat/completions",
        "model": model,
        "carrier": active_carrier,
        "assistant_carrier": assistant_carrier_protocol(active_carrier),
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
        options["tool_choice"] = "auto"
    else:
        options["response_format"] = {"type": "json_object"}
    return options


class ProviderError(RuntimeError):
    """Base class for provider/transport failures."""

    def __init__(
        self,
        message: str,
        *,
        accumulated_usage: dict[str, int] | None = None,
        retry_events: list[dict[str, Any]] | None = None,
        provider_attempt_count: int | None = None,
        provider_elapsed_seconds: float | None = None,
        provider_attempt_events: list[dict[str, Any]] | None = None,
        finish_reason: str | None = None,
        response_envelope_sha256: str | None = None,
        response_model: str | None = None,
        raw_assistant_message: dict[str, Any] | None = None,
        http_status: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.accumulated_usage = dict(accumulated_usage or {})
        self.retry_events = deepcopy(retry_events or [])
        self.provider_attempt_count = provider_attempt_count
        self.provider_elapsed_seconds = provider_elapsed_seconds
        self.provider_attempt_events = deepcopy(provider_attempt_events or [])
        self.finish_reason = finish_reason
        self.response_envelope_sha256 = response_envelope_sha256
        self.response_model = response_model
        self.raw_assistant_message = deepcopy(raw_assistant_message)
        self.http_status = http_status
        self.retryable = retryable


class ProviderContextOverflow(ProviderError):
    """The provider rejected the complete, untruncated prompt as too long."""


class ProviderShapeError(ProviderError):
    """The response did not contain a usable assistant tool-call message."""


class ProviderEmptyText(ProviderShapeError):
    """The text-json response contained no visible action content."""


class ProviderCompletionTruncated(ProviderError):
    """The provider stopped because the requested completion budget was exhausted."""


class ProviderModelMismatch(ProviderError):
    """The provider response identity differs from the explicitly requested model."""


class ProviderInsufficientSystemResource(ProviderError):
    """The provider could not complete the request because capacity was unavailable."""


class ProviderContentFiltered(ProviderError):
    """The provider stopped the completion because its content filter fired."""


_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429})


def _http_status_is_retryable(status: int) -> bool:
    """Return the bounded same-turn retry policy for one HTTP response.

    Authentication, billing, permission, and other deterministic 4xx failures
    cannot be repaired by repeating an identical request.  The small set of
    transient 4xx statuses and all 5xx responses may be retried by the existing
    bounded transport loop.
    """

    return status in _RETRYABLE_HTTP_STATUSES or 500 <= status <= 599


_REQUIRED_ACCEPTED_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _provider_usage_counts(raw_usage: Any) -> dict[str, int]:
    """Flatten only non-negative provider-authored integer counters.

    DeepSeek currently returns cache counters at the top level and may return
    reasoning counters in nested ``*_tokens_details`` objects.  Dotted keys
    retain that provenance while keeping turn/record aggregation content-free.
    """

    counts: dict[str, int] = {}

    def visit(value: Any, prefix: str) -> None:
        if not isinstance(value, dict):
            return
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                continue
            key = f"{prefix}.{raw_key}" if prefix else raw_key
            if isinstance(child, int) and not isinstance(child, bool) and child >= 0:
                counts[key] = child
            elif isinstance(child, dict):
                visit(child, key)

    visit(raw_usage, "")
    return counts


def _validated_provider_usage(raw_usage: Any) -> dict[str, int]:
    """Require the official core usage counters before an action can execute."""

    if not isinstance(raw_usage, dict):
        raise ProviderShapeError("provider response usage must be an object")
    for key in _REQUIRED_ACCEPTED_USAGE_KEYS:
        value = raw_usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProviderShapeError(
                f"provider response usage.{key} must be a non-negative integer"
            )
    prompt_tokens = raw_usage["prompt_tokens"]
    completion_tokens = raw_usage["completion_tokens"]
    total_tokens = raw_usage["total_tokens"]
    if total_tokens != prompt_tokens + completion_tokens:
        raise ProviderShapeError(
            "provider response usage.total_tokens must equal prompt_tokens + completion_tokens"
        )

    def reject_invalid_counters(value: Any, prefix: str) -> None:
        if not isinstance(value, dict):
            return
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                continue
            key = f"{prefix}.{raw_key}" if prefix else raw_key
            if isinstance(child, bool) or (
                isinstance(child, int) and child < 0
            ):
                raise ProviderShapeError(
                    f"provider response usage.{key} is not a non-negative integer counter"
                )
            if isinstance(child, dict):
                reject_invalid_counters(child, key)

    reject_invalid_counters(raw_usage, "")
    cache_hit = raw_usage.get("prompt_cache_hit_tokens")
    cache_miss = raw_usage.get("prompt_cache_miss_tokens")
    for key, value in (
        ("prompt_cache_hit_tokens", cache_hit),
        ("prompt_cache_miss_tokens", cache_miss),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ProviderShapeError(
                f"provider response usage.{key} must be a non-negative integer"
            )
    if isinstance(cache_hit, int) and isinstance(cache_miss, int) and (
        cache_hit + cache_miss != prompt_tokens
    ):
        raise ProviderShapeError(
            "provider prompt cache hit + miss counters must equal prompt_tokens"
        )
    details = raw_usage.get("completion_tokens_details")
    reasoning_tokens = raw_usage.get("completion_tokens_details.reasoning_tokens")
    if isinstance(details, dict) and "reasoning_tokens" in details:
        reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is not None:
        if (
            isinstance(reasoning_tokens, bool)
            or not isinstance(reasoning_tokens, int)
            or reasoning_tokens < 0
        ):
            raise ProviderShapeError(
                "provider reasoning_tokens must be a non-negative integer"
            )
        if reasoning_tokens > completion_tokens:
            raise ProviderShapeError(
                "provider reasoning_tokens exceeds completion_tokens"
            )
    return _provider_usage_counts(raw_usage)


def _looks_like_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return (
        "maximum context length" in lowered
        or "context length" in lowered and "max_tokens" in lowered
        or "too many tokens" in lowered
    )


@dataclass(frozen=True)
class NativeToolCall:
    call_id: str
    name: str | None
    arguments_json: str | None

    def parsed_arguments(self) -> dict[str, Any]:
        if not isinstance(self.arguments_json, str):
            raise ProviderShapeError("tool call function.arguments must be a JSON string")
        try:
            value = json.loads(self.arguments_json)
        except json.JSONDecodeError as exc:
            raise ProviderShapeError(f"tool call arguments are invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ProviderShapeError("tool call arguments JSON must decode to an object")
        return value


@dataclass
class NativeAssistantResponse:
    message: dict[str, Any]
    calls: tuple[NativeToolCall, ...]
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    provider_attempt_count: int = 1
    provider_elapsed_seconds: float | None = None
    provider_attempt_events: list[dict[str, Any]] = field(default_factory=list)
    raw_assistant_message: dict[str, Any] | None = None

    @property
    def reasoning_content(self) -> str:
        value = self.message.get("reasoning_content")
        return value if isinstance(value, str) else ""


def _normalize_assistant_message(
    message: Any,
    *,
    carrier: str = DEFAULT_CARRIER,
) -> tuple[dict[str, Any], tuple[NativeToolCall, ...]]:
    active_carrier = normalize_carrier(carrier)
    if not isinstance(message, dict):
        raise ProviderShapeError("provider choice.message must be an object")
    if message.get("role") != "assistant":
        raise ProviderShapeError("provider choice.message.role must be 'assistant'")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ProviderShapeError(
            "thinking-mode assistant message requires non-empty reasoning_content"
        )
    if active_carrier == CARRIER_TEXT_JSON:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderEmptyText(
                "text-json assistant message requires non-empty visible content"
            )
        # Semantic parsing remains downstream and strict.  In particular, JSON
        # fences/prose/arrays are not repaired here and consume an agent action.
        return deepcopy(message), ()
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ProviderShapeError("provider assistant tool_calls must be a list")
    calls: list[NativeToolCall] = []
    seen_ids: set[str] = set()
    indexed_calls = 0
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise ProviderShapeError(f"tool_calls[{index}] must be an object")
        if raw_call.get("type") != "function":
            raise ProviderShapeError(f"tool_calls[{index}].type must be 'function'")
        if "index" in raw_call:
            indexed_calls += 1
            call_index = raw_call.get("index")
            if (
                isinstance(call_index, bool)
                or not isinstance(call_index, int)
                or call_index != index
            ):
                raise ProviderShapeError(
                    f"tool_calls[{index}].index must equal provider order {index}"
                )
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise ProviderShapeError(f"tool_calls[{index}].id must be non-empty")
        if call_id in seen_ids:
            raise ProviderShapeError(f"duplicate tool call id {call_id!r}")
        seen_ids.add(call_id)
        if not isinstance(function, dict):
            raise ProviderShapeError(f"tool_calls[{index}].function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
        if name is not None and not isinstance(name, str):
            raise ProviderShapeError(f"tool_calls[{index}].function.name must be a string")
        if arguments is not None and not isinstance(arguments, str):
            raise ProviderShapeError(
                f"tool_calls[{index}].function.arguments must be a JSON string"
            )
        calls.append(NativeToolCall(call_id, name, arguments))
    if indexed_calls not in {0, len(raw_calls)}:
        raise ProviderShapeError(
            "provider tool call indices must be present on every call or omitted from every call"
        )
    # Persist exactly what the provider authored.  Role/type defaults and
    # unexpected fields are semantic audit inputs; normalizing them here would
    # make a malformed response indistinguishable from a valid one.
    return deepcopy(message), tuple(calls)


def _response_envelope_sha256(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DeepSeekNativeClient:
    """Minimal official Chat Completions client with bounded transport retries.

    The historical class name is retained so native callers keep the same API.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int = 300,
        carrier: str = DEFAULT_CARRIER,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.carrier = normalize_carrier(carrier)
        if self.base_url != OFFICIAL_DEEPSEEK_BASE_URL:
            raise ValueError(
                "new DeepSeek requests require BASE_URL=https://api.deepseek.com"
            )

    @property
    def request_audit_options(self) -> dict[str, Any]:
        return provider_request_audit_options(
            base_url=self.base_url,
            model=self.model,
            carrier=self.carrier,
        )

    def _request_json(
        self,
        request: urllib.request.Request,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout or self.timeout_seconds,
            ) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            context_overflow = (
                400 <= exc.code <= 499
                and not _http_status_is_retryable(exc.code)
                and _looks_like_context_overflow(detail)
            )
            error_cls = ProviderContextOverflow if context_overflow else ProviderError
            retryable = (
                False if context_overflow else _http_status_is_retryable(exc.code)
            )
            raise error_cls(
                f"HTTP {exc.code}: provider request failed",
                http_status=exc.code,
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"URL error: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"invalid provider response: {type(exc).__name__}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("provider response root must be an object")
        return value

    def verify_model(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + "/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        payload = self._request_json(request)
        model_ids = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict)
        }
        if self.model not in model_ids:
            raise ProviderError(
                f"requested model {self.model!r} is not advertised by the configured provider"
            )
        return {"authenticated": True, "model_available": True, "model": self.model}

    def request_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> NativeAssistantResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.carrier == CARRIER_NATIVE_TOOL_CALLS:
            payload["tools"] = tools or []
            payload["tool_choice"] = "auto"
        else:
            payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "high"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        data = self._request_json(request)
        raw_usage = data.get("usage")
        response_usage = _provider_usage_counts(raw_usage)
        choices = data.get("choices")
        first_choice = (
            choices[0]
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        response_model = data.get("model")
        if not isinstance(response_model, str) or not response_model:
            finish_reason = (
                first_choice.get("finish_reason")
                if isinstance(first_choice, dict)
                else None
            )
            raw_message = (
                first_choice.get("message")
                if isinstance(first_choice, dict)
                else None
            )
            exc = ProviderShapeError("provider response model is missing")
            exc.accumulated_usage = response_usage
            exc.finish_reason = finish_reason if isinstance(finish_reason, str) else None
            exc.response_envelope_sha256 = _response_envelope_sha256(raw_message)
            exc.raw_assistant_message = (
                deepcopy(raw_message) if isinstance(raw_message, dict) else None
            )
            raise exc
        if response_model != self.model:
            finish_reason = (
                first_choice.get("finish_reason")
                if isinstance(first_choice, dict)
                else None
            )
            raw_message = (
                first_choice.get("message")
                if isinstance(first_choice, dict)
                else None
            )
            exc = ProviderModelMismatch(
                f"provider response model {response_model!r} differs from requested model"
            )
            exc.accumulated_usage = response_usage
            exc.response_model = response_model if isinstance(response_model, str) else None
            exc.finish_reason = finish_reason if isinstance(finish_reason, str) else None
            exc.response_envelope_sha256 = _response_envelope_sha256(raw_message)
            exc.raw_assistant_message = (
                deepcopy(raw_message) if isinstance(raw_message, dict) else None
            )
            raise exc
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            exc = ProviderShapeError("provider response must contain choices[0]")
            exc.accumulated_usage = response_usage
            exc.response_model = response_model if isinstance(response_model, str) else None
            raise exc
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        try:
            response_usage = _validated_provider_usage(raw_usage)
        except ProviderShapeError as exc:
            exc.accumulated_usage = response_usage
            exc.finish_reason = finish_reason if isinstance(finish_reason, str) else None
            exc.response_envelope_sha256 = _response_envelope_sha256(
                choice.get("message")
            )
            exc.response_model = response_model if isinstance(response_model, str) else None
            raw_message = choice.get("message")
            exc.raw_assistant_message = (
                deepcopy(raw_message) if isinstance(raw_message, dict) else None
            )
            raise
        expected_finish_reason = (
            "tool_calls" if self.carrier == CARRIER_NATIVE_TOOL_CALLS else "stop"
        )
        if finish_reason != expected_finish_reason:
            # A length stop often carries empty or partial content/tool_calls.
            # Resource, filter, and other non-accepted stops can likewise carry
            # partial envelopes.  Preserve them without semantic normalization;
            # the shared retry loop applies the explicit bounded policy.
            raw_message = choice.get("message")
            return NativeAssistantResponse(
                message=deepcopy(raw_message) if isinstance(raw_message, dict) else {},
                calls=(),
                finish_reason=(finish_reason if isinstance(finish_reason, str) else None),
                usage=response_usage,
                response_metadata={
                    key: data.get(key)
                    for key in ("id", "model", "system_fingerprint", "object")
                    if data.get(key) is not None
                },
                raw_assistant_message=(
                    deepcopy(raw_message) if isinstance(raw_message, dict) else None
                ),
            )
        try:
            message, calls = _normalize_assistant_message(
                choice.get("message"),
                carrier=self.carrier,
            )
        except ProviderError as exc:
            exc.accumulated_usage = response_usage
            exc.finish_reason = finish_reason if isinstance(finish_reason, str) else None
            exc.response_envelope_sha256 = _response_envelope_sha256(
                choice.get("message")
            )
            exc.response_model = response_model if isinstance(response_model, str) else None
            raw_message = choice.get("message")
            exc.raw_assistant_message = (
                deepcopy(raw_message) if isinstance(raw_message, dict) else None
            )
            raise
        return NativeAssistantResponse(
            message=message,
            calls=calls,
            finish_reason=finish_reason,
            usage=response_usage,
            response_metadata={
                key: data.get(key)
                for key in ("id", "model", "system_fingerprint", "object")
                if data.get(key) is not None
            },
            raw_assistant_message=deepcopy(message),
        )

    def request_turn_with_retries(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        retries: int,
        max_completion_tokens: int = 8192,
    ) -> NativeAssistantResponse:
        """Retry transport, length truncation, and zero-call provider-shape defects.

        Multiple authored calls are intentionally returned to the runtime as one semantic protocol
        error.  Argument JSON and unknown functions are likewise never repaired here.
        """

        retry_events: list[dict[str, Any]] = []
        attempt_events: list[dict[str, Any]] = []
        accumulated_usage: dict[str, int] = {}
        budget = max_tokens
        last_error: BaseException | None = None
        started = time.monotonic()
        attempt_count = 0

        def elapsed_offset() -> float:
            return round(time.monotonic() - started, 6)

        def audited_error(exc: ProviderError) -> ProviderError:
            exc.accumulated_usage = dict(accumulated_usage)
            exc.retry_events = deepcopy(retry_events)
            exc.provider_attempt_count = attempt_count
            exc.provider_elapsed_seconds = elapsed_offset()
            exc.provider_attempt_events = deepcopy(attempt_events)
            return exc

        def accumulate_failed_usage(exc: ProviderError) -> None:
            for key, value in exc.accumulated_usage.items():
                accumulated_usage[key] = accumulated_usage.get(key, 0) + value

        for attempt in range(max(1, retries)):
            attempt_count += 1
            attempt_event_recorded = False
            try:
                response = self.request_turn(
                    messages=messages,
                    tools=tools,
                    max_tokens=budget,
                )
                raw_assistant_message = response.raw_assistant_message
                response_model = response.response_metadata.get("model")
                if not isinstance(response_model, str) or not response_model:
                    exc = ProviderShapeError(
                        "provider response metadata.model is missing",
                        accumulated_usage=_provider_usage_counts(response.usage),
                        finish_reason=response.finish_reason,
                        response_envelope_sha256=_response_envelope_sha256(
                            raw_assistant_message
                        ),
                        raw_assistant_message=raw_assistant_message,
                    )
                    raise exc
                if response_model != self.model:
                    exc = ProviderModelMismatch(
                        f"provider response model {response_model!r} differs from requested model",
                        accumulated_usage=_provider_usage_counts(response.usage),
                        finish_reason=response.finish_reason,
                        response_envelope_sha256=_response_envelope_sha256(
                            raw_assistant_message
                        ),
                        response_model=(
                            response_model if isinstance(response_model, str) else None
                        ),
                        raw_assistant_message=raw_assistant_message,
                    )
                    raise exc
                try:
                    response_usage = _validated_provider_usage(response.usage)
                except ProviderShapeError as exc:
                    exc.accumulated_usage = _provider_usage_counts(response.usage)
                    exc.finish_reason = response.finish_reason
                    exc.response_envelope_sha256 = _response_envelope_sha256(
                        raw_assistant_message
                    )
                    exc.response_model = response_model
                    exc.raw_assistant_message = deepcopy(raw_assistant_message)
                    raise
                response.usage = response_usage
                for key, value in response_usage.items():
                    accumulated_usage[key] = accumulated_usage.get(key, 0) + value
                shape_category = "accepted_response"
                if response.finish_reason == "length":
                    shape_category = "completion_length"
                elif response.finish_reason == "insufficient_system_resource":
                    shape_category = "insufficient_system_resource"
                elif response.finish_reason == "content_filter":
                    shape_category = "content_filter"
                elif response.finish_reason != (
                    "tool_calls"
                    if self.carrier == CARRIER_NATIVE_TOOL_CALLS
                    else "stop"
                ):
                    shape_category = "provider_shape"
                elif self.carrier == CARRIER_NATIVE_TOOL_CALLS and not response.calls:
                    shape_category = "missing_tool_calls"
                attempt_events.append({
                    "attempt_index": attempt_count,
                    "max_tokens": budget,
                    "finish_reason": response.finish_reason,
                    "usage": {
                        key: value
                        for key, value in response.usage.items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    },
                    "shape_category": shape_category,
                    "response_envelope_sha256": _response_envelope_sha256(
                        raw_assistant_message
                    ),
                    "response_model": response.response_metadata.get("model"),
                    "raw_assistant_message": deepcopy(raw_assistant_message),
                    "elapsed_seconds_from_request_start": elapsed_offset(),
                })
                attempt_event_recorded = True
                if response.finish_reason == "insufficient_system_resource":
                    exc = ProviderInsufficientSystemResource(
                        "provider reported insufficient system resources"
                    )
                    exc.finish_reason = response.finish_reason
                    exc.response_envelope_sha256 = _response_envelope_sha256(
                        raw_assistant_message
                    )
                    exc.response_model = response_model
                    exc.raw_assistant_message = deepcopy(raw_assistant_message)
                    raise exc
                if response.finish_reason == "content_filter":
                    exc = ProviderContentFiltered(
                        "provider completion was blocked by content filtering"
                    )
                    exc.finish_reason = response.finish_reason
                    exc.response_envelope_sha256 = _response_envelope_sha256(
                        raw_assistant_message
                    )
                    exc.response_model = response_model
                    exc.raw_assistant_message = deepcopy(raw_assistant_message)
                    raise exc
                expected_finish_reason = (
                    "tool_calls"
                    if self.carrier == CARRIER_NATIVE_TOOL_CALLS
                    else "stop"
                )
                if response.finish_reason not in {"length", expected_finish_reason}:
                    exc = ProviderShapeError(
                        "provider finish_reason "
                        f"{response.finish_reason!r} is invalid for {self.carrier}"
                    )
                    exc.finish_reason = response.finish_reason
                    exc.response_envelope_sha256 = _response_envelope_sha256(
                        raw_assistant_message
                    )
                    exc.response_model = response_model
                    exc.raw_assistant_message = deepcopy(raw_assistant_message)
                    raise exc
                if response.finish_reason == "length":
                    if attempt + 1 >= max(1, retries) or budget >= max_completion_tokens:
                        exc = ProviderCompletionTruncated(
                            "provider completion remained length-truncated after bounded retries"
                        )
                        exc.finish_reason = response.finish_reason
                        exc.response_envelope_sha256 = _response_envelope_sha256(
                            raw_assistant_message
                        )
                        exc.response_model = response.response_metadata.get("model")
                        exc.raw_assistant_message = deepcopy(raw_assistant_message)
                        raise exc
                    next_budget = min(max_completion_tokens, max(budget + 1, budget * 2))
                    retry_events.append({
                        "type": "completion_length",
                        "attempt": attempt + 1,
                        "max_tokens": budget,
                        "next_max_tokens": next_budget,
                    })
                    budget = next_budget
                    continue
                if self.carrier == CARRIER_NATIVE_TOOL_CALLS and not response.calls:
                    if attempt + 1 >= max(1, retries):
                        exc = ProviderShapeError(
                            "provider returned zero tool calls after bounded same-turn retries"
                        )
                        exc.finish_reason = response.finish_reason
                        exc.response_envelope_sha256 = _response_envelope_sha256(
                            raw_assistant_message
                        )
                        exc.response_model = response.response_metadata.get("model")
                        exc.raw_assistant_message = deepcopy(raw_assistant_message)
                        raise exc
                    retry_events.append({
                        "type": "missing_tool_calls",
                        "attempt": attempt + 1,
                    })
                    continue
                response.usage = {**response.usage, **accumulated_usage}
                response.retry_events = retry_events
                response.provider_attempt_count = attempt_count
                response.provider_elapsed_seconds = elapsed_offset()
                response.provider_attempt_events = deepcopy(attempt_events)
                return response
            except ProviderModelMismatch as exc:
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": "model_mismatch",
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                raise audited_error(exc)
            except ProviderInsufficientSystemResource as exc:
                last_error = exc
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": "insufficient_system_resource",
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                if attempt + 1 >= max(1, retries):
                    raise audited_error(exc)
                retry_events.append({
                    "type": "insufficient_system_resource",
                    "attempt": attempt + 1,
                })
                continue
            except ProviderContentFiltered as exc:
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": "content_filter",
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                raise audited_error(exc)
            except ProviderContextOverflow as exc:
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": "context_overflow",
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                raise audited_error(exc)
            except (ProviderCompletionTruncated, ProviderShapeError) as exc:
                last_error = exc
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": (
                            "completion_length"
                            if isinstance(exc, ProviderCompletionTruncated)
                            else (
                                "empty_text"
                                if isinstance(exc, ProviderEmptyText)
                                else "provider_shape"
                            )
                        ),
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                if isinstance(exc, ProviderCompletionTruncated):
                    raise audited_error(exc)
                if attempt + 1 >= max(1, retries):
                    raise audited_error(exc)
                retry_events.append({
                    "type": "provider_shape",
                    "attempt": attempt + 1,
                    "error": type(exc).__name__,
                })
            except ProviderError as exc:
                last_error = exc
                if not attempt_event_recorded:
                    attempt_events.append({
                        "attempt_index": attempt_count,
                        "max_tokens": budget,
                        "finish_reason": exc.finish_reason,
                        "usage": dict(exc.accumulated_usage),
                        "shape_category": "transport",
                        "response_envelope_sha256": exc.response_envelope_sha256,
                        "response_model": exc.response_model,
                        "raw_assistant_message": deepcopy(exc.raw_assistant_message),
                        "elapsed_seconds_from_request_start": elapsed_offset(),
                    })
                accumulate_failed_usage(exc)
                if not exc.retryable or attempt + 1 >= max(1, retries):
                    raise audited_error(exc)
                retry_events.append({
                    "type": "transport",
                    "attempt": attempt + 1,
                    "error": type(exc).__name__,
                })
                time.sleep(min(2 ** attempt, 8))
        raise ProviderError(
            f"provider request failed: {last_error}",
            accumulated_usage=accumulated_usage,
            retry_events=retry_events,
            provider_attempt_count=attempt_count,
            provider_elapsed_seconds=elapsed_offset(),
            provider_attempt_events=attempt_events,
        )


def tool_result_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
    }
