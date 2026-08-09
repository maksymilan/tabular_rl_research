"""Official DeepSeek native single-tool-call transport for checkpoint-relalg.

The provider transports model-authored reasoning and function calls.  This module does not
validate relational semantics, execute tools, repair arguments, or inspect verifier data.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class ProviderError(RuntimeError):
    """Base class for provider/transport failures."""

    def __init__(
        self,
        message: str,
        *,
        accumulated_usage: dict[str, int] | None = None,
        retry_events: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.accumulated_usage = dict(accumulated_usage or {})
        self.retry_events = deepcopy(retry_events or [])


class ProviderContextOverflow(ProviderError):
    """The provider rejected the complete, untruncated prompt as too long."""


class ProviderShapeError(ProviderError):
    """The response did not contain a usable assistant tool-call message."""


class ProviderCompletionTruncated(ProviderError):
    """The provider stopped because the requested completion budget was exhausted."""


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

    @property
    def reasoning_content(self) -> str:
        value = self.message.get("reasoning_content")
        return value if isinstance(value, str) else ""


def _normalize_assistant_message(message: Any) -> tuple[dict[str, Any], tuple[NativeToolCall, ...]]:
    if not isinstance(message, dict):
        raise ProviderShapeError("provider choice.message must be an object")
    if message.get("role") != "assistant":
        raise ProviderShapeError("provider choice.message.role must be 'assistant'")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ProviderShapeError(
            "thinking-mode assistant message requires non-empty reasoning_content"
        )
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ProviderShapeError("provider assistant tool_calls must be a list")
    calls: list[NativeToolCall] = []
    seen_ids: set[str] = set()
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise ProviderShapeError(f"tool_calls[{index}] must be an object")
        if raw_call.get("type") != "function":
            raise ProviderShapeError(f"tool_calls[{index}].type must be 'function'")
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
    # Persist exactly what the provider authored.  Role/type defaults and
    # unexpected fields are semantic audit inputs; normalizing them here would
    # make a malformed response indistinguishable from a valid one.
    return deepcopy(message), tuple(calls)


class DeepSeekNativeClient:
    """Minimal official Chat Completions client with bounded transport retries."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int = 300,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        if self.base_url != OFFICIAL_DEEPSEEK_BASE_URL:
            raise ValueError(
                "new DeepSeek requests require BASE_URL=https://api.deepseek.com"
            )

    @property
    def request_audit_options(self) -> dict[str, Any]:
        return {
            "endpoint": self.base_url + "/chat/completions",
            "model": self.model,
            "tool_choice": "auto",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }

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
            error_cls = ProviderContextOverflow if _looks_like_context_overflow(detail) else ProviderError
            raise error_cls(f"HTTP {exc.code}: {detail[:1200]}") from exc
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
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> NativeAssistantResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        data = self._request_json(request)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderShapeError("provider response must contain choices[0]")
        choice = choices[0]
        try:
            message, calls = _normalize_assistant_message(choice.get("message"))
        except ProviderError as exc:
            exc.accumulated_usage = {
                key: value
                for key, value in (data.get("usage") or {}).items()
                if isinstance(value, int)
            }
            raise
        return NativeAssistantResponse(
            message=message,
            calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=dict(data.get("usage") or {}),
            response_metadata={
                key: data.get(key)
                for key in ("id", "model", "system_fingerprint", "object")
                if data.get(key) is not None
            },
        )

    def request_turn_with_retries(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        retries: int,
        max_completion_tokens: int = 8192,
    ) -> NativeAssistantResponse:
        """Retry transport, length truncation, and zero-call provider-shape defects.

        Multiple authored calls are intentionally returned to the runtime as one semantic protocol
        error.  Argument JSON and unknown functions are likewise never repaired here.
        """

        retry_events: list[dict[str, Any]] = []
        accumulated_usage: dict[str, int] = {}
        budget = max_tokens
        last_error: BaseException | None = None

        def audited_error(exc: ProviderError) -> ProviderError:
            exc.accumulated_usage = dict(accumulated_usage)
            exc.retry_events = deepcopy(retry_events)
            return exc

        def accumulate_failed_usage(exc: ProviderError) -> None:
            for key, value in exc.accumulated_usage.items():
                accumulated_usage[key] = accumulated_usage.get(key, 0) + value

        for attempt in range(max(1, retries)):
            try:
                response = self.request_turn(
                    messages=messages,
                    tools=tools,
                    max_tokens=budget,
                )
                for key, value in response.usage.items():
                    if isinstance(value, int):
                        accumulated_usage[key] = accumulated_usage.get(key, 0) + value
                if response.finish_reason == "length":
                    if attempt + 1 >= max(1, retries) or budget >= max_completion_tokens:
                        raise ProviderCompletionTruncated(
                            "provider completion remained length-truncated after bounded retries"
                        )
                    next_budget = min(max_completion_tokens, max(budget + 1, budget * 2))
                    retry_events.append({
                        "type": "completion_length",
                        "attempt": attempt + 1,
                        "max_tokens": budget,
                        "next_max_tokens": next_budget,
                    })
                    budget = next_budget
                    continue
                if not response.calls:
                    if attempt + 1 >= max(1, retries):
                        raise ProviderShapeError(
                            "provider returned zero tool calls after bounded same-turn retries"
                        )
                    retry_events.append({
                        "type": "missing_tool_calls",
                        "attempt": attempt + 1,
                    })
                    continue
                response.usage = {**response.usage, **accumulated_usage}
                response.retry_events = retry_events
                return response
            except ProviderContextOverflow as exc:
                accumulate_failed_usage(exc)
                raise audited_error(exc)
            except (ProviderCompletionTruncated, ProviderShapeError) as exc:
                last_error = exc
                accumulate_failed_usage(exc)
                if attempt + 1 >= max(1, retries):
                    raise audited_error(exc)
                retry_events.append({
                    "type": "provider_shape",
                    "attempt": attempt + 1,
                    "error": type(exc).__name__,
                })
            except ProviderError as exc:
                last_error = exc
                accumulate_failed_usage(exc)
                if attempt + 1 >= max(1, retries):
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
        )


def tool_result_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
    }
