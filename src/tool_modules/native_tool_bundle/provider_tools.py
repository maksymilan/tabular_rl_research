"""DeepSeek-native function-call carriers for the table-tool agent.

Version50 changes only the provider-facing carrier and requires one call. Version51 follows the
provider's documented multi-call semantics: one assistant turn may contain a bounded bundle of
independent atomic calls, with one tool-result message returned for every call. The harness remains
the authority for validation, execution, resident state, and terminal scoring.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

from prompt_contract import CANONICAL_ACTION_RULE
from protocol import (
    MODEL_ARG_SCHEMA,
    TOOL_SPECS,
    assistant_message,
    compact_resident_observation,
    first_user_message,
    parse_assistant_strict,
    state_context_message,
)


NATIVE_ASSISTANT_CARRIER = "deepseek-native-function-call-v1"
NATIVE_BUNDLE_ASSISTANT_CARRIER = "deepseek-native-tool-bundle-v1"
MAX_NATIVE_BUNDLE_CALLS = 8
NATIVE_ACTION_RULE = (
    "1. Each turn, use the provider's native function-calling interface to emit exactly one "
    "function call. Put non-empty reasoning only in the native reasoning field. Do not write an "
    "action, tool arguments, or answer data in assistant content, and never emit a second call."
)


class NativeToolCallError(RuntimeError):
    """Provider transport or native-carrier response failure."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def native_system_prompt(system_prompt: str) -> str:
    """Replace only the canonical text-carrier rule with the native function-call rule."""
    count = system_prompt.count(CANONICAL_ACTION_RULE)
    if count != 1:
        raise ValueError(
            "native carrier requires the canonical student prompt with exactly one action rule; "
            f"found {count}"
        )
    return system_prompt.replace(CANONICAL_ACTION_RULE, NATIVE_ACTION_RULE)


def _array(item_schema: dict[str, Any], *, min_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": item_schema}
    if min_items is not None:
        schema["minItems"] = min_items
    return schema


def _object(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
    additional_properties: bool = True,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    return schema


_STRING = {"type": "string"}
_NONEMPTY_STRING = {"type": "string", "minLength": 1}
_STRING_ARRAY = _array(_NONEMPTY_STRING, min_items=1)
_PREDICATE = {
    "description": "Typed predicate tree accepted by the harness.",
    "anyOf": [{"type": "object"}, {"type": "array"}],
}
_SCALAR = {
    "description": "A non-null JSON scalar.",
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
    ],
}

_PLAN_OP = _object(
    {
        "op": {"type": "string", "enum": ["create", "add", "update", "delete"]},
        "id": _NONEMPTY_STRING,
        "goal": _STRING,
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "done", "blocked"],
        },
        "evidence": _NONEMPTY_STRING,
    }
)
_JOIN_EDGE = _object(
    {"left": _NONEMPTY_STRING, "right": _NONEMPTY_STRING},
    required=["left", "right"],
    additional_properties=False,
)
_JOIN_ITEM = _object(
    {
        "table": _NONEMPTY_STRING,
        "on": _array(_JOIN_EDGE),
        "type": {"type": "string", "enum": ["inner", "left", "cross"]},
        "role": _NONEMPTY_STRING,
    },
    required=["table", "on"],
    additional_properties=False,
)
_AGGREGATION = _object(
    {
        "op": {
            "type": "string",
            "enum": ["sum", "count", "count_distinct", "mean", "min", "max"],
        },
        "column": _NONEMPTY_STRING,
        "as": _NONEMPTY_STRING,
        "where": _PREDICATE,
    },
    required=["op", "column", "as"],
    additional_properties=False,
)
_SCALAR_OPERAND = _object(
    {
        "value": _SCALAR,
        "value_ref": _NONEMPTY_STRING,
        "column": _NONEMPTY_STRING,
    }
)
_PROJECT_EXPRESSION = {
    "anyOf": [
        _NONEMPTY_STRING,
        _object(
            {
                "op": {"type": "string", "enum": ["date_diff_days", "extract_year"]},
                "operands": _array(
                    _object({"column": _NONEMPTY_STRING, "value": _SCALAR}),
                    min_items=1,
                ),
                "as": _NONEMPTY_STRING,
            },
            required=["op", "operands", "as"],
            additional_properties=False,
        ),
    ]
}


NATIVE_PROPERTY_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "plan": {"ops": _array(_PLAN_OP, min_items=1)},
    "describe_table": {"tables": _STRING_ARRAY},
    "inspect_column": {
        "table": _NONEMPTY_STRING,
        "column": _NONEMPTY_STRING,
        "top_k": {"type": "integer", "minimum": 1},
    },
    "read_subtable": {
        "table": _NONEMPTY_STRING,
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        "columns": _STRING_ARRAY,
        "conditions": _PREDICATE,
        "order_by": _STRING_ARRAY,
        "offset": {"type": "integer", "minimum": 0},
    },
    "condition_filter": {
        "table": _NONEMPTY_STRING,
        "conditions": _PREDICATE,
        "return_columns": _STRING_ARRAY,
    },
    "project": {
        "table": _NONEMPTY_STRING,
        "expressions": _array(_PROJECT_EXPRESSION, min_items=1),
        "distinct": {"type": "boolean"},
    },
    "scalar_compute": {
        "operation": {
            "type": "string",
            "enum": [
                "add",
                "subtract",
                "multiply",
                "divide",
                "percent",
                "percent_change",
                "date_diff_days",
            ],
        },
        "operands": _array(_SCALAR_OPERAND, min_items=2),
        "result_name": _NONEMPTY_STRING,
    },
    "join_tables": {
        "base": _NONEMPTY_STRING,
        "joins": _array(_JOIN_ITEM, min_items=1),
        "base_role": _NONEMPTY_STRING,
    },
    "group_aggregate": {
        "table": _NONEMPTY_STRING,
        "group_by": _array(_NONEMPTY_STRING),
        "aggregations": _array(_AGGREGATION, min_items=1),
        "passthrough": _array(_NONEMPTY_STRING),
        "output_layout": {"type": "string", "enum": ["rows", "columns"]},
        "category_values": _array(_SCALAR, min_items=1),
        "output_columns": _STRING_ARRAY,
    },
    "extreme_value_select": {
        "table": _NONEMPTY_STRING,
        "order_by": _STRING_ARRAY,
        "top_k": {"type": "integer", "minimum": 1},
        "return_columns": _STRING_ARRAY,
    },
    "set_op": {
        "left": _NONEMPTY_STRING,
        "right": _NONEMPTY_STRING,
        "op": {
            "type": "string",
            "enum": ["union", "union_all", "intersect", "except"],
        },
    },
    "answer_from_context": {
        "evidence": _object(
            {"table": _NONEMPTY_STRING},
            required=["table"],
            additional_properties=False,
        ),
        "reason": _STRING,
    },
}


def _validate_native_schema_coverage() -> None:
    if set(NATIVE_PROPERTY_SCHEMAS) != set(MODEL_ARG_SCHEMA):
        raise RuntimeError("native schemas and the canonical atomic tool surface have drifted")
    for tool, (required, optional) in MODEL_ARG_SCHEMA.items():
        expected = required | optional
        actual = set(NATIVE_PROPERTY_SCHEMAS[tool])
        if actual != expected:
            raise RuntimeError(
                f"native schema for {tool} has fields {sorted(actual)}; expected {sorted(expected)}"
            )


_validate_native_schema_coverage()


def native_atomic_tools(
    model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
) -> list[dict[str, Any]]:
    """Render one explicit public atomic surface as provider-native function tools."""
    active_schema = MODEL_ARG_SCHEMA if model_arg_schema is None else model_arg_schema
    tools: list[dict[str, Any]] = []
    for name, (required, _optional) in active_schema.items():
        if name not in NATIVE_PROPERTY_SCHEMAS or name not in TOOL_SPECS:
            raise ValueError(f"cannot render unknown native tool {name!r}")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_SPECS[name],
                    "parameters": {
                        "type": "object",
                        "properties": deepcopy(NATIVE_PROPERTY_SCHEMAS[name]),
                        "required": sorted(required),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def native_tools_sha256(
    model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
) -> str:
    payload = json.dumps(
        native_atomic_tools(model_arg_schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_messages_to_native(
    messages: list[dict[str, Any]],
    native_assistant_history: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical legal-history pairs into native assistant/tool messages.

    Live responses use the cached original assistant message, preserving DeepSeek's exact
    ``reasoning_content`` and tool-call ID.  The deterministic fallback exists for replay/tests.
    """
    canonical_assistants = [
        message.get("content")
        for message in messages
        if message.get("role") == "assistant"
    ]
    resolved_cache: list[dict[str, Any] | None] = [None] * len(canonical_assistants)
    history = native_assistant_history or []
    # Legal history is an ordered subsequence of all provider responses because rejected calls
    # are omitted. Resolve from the newest end so repeated non-adjacent canonical calls keep their
    # own provider IDs instead of collapsing through a text-keyed dictionary.
    history_cursor = len(history)
    for assistant_index in range(len(canonical_assistants) - 1, -1, -1):
        canonical = canonical_assistants[assistant_index]
        for history_index in range(history_cursor - 1, -1, -1):
            if history[history_index][0] == canonical:
                resolved_cache[assistant_index] = history[history_index][1]
                history_cursor = history_index
                break
    converted: list[dict[str, Any]] = []
    pending_tool_call_id: str | None = None
    fallback_index = 0
    assistant_index = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            if pending_tool_call_id is not None:
                raise ValueError("native history contains adjacent assistant calls without a result")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"canonical assistant message {index} is empty")
            native_message = resolved_cache[assistant_index]
            assistant_index += 1
            if native_message is None:
                think, tool, arguments = parse_assistant_strict(content)
                fallback_index += 1
                pending_tool_call_id = f"call_replay_{fallback_index}"
                native_message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": think,
                    "tool_calls": [
                        {
                            "id": pending_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(
                                    arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                }
            else:
                native_message = deepcopy(native_message)
                tool_calls = native_message.get("tool_calls")
                if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                    raise ValueError("cached native assistant message must contain one tool call")
                pending_tool_call_id = tool_calls[0].get("id")
                if not isinstance(pending_tool_call_id, str) or not pending_tool_call_id:
                    raise ValueError("cached native tool call has no id")
            converted.append(native_message)
            continue
        if role == "user" and pending_tool_call_id is not None:
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"tool result message {index} is empty")
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_tool_call_id,
                    "content": content,
                }
            )
            pending_tool_call_id = None
            continue
        if role not in {"system", "user"}:
            raise ValueError(f"unsupported canonical history role at {index}: {role!r}")
        if pending_tool_call_id is not None:
            raise ValueError("native tool call must be followed immediately by its tool result")
        converted.append(deepcopy(message))
    if pending_tool_call_id is not None:
        raise ValueError("native history ends with an unresolved tool call")
    return converted


def native_bundle_history_messages(
    *,
    system_prompt: str,
    overview: dict[str, Any],
    question: str,
    state: dict[str, Any] | None,
    last_error: dict[str, Any] | None,
    external_knowledge: str | None,
    history: list[dict[str, Any]],
    history_turns: int,
) -> list[dict[str, Any]]:
    """Render bounded provider-native assistant turns and all of their tool results.

    A history item represents one actual model turn, not one primitive execution. Its assistant
    message therefore retains the complete native ``tool_calls`` list and is followed by exactly
    one ``role=tool`` result for every call id, in provider order. Observations use the same compact
    resident rendering as version39; the current resident state is appended only to the final tool
    result (or the opening user message when history is empty).
    """
    if history_turns < 1:
        raise ValueError("native bundle history_turns must be positive")
    initial = first_user_message(overview, question, external_knowledge)
    retained = history[-history_turns:]
    if not retained:
        if state or last_error:
            initial += "\n\n" + state_context_message(state, last_error)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial},
        ]

    rendered: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial},
    ]
    for history_index, item in enumerate(retained):
        assistant = item.get("assistant")
        tool_messages = item.get("tool_messages")
        if not isinstance(assistant, dict):
            raise ValueError("native bundle history assistant must be an object")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("native bundle history assistant has no tool calls")
        if not isinstance(tool_messages, list) or len(tool_messages) != len(calls):
            raise ValueError(
                "native bundle history must contain one tool result for every tool call"
            )
        expected_ids = [call.get("id") for call in calls if isinstance(call, dict)]
        actual_ids = [message.get("tool_call_id") for message in tool_messages]
        if expected_ids != actual_ids:
            raise ValueError("native bundle tool results do not preserve provider call order")
        rendered.append(deepcopy(assistant))
        for message_index, message in enumerate(tool_messages):
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("native bundle tool result content must be non-empty")
            compact = compact_resident_observation(content)
            if (
                history_index == len(retained) - 1
                and message_index == len(tool_messages) - 1
            ):
                compact += "\n\n" + state_context_message(state, last_error)
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": compact,
                }
            )
    return rendered


def _is_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return (
        "maximum context length" in lowered
        or ("context length" in lowered and "max_tokens" in lowered)
        or "too many tokens" in lowered
    )


class DeepSeekNativeAtomicDriver:
    """Callable adapter matching ``rollout.chat`` while using DeepSeek native tools."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        reasoning_effort: str = "high",
        timeout_seconds: int = 300,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.events: list[dict[str, Any]] = []
        self._native_assistant_history: list[tuple[str, dict[str, Any]]] = []

    def _safe_http_error(self, status: int, raw_body: str) -> NativeToolCallError:
        message = "provider rejected the request"
        try:
            payload = json.loads(raw_body)
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                parts = [error.get("type"), error.get("code"), error.get("message")]
                message = " | ".join(str(part) for part in parts if part)
        except json.JSONDecodeError:
            pass
        if self.api_key:
            message = message.replace(self.api_key, "***")
        return NativeToolCallError(f"DeepSeek HTTP {status}: {message[:500]}", status=status)

    def _post_once(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": native_atomic_tools(),
            # DeepSeek thinking mode currently rejects tool_choice="required".  The prompt asks
            # for one call, while _lower_response enforces exactly one before the harness sees it.
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "thinking": {"type": "enabled"},
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise self._safe_http_error(exc.code, raw) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NativeToolCallError("DeepSeek returned non-JSON response data") from exc
        if not isinstance(parsed, dict):
            raise NativeToolCallError("DeepSeek response root must be an object")
        return parsed

    def _lower_response(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise NativeToolCallError("DeepSeek response contains no completion choice")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise NativeToolCallError("DeepSeek native tool completion was length-truncated")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise NativeToolCallError("DeepSeek completion choice contains no assistant message")
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            count = len(tool_calls) if isinstance(tool_calls, list) else 0
            raise NativeToolCallError(
                f"native atomic carrier requires exactly one tool call; provider returned {count}"
            )
        call = tool_calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            raise NativeToolCallError("native tool call has no function payload")
        tool = function.get("name")
        if tool not in MODEL_ARG_SCHEMA:
            raise NativeToolCallError(f"provider returned unknown atomic tool {tool!r}")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise NativeToolCallError("native tool arguments are not valid JSON") from exc
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
            raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        else:
            raise NativeToolCallError("native tool arguments must be a JSON object string")
        if not isinstance(arguments, dict):
            raise NativeToolCallError("native tool arguments must decode to an object")
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise NativeToolCallError("thinking-mode native tool call has empty reasoning_content")
        content = message.get("content")
        if content is not None and (not isinstance(content, str) or content.strip()):
            raise NativeToolCallError(
                "native tool call also contained assistant content; refusing a dual carrier"
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise NativeToolCallError("native tool call has no id")
        normalized_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": raw_arguments},
        }
        canonical = assistant_message(reasoning.strip(), tool, arguments)
        self._native_assistant_history.append(
            (
                canonical,
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": [normalized_call],
                },
            )
        )
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        self.events.append(
            {
                "request_index": len(self.events),
                "response_id": response.get("id"),
                "provider_model": response.get("model"),
                "system_fingerprint": response.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason"),
                "tool_call_id": call_id,
                "tool": tool,
                "reasoning_content_present": True,
                "assistant_content_present": bool(content),
                "usage": deepcopy(usage),
            }
        )
        return canonical

    def __call__(
        self,
        base_url: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        retries: int,
        min_context_retry_tokens: int = 128,
        retry_stats: dict[str, Any] | None = None,
    ) -> str:
        request_base = base_url.rstrip("/")
        if request_base != self.base_url:
            raise ValueError("runtime base URL differs from the configured DeepSeek base URL")
        native_messages = canonical_messages_to_native(
            messages, self._native_assistant_history
        )
        current_max_tokens = max_tokens
        transport_retries = 0
        context_retries = 0
        while True:
            try:
                response = self._post_once(
                    base_url=request_base,
                    model=model,
                    messages=native_messages,
                    max_tokens=current_max_tokens,
                )
                canonical = self._lower_response(response)
                if retry_stats is not None:
                    retry_stats.update(
                        {
                            "api_request_attempts": transport_retries + context_retries + 1,
                            "api_transport_retries": transport_retries,
                            "api_context_retries": context_retries,
                        }
                    )
                return canonical
            except NativeToolCallError as exc:
                if _is_context_overflow(str(exc)) and current_max_tokens > min_context_retry_tokens:
                    current_max_tokens = max(min_context_retry_tokens, current_max_tokens // 2)
                    context_retries += 1
                    continue
                if (
                    exc.status is not None
                    and (exc.status == 429 or exc.status >= 500)
                    and transport_retries < retries
                ):
                    transport_retries += 1
                    time.sleep(min(2**transport_retries, 8))
                    continue
                raise
            except (TimeoutError, urllib.error.URLError) as exc:
                if transport_retries < retries:
                    transport_retries += 1
                    time.sleep(min(2**transport_retries, 8))
                    continue
                raise NativeToolCallError(f"DeepSeek transport failed: {type(exc).__name__}") from exc
