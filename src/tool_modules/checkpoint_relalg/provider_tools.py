#!/usr/bin/env python3
"""Provider-native tool rendering and strict one-call lowering."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .protocol import (
    MODE_TOOLS,
    NATIVE_ASSISTANT_CARRIER,
    ProtocolValidationError,
    normalize_mode,
    provider_tool_definitions,
    tool_schema_hash,
    validate_arguments,
)


MAX_NATIVE_TOOL_CALLS = 1
MIN_NATIVE_TOOL_CALLS = 1


class NativeToolCallError(ProtocolValidationError):
    """The provider response is not one legal native function call."""


def native_tools(mode: str) -> list[dict[str, Any]]:
    return provider_tool_definitions(mode)


get_provider_tools = native_tools
provider_tools_for_mode = native_tools


def native_tools_sha256(mode: str) -> str:
    return tool_schema_hash(mode)


def _decode_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise NativeToolCallError(
                "native function arguments are not valid JSON",
                code="invalid_json_arguments",
                path="$.tool_calls[0].function.arguments",
            ) from exc
    elif isinstance(raw_arguments, Mapping):
        parsed = deepcopy(dict(raw_arguments))
    else:
        raise NativeToolCallError(
            "native function arguments must be an object or encoded object",
            code="invalid_json_arguments",
            path="$.tool_calls[0].function.arguments",
        )
    if not isinstance(parsed, dict):
        raise NativeToolCallError(
            "native function arguments must decode to an object",
            code="invalid_json_arguments",
            path="$.tool_calls[0].function.arguments",
        )
    return parsed


def validate_native_tool_calls(
    mode: str,
    tool_calls: Sequence[Mapping[str, Any]] | Any,
) -> dict[str, Any]:
    """Lower exactly one provider-native call to a canonical action record."""
    active_mode = normalize_mode(mode)
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        count = len(tool_calls) if isinstance(tool_calls, list) else 0
        raise NativeToolCallError(
            f"each assistant turn requires exactly one native tool call; received {count}",
            code="invalid_tool_call_count",
            path="$.tool_calls",
        )
    call = tool_calls[0]
    if not isinstance(call, Mapping):
        raise NativeToolCallError("native tool call must be an object", path="$.tool_calls[0]")
    if any(not isinstance(name, str) for name in call):
        raise NativeToolCallError(
            "native call field names must be strings",
            code="unexpected_field",
            path="$.tool_calls[0]",
        )
    unexpected = sorted(set(call) - {"id", "type", "function"})
    if unexpected:
        raise NativeToolCallError(
            f"unexpected native call field(s): {', '.join(unexpected)}",
            code="unexpected_field",
            path="$.tool_calls[0]",
        )
    if call.get("type", "function") != "function":
        raise NativeToolCallError(
            "native tool call type must be 'function'",
            path="$.tool_calls[0].type",
        )
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise NativeToolCallError(
            "native tool call requires a function object",
            code="missing_required_field",
            path="$.tool_calls[0].function",
        )
    if any(not isinstance(name, str) for name in function):
        raise NativeToolCallError(
            "native function field names must be strings",
            code="unexpected_field",
            path="$.tool_calls[0].function",
        )
    unexpected_function = sorted(set(function) - {"name", "arguments"})
    if unexpected_function:
        raise NativeToolCallError(
            f"unexpected function field(s): {', '.join(unexpected_function)}",
            code="unexpected_field",
            path="$.tool_calls[0].function",
        )
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise NativeToolCallError(
            "native function name must be non-empty",
            code="missing_required_field",
            path="$.tool_calls[0].function.name",
        )
    arguments = _decode_arguments(function.get("arguments"))
    try:
        validated = validate_arguments(name, arguments, mode=active_mode)
    except ProtocolValidationError as exc:
        raise NativeToolCallError(exc.message, code=exc.code, path=exc.path) from exc
    call_id = call.get("id")
    if call_id is not None and (not isinstance(call_id, str) or not call_id):
        raise NativeToolCallError(
            "native tool call id must be a non-empty string when present",
            path="$.tool_calls[0].id",
        )
    return {"tool": name, "arguments": validated, "tool_call_id": call_id}


parse_native_tool_calls = validate_native_tool_calls
lower_native_tool_calls = validate_native_tool_calls


def validate_native_assistant_message(
    mode: str,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the call-bearing portion of one native assistant message.

    Reasoning and ordinary assistant content are retained only as model-authored audit data;
    neither can become relational evidence.
    """
    if not isinstance(message, Mapping):
        raise NativeToolCallError("assistant message must be an object", path="$.")
    if message.get("role") != "assistant":
        raise NativeToolCallError(
            "native call-bearing message role must be 'assistant'",
            code="invalid_assistant_message",
            path="$.role",
        )
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise NativeToolCallError(
            "thinking-mode assistant message requires non-empty reasoning_content",
            code="invalid_assistant_message",
            path="$.reasoning_content",
        )
    unexpected = sorted(
        str(key)
        for key in message
        if not isinstance(key, str)
        or key not in {"role", "content", "reasoning_content", "tool_calls"}
    )
    if unexpected:
        raise NativeToolCallError(
            f"unexpected assistant message field(s): {', '.join(unexpected)}",
            code="unexpected_field",
            path="$.",
        )
    action = validate_native_tool_calls(mode, message.get("tool_calls"))
    action["reasoning_content"] = message.get("reasoning_content")
    action["assistant_content"] = message.get("content")
    return action


def attempted_action_from_native_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the authored call surface without decoding or repairing it.

    This is deliberately independent of validation.  It is the single source
    for state-preserving rejection envelopes and audit replay, preventing a
    stored ``attempted_action`` from becoming an unaudited assertion.
    """

    raw_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    calls = raw_calls if isinstance(raw_calls, list) else []
    names: list[Any] = []
    arguments: list[Any] = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            names.append(deepcopy(function.get("name")))
            arguments.append(deepcopy(function.get("arguments")))
        else:
            names.append(None)
            arguments.append(None)
    return {
        "tool": names[0] if len(names) == 1 else names,
        "arguments": arguments[0] if len(arguments) == 1 else arguments,
    }


__all__ = [
    "MAX_NATIVE_TOOL_CALLS",
    "MIN_NATIVE_TOOL_CALLS",
    "MODE_TOOLS",
    "NATIVE_ASSISTANT_CARRIER",
    "NativeToolCallError",
    "get_provider_tools",
    "lower_native_tool_calls",
    "native_tools",
    "native_tools_sha256",
    "parse_native_tool_calls",
    "provider_tools_for_mode",
    "validate_native_assistant_message",
    "validate_native_tool_calls",
    "attempted_action_from_native_message",
]
