#!/usr/bin/env python3
"""Strict raw-text JSON carrier for the checkpoint-relalg diagnostic A/B.

This module only lowers the exact provider-authored visible content.  It never
extracts JSON from prose, strips fences, repairs arguments, or executes a tool.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from .protocol import (
    DEFAULT_ATOMIC_OPERATOR_PROFILE,
    ProtocolValidationError,
    validate_model_action,
)


TEXT_RESULT_MESSAGE_TYPE = "checkpoint_relalg_tool_result"


class TextJSONActionError(ProtocolValidationError):
    """One text-json assistant message is not one exact canonical action."""


def _decode_visible_action(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise TextJSONActionError(
            "text-json visible content must be a non-empty JSON object",
            code="invalid_text_json_content",
            path="$.content",
        )
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TextJSONActionError(
            "text-json visible content is not one raw JSON object",
            code="invalid_text_json_content",
            path="$.content",
        ) from exc
    if not isinstance(value, dict):
        raise TextJSONActionError(
            "text-json visible content must decode to an object",
            code="invalid_text_json_content",
            path="$.content",
        )
    return value


def validate_text_json_assistant_message(
    mode: str,
    message: Mapping[str, Any],
    *,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> dict[str, Any]:
    """Lower one exact ``{tool, arguments}`` object from visible content."""

    if not isinstance(message, Mapping):
        raise TextJSONActionError("assistant message must be an object", path="$.")
    if message.get("role") != "assistant":
        raise TextJSONActionError(
            "text-json call-bearing message role must be 'assistant'",
            code="invalid_assistant_message",
            path="$.role",
        )
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise TextJSONActionError(
            "thinking-mode assistant message requires non-empty reasoning_content",
            code="invalid_assistant_message",
            path="$.reasoning_content",
        )
    unexpected = sorted(
        str(key)
        for key in message
        if not isinstance(key, str)
        or key not in {"role", "content", "reasoning_content"}
    )
    if unexpected:
        raise TextJSONActionError(
            f"unexpected assistant message field(s): {', '.join(unexpected)}",
            code="unexpected_field",
            path="$.",
        )
    raw_action = _decode_visible_action(message.get("content"))
    try:
        action = validate_model_action(
            raw_action,
            mode=mode,
            atomic_operator_profile=atomic_operator_profile,
        )
    except ProtocolValidationError as exc:
        raise TextJSONActionError(exc.message, code=exc.code, path=exc.path) from exc
    return {
        **action,
        "reasoning_content": reasoning,
        "assistant_content": message.get("content"),
    }


def attempted_action_from_text_json_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the authored surface for deterministic rejection provenance."""

    content = message.get("content") if isinstance(message, Mapping) else None
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        return {
            "tool": deepcopy(value.get("tool")),
            "arguments": deepcopy(value.get("arguments")),
        }
    return {"tool": None, "arguments": deepcopy(content)}


def text_json_authored_action_count(message: Mapping[str, Any]) -> int:
    """Count authored JSON action objects without retaining additional content."""

    content = message.get("content") if isinstance(message, Mapping) else None
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        return 0
    if isinstance(value, dict):
        return 1
    if isinstance(value, list):
        return sum(isinstance(item, dict) for item in value)
    return 0


def text_json_exact_single_action(message: Mapping[str, Any]) -> bool:
    """Check the raw visible action wrapper without validating tool semantics."""

    content = message.get("content") if isinstance(message, Mapping) else None
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"tool", "arguments"}
        and isinstance(value.get("tool"), str)
        and bool(value["tool"])
        and isinstance(value.get("arguments"), dict)
    )


def text_json_carrier_envelope_valid(message: Mapping[str, Any]) -> bool:
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return False
    if not isinstance(message.get("reasoning_content"), str) or not message[
        "reasoning_content"
    ].strip():
        return False
    if set(message) != {"role", "content", "reasoning_content"}:
        return False
    return text_json_exact_single_action(message)


def text_json_result_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return causal Harness feedback without using an unmatched ``role=tool``."""

    envelope = {
        "type": TEXT_RESULT_MESSAGE_TYPE,
        "result": deepcopy(dict(payload)),
    }
    return {
        "role": "user",
        "content": json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    }


def decode_text_json_result_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly decode one deterministic text-carrier feedback message."""

    if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
        raise ValueError("text-json feedback message must contain role and content")
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("text-json feedback message must be a user JSON message")
    value = json.loads(message["content"])
    if not isinstance(value, dict) or set(value) != {"type", "result"}:
        raise ValueError("text-json feedback envelope has an invalid shape")
    if value.get("type") != TEXT_RESULT_MESSAGE_TYPE or not isinstance(
        value.get("result"), dict
    ):
        raise ValueError("text-json feedback envelope has an invalid type or result")
    return value["result"]


__all__ = [
    "TEXT_RESULT_MESSAGE_TYPE",
    "TextJSONActionError",
    "attempted_action_from_text_json_message",
    "decode_text_json_result_message",
    "text_json_result_message",
    "text_json_authored_action_count",
    "text_json_carrier_envelope_valid",
    "text_json_exact_single_action",
    "validate_text_json_assistant_message",
]
