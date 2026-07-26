"""Serialize and parse the model-visible envelope around one structured tool action.

Carrier syntax is transport, not tool semantics. The active carrier keeps one non-empty
``<think>`` block followed by one raw JSON action object. The former tagged JSON carrier remains
available only for deterministic migration of existing artifacts.
"""
from __future__ import annotations

import json
import re
from typing import Any


ACTIVE_ACTION_CARRIER = "think-json-v1"
LEGACY_TAGGED_ACTION_CARRIER = "think-tagged-json-v1"

_THINK_JSON_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*(?P<action>\{.*\})\s*\Z",
    re.DOTALL,
)
_THINK_TAGGED_JSON_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<tool_call>\s*(?P<action>\{.*\})\s*</tool_call>\s*\Z",
    re.DOTALL,
)


class ActionCarrierError(ValueError):
    """The visible assistant text is not one exact action envelope."""


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def render_action_carrier(
    think: str,
    tool: str,
    arguments: dict[str, Any],
) -> str:
    return (
        f"<think>{think}</think>\n"
        f"{_compact({'tool': tool, 'arguments': arguments})}"
    )


def _parse_match(
    text: str,
    pattern: re.Pattern[str],
    *,
    expected: str,
) -> tuple[str, dict[str, Any]]:
    match = pattern.fullmatch(text)
    if match is None:
        raise ActionCarrierError(expected)
    think = match.group("think").strip()
    if not think:
        raise ActionCarrierError("expected one non-empty <think>...</think> block")
    try:
        action = json.loads(match.group("action"))
    except json.JSONDecodeError as exc:
        raise ActionCarrierError(f"action is not valid JSON: {exc}") from exc
    if not isinstance(action, dict):
        raise ActionCarrierError("action JSON must be an object")
    return think, action


def parse_action_carrier(text: str) -> tuple[str, dict[str, Any]]:
    """Strictly parse the only model-visible carrier accepted by active runtimes."""
    if "<tool_call>" in text or "</tool_call>" in text:
        raise ActionCarrierError(
            "tool_call tags are not part of the active carrier; emit the JSON object directly"
        )
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        raise ActionCarrierError(
            "expected exactly one <think>...</think> block followed by one JSON action object"
        )
    return _parse_match(
        text,
        _THINK_JSON_RE,
        expected=(
            "expected exactly one <think>...</think> block followed by one JSON action object "
            "and nothing else"
        ),
    )


def parse_legacy_tagged_action_carrier(
    text: str,
) -> tuple[str, dict[str, Any]]:
    """Strictly parse the retired carrier for offline artifact migration only."""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        raise ActionCarrierError("legacy action must contain exactly one think block")
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ActionCarrierError("legacy action must contain exactly one tool_call block")
    return _parse_match(
        text,
        _THINK_TAGGED_JSON_RE,
        expected="expected one complete legacy think + tagged JSON action",
    )
