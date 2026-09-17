"""Serialize and parse the model-visible envelope around one structured tool action.

Carrier syntax is transport, not tool semantics. The active student carrier keeps one
non-empty ``<think>`` block followed by one raw JSON action object. The former tagged
JSON carrier remains available only for deterministic migration of existing artifacts.
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

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_action_carrier",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def render_action_carrier(think: str, tool: str, arguments: dict[str, Any]) -> str:
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
        raise ActionCarrierError(expected, code="carrier_shape_error")
    think = match.group("think").strip()
    if not think:
        raise ActionCarrierError(
            "expected one non-empty <think>...</think> block",
            code="empty_think",
        )
    try:
        action = json.loads(match.group("action"))
    except json.JSONDecodeError as exc:
        raise ActionCarrierError(
            f"action is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
            code="invalid_action_json",
            details={
                "json_error": exc.msg,
                "line": exc.lineno,
                "column": exc.colno,
            },
        ) from exc
    if not isinstance(action, dict):
        raise ActionCarrierError(
            "action JSON must be an object",
            code="action_json_not_object",
            details={"received_type": type(action).__name__},
        )
    return think, action


_FENCED_ACTION_RE = re.compile(
    r"```[A-Za-z0-9_-]*\s*(?P<body>.*?)```",
    re.DOTALL,
)


def _iter_balanced_json_objects(text: str) -> list[str]:
    """Spans of top-level ``{...}`` objects, ignoring braces inside JSON strings."""
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def repair_action_carrier(text: str) -> tuple[str, str] | None:
    """Best-effort transport repair for one carrier-shaped slip.

    This exists for the **runtime** only: a transport typo must not be scored as a reasoning
    failure.  Only unambiguous slips are repaired, and the repaired text is re-parsed by the
    strict carrier afterwards, so tool/argument semantics are unchanged:

    * an unclosed ``<think>`` block (``</think>`` missing);
    * the action object wrapped in a markdown code fence;
    * extra prose before/after the action object.

    Retired or ambiguous carriers stay hard errors: ``<tool_call>`` tags, duplicate
    ``<think>`` blocks, an empty think block, or any text without a schema-shaped action
    object. Returns ``(repaired_text, kind)`` or ``None`` when no repair applies.
    """
    if "<tool_call>" in text or "</tool_call>" in text:
        return None
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    if open_count > 1 or close_count > 1 or (open_count == 0 and close_count == 1):
        return None
    if open_count == 1 and close_count == 0:
        think_raw, _, rest = text.partition("<think>")[2].partition("{")
        think = think_raw.strip()
        rest = "{" + rest
        kind = "unclosed_think"
    elif open_count == 1 and close_count == 1:
        match = re.match(r"\A\s*<think>(?P<think>.*?)</think>", text, re.DOTALL)
        if match is None:
            return None
        think = match.group("think").strip()
        rest = text[match.end() :]
        kind = "extra_text_outside_action"
    else:
        return None
    if not think:
        return None
    fenced = _FENCED_ACTION_RE.search(rest)
    if fenced is not None:
        rest = fenced.group("body")
        kind = "code_fence"
    for source in (rest, text):
        for candidate in reversed(_iter_balanced_json_objects(source)):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
                continue
            if not isinstance(value.get("tool"), str) or not value["tool"]:
                continue
            if not isinstance(value.get("arguments"), dict):
                continue
            return f"<think>{think}</think>\n{candidate}", kind
    return None


def parse_action_carrier(text: str) -> tuple[str, dict[str, Any]]:
    """Strictly parse the only model-visible carrier accepted by active runtimes."""
    if "<tool_call>" in text or "</tool_call>" in text:
        raise ActionCarrierError(
            "tool_call tags are not part of the active carrier; emit the JSON object directly",
            code="legacy_tool_call_tags",
        )
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    if open_count == 1 and close_count == 0:
        raise ActionCarrierError(
            "the <think> block is not closed; add exactly one </think> before the JSON action",
            code="unclosed_think",
            details={"open_think_tags": open_count, "close_think_tags": close_count},
        )
    if open_count == 0 and close_count == 1:
        raise ActionCarrierError(
            "a closing </think> tag appears without one opening <think> tag",
            code="unexpected_closing_think",
            details={"open_think_tags": open_count, "close_think_tags": close_count},
        )
    if open_count == 0 and close_count == 0:
        raise ActionCarrierError(
            "missing the required non-empty <think>...</think> block before the JSON action",
            code="missing_think",
            details={"open_think_tags": open_count, "close_think_tags": close_count},
        )
    if open_count != 1 or close_count != 1:
        raise ActionCarrierError(
            "expected exactly one non-nested <think>...</think> block",
            code="multiple_or_nested_think",
            details={"open_think_tags": open_count, "close_think_tags": close_count},
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
