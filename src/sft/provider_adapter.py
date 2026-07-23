"""Provider-specific transport adapters before the shared strict protocol parser.

Adapters map an API provider's *separate response fields* into the canonical assistant envelope.
They are deliberately narrower than parser repair: raw content is preserved, tool JSON is never
changed, and malformed XML/JSON remains a protocol error after adaptation.
"""
from __future__ import annotations

import json
import re
from typing import Any


DEEPSEEK_V4_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
DEEPSEEK_V4_DEFAULT_MAX_TOKENS = 2048
DEEPSEEK_V4_REASONING_EFFORT = "high"
DEEPSEEK_CARRIER_JSON_OUTPUT = "json-output"
DEEPSEEK_CARRIER_TOOL_CALL = "tool-call"
DEEPSEEK_CARRIER_CHOICES = (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    DEEPSEEK_CARRIER_TOOL_CALL,
)
_CANONICAL_SYSTEM_RESPONSE_RULE = (
    '1. Each turn, output exactly: <think>brief reasoning</think> then '
    '<tool_call>{"tool": "<name>", "arguments": {...}}</tool_call>. Nothing else.'
)
_SPLIT_SYSTEM_RESPONSE_RULE = (
    "1. Produce exactly one tool action per turn using the provider-specific response envelope "
    "at the end of this prompt."
)
_CANONICAL_COMPACT_RESPONSE_RULE = (
    "Each turn output exactly:\n"
    "<think>brief reason for this action</think>\n"
    '<tool_call>{"tool":"...","arguments":{...}}</tool_call>\n'
    "No text outside these tags."
)
_SPLIT_COMPACT_RESPONSE_RULE = (
    "Each turn produces exactly one tool action using the provider-specific response envelope at "
    "the end of this prompt."
)
_CANONICAL_ROLLING_COMPACT_RESPONSE_RULE = (
    "Output only <think>specific reason for the next action</think> followed by one complete "
    '<tool_call>{"tool":"name","arguments":{...}}</tool_call>. No prose outside the tags, no '
    "second action, no shorthand JSON, and no legacy tool fields."
)
_SPLIT_ROLLING_COMPACT_RESPONSE_RULE = (
    "Produce exactly one tool action using the provider-specific response envelope at the end of "
    "this prompt. Do not emit a second action, shorthand JSON, or legacy tool fields."
)
_CANONICAL_GENERATION_RESPONSE_RULE = (
    "ONE REQUEST = ONE ACTION. Emit exactly one non-empty <think> block and exactly one "
    "<tool_call> block. Immediately STOP after that closing </tool_call>: never emit a second "
    "<think>, a second tool call, a numbered plan of calls, or a complete multi-step solution in "
    "one response. The harness will execute only this one action and return a fresh state before "
    "you choose the next action. Your <think> block must be non-empty on every turn. Put the reason "
    "inside <think> tags, not as plain text before the tool call."
)
_SPLIT_GENERATION_RESPONSE_RULE = (
    "ONE REQUEST = ONE ACTION. Produce one non-empty brief action reason in the provider's native "
    "reasoning field and exactly one action in the provider's visible field, using the "
    "provider-specific response envelope at the end of this prompt. Immediately STOP after that "
    "single action: never emit a second reason, a second action, a numbered "
    "plan of calls, or a complete multi-step solution in one response. The harness will execute "
    "only this one action and return a fresh state before you choose the next action. Follow the "
    "provider-specific field placement at the end of this prompt."
)
_CANONICAL_ASSISTANT_HISTORY_RE = re.compile(
    r"^\s*<think>(?P<reasoning>.*?)</think>\s*"
    r"<tool_call>\s*(?P<call_json>\{.*\})\s*</tool_call>\s*$",
    re.DOTALL,
)

def is_deepseek_split_model(model: str) -> bool:
    return str(model).strip().lower() in DEEPSEEK_V4_MODELS


def provider_default_max_tokens(model: str, fallback: int) -> int:
    """Use a larger explicit budget for the known split-response provider only."""
    return DEEPSEEK_V4_DEFAULT_MAX_TOKENS if is_deepseek_split_model(model) else fallback


def provider_request_options(
    model: str,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> dict[str, Any]:
    """Return provider controls that must be explicit and auditable."""
    if not is_deepseek_split_model(model):
        return {}
    if carrier not in DEEPSEEK_CARRIER_CHOICES:
        raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
    options = {
        "thinking": {"type": "enabled"},
        "reasoning_effort": DEEPSEEK_V4_REASONING_EFFORT,
    }
    if carrier == DEEPSEEK_CARRIER_JSON_OUTPUT:
        options["response_format"] = {"type": "json_object"}
    return options


def provider_system_prompt(
    model: str,
    canonical_prompt: str,
    *,
    example_visible_content: str | None = None,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return one unambiguous API-facing response contract for the selected provider.

    The internal trajectory protocol remains canonical ``<think>`` + ``<tool_call>``. A provider
    with native reasoning transport must not see that canonical envelope as a competing positive
    output instruction, so every known canonical response clause is replaced before the split-field
    contract is appended. Tool descriptions and interface-specific examples are left untouched.
    """
    if not is_deepseek_split_model(model):
        return canonical_prompt
    response_replacements = (
        (_CANONICAL_SYSTEM_RESPONSE_RULE, _SPLIT_SYSTEM_RESPONSE_RULE),
        (_CANONICAL_COMPACT_RESPONSE_RULE, _SPLIT_COMPACT_RESPONSE_RULE),
        (
            _CANONICAL_ROLLING_COMPACT_RESPONSE_RULE,
            _SPLIT_ROLLING_COMPACT_RESPONSE_RULE,
        ),
    )
    matched = [
        (canonical, split)
        for canonical, split in response_replacements
        if canonical in canonical_prompt
    ]
    if len(matched) != 1 or _CANONICAL_GENERATION_RESPONSE_RULE not in canonical_prompt:
        raise ValueError(
            "DeepSeek API-facing prompt cannot remove canonical response clauses; "
            "the prompt template drifted"
        )
    canonical_response, split_response = matched[0]
    prompt = canonical_prompt.replace(
        canonical_response,
        split_response,
        1,
    ).replace(
        _CANONICAL_GENERATION_RESPONSE_RULE,
        _SPLIT_GENERATION_RESPONSE_RULE,
        1,
    )
    return prompt + provider_instruction(
        model,
        example_visible_content=example_visible_content,
        carrier=carrier,
    )


def provider_request_messages(
    model: str,
    messages: list[dict],
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> list[dict]:
    """Render canonical legal-history actions in the provider's API-facing carrier.

    DeepSeek receives prior assistant actions as one raw JSON action object. Their old reasoning is
    deliberately omitted: resident state and tool observations are authoritative, and replaying a
    canonical ``<think>`` envelope in visible history contradicts the current split-field contract.
    The returned list is a copy; canonical audit/history records are not mutated.
    """
    rendered = [dict(message) for message in messages]
    if not is_deepseek_split_model(model):
        return rendered
    if carrier not in DEEPSEEK_CARRIER_CHOICES:
        raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
    for message in rendered:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("DeepSeek assistant history content must be a string")
        match = _CANONICAL_ASSISTANT_HISTORY_RE.fullmatch(content)
        if match:
            call_json = match.group("call_json").strip()
            if carrier == DEEPSEEK_CARRIER_TOOL_CALL:
                message["content"] = f"<tool_call>{call_json}</tool_call>"
            else:
                message["content"] = call_json
            continue
        stripped = content.strip()
        if carrier == DEEPSEEK_CARRIER_JSON_OUTPUT and _is_exact_json_action(stripped):
            message["content"] = stripped
            continue
        if (
            carrier == DEEPSEEK_CARRIER_TOOL_CALL
            and stripped.startswith("<tool_call>")
            and stripped.endswith("</tool_call>")
            and _is_exact_json_action(
                stripped[len("<tool_call>"):-len("</tool_call>")].strip()
            )
        ):
            message["content"] = stripped
            continue
        raise ValueError(
            "DeepSeek assistant history must match the selected provider carrier"
        )
    return rendered


def provider_instruction(
    model: str,
    *,
    example_visible_content: str | None = None,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return an explicit transport instruction only for a provider with a known split response."""
    if is_deepseek_split_model(model):
        if carrier not in DEEPSEEK_CARRIER_CHOICES:
            raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
        example = example_visible_content or (
            '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        )
        if carrier == DEEPSEEK_CARRIER_TOOL_CALL:
            return (
                "\n\nDEEPSEEK SPLIT-RESPONSE TOOL-CALL CONTRACT\n"
                "Use the API's separate native reasoning channel for one non-empty brief reason. "
                "Keep that reason under 120 words and reserve output budget for the final action; "
                "reasoning is incomplete until the action is emitted. If uncertain, emit the next "
                "legal inspection or data action instead of continuing analysis. "
                "After reasoning, your ENTIRE final response must be exactly one complete "
                "<tool_call>{...}</tool_call> block. Its first character is < and its last "
                "character is >.\n"
                "Copy this final-response shape, replacing only the JSON values:\n"
                f"<tool_call>{example}</tool_call>\n"
                "Put every tool parameter inside arguments; never add a tool parameter as an "
                "extra top-level key. "
                "Do not describe the response envelope, name its channels, repeat the reason, use "
                "Markdown or think tags, add a second action, or put any text before or after that "
                "single block. The client preserves the separate reason and wraps the unchanged "
                "action in the internal canonical envelope."
            )
        return (
            "\n\nDEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT\n"
            "Use the API's separate native reasoning channel for one non-empty brief reason. "
            "Keep that reason under 120 words and reserve output budget for the final action; "
            "reasoning is incomplete until the action is emitted. If uncertain, emit the next "
            "legal inspection or data action instead of continuing analysis. "
            "After reasoning, your ENTIRE final response must be exactly one JSON object with only "
            "the keys \"tool\" and \"arguments\". JSON Output is enabled.\n"
            "Copy this final-response shape, replacing only the JSON values:\n"
            f"{example}\n"
            "Put every tool parameter inside arguments; never add a tool parameter as an extra "
            "top-level key. "
            "Do not describe the response envelope, name its channels, repeat the reason, use "
            "Markdown or XML tags, add a second action, or put any text before or after the JSON. "
            "The client preserves the separate reason and raw JSON for audit, wraps the unchanged "
            "JSON in the internal canonical tool_call envelope, and rejects a missing reason or "
            "extra top-level keys."
        )
    return ""


def _is_exact_json_action(content: str) -> bool:
    try:
        action = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(action, dict)
        and set(action) == {"tool", "arguments"}
        and isinstance(action.get("tool"), str)
        and bool(action["tool"].strip())
        and isinstance(action.get("arguments"), dict)
    )


def adapt_provider_response(
    model: str,
    content: str,
    reasoning_content: str,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> tuple[str, dict[str, Any]]:
    """Return canonical assistant text plus an auditable adapter record.

    Only DeepSeek Flash's known split carrier is supported: a non-empty native reasoning field and
    visible content containing an unmodified tool-call block with no think tags. The adapter refuses
    any ambiguous/malformed form so ``parse_assistant_strict`` remains the final protocol gate.
    """
    raw_content = content or ""
    raw_reasoning = reasoning_content or ""
    record: dict[str, Any] = {
        "name": "none",
        "applied": False,
        "raw_content_present": bool(raw_content.strip()),
        "provider_reasoning_present": bool(raw_reasoning.strip()),
    }
    if not is_deepseek_split_model(model):
        return raw_content, record
    if carrier not in DEEPSEEK_CARRIER_CHOICES:
        raise ValueError(f"unknown DeepSeek carrier {carrier!r}")

    content_text = raw_content.strip()
    reasoning_text = raw_reasoning.strip()
    reasoning_has_tag = "<think" in reasoning_text.lower() or "</think>" in reasoning_text.lower()
    record["carrier"] = carrier
    record["name"] = (
        "deepseek_reasoning_tool_call_v1"
        if carrier == DEEPSEEK_CARRIER_TOOL_CALL
        else "deepseek_reasoning_json_content_v2"
    )
    if not reasoning_text:
        record["rejection_reason"] = "missing_reasoning_content"
    elif not content_text:
        record["rejection_reason"] = "empty_visible_content"
    elif reasoning_has_tag:
        record["rejection_reason"] = "think_tag_in_reasoning_content"
    elif carrier == DEEPSEEK_CARRIER_TOOL_CALL:
        has_think_tag = "<think" in content_text.lower() or "</think>" in content_text.lower()
        if has_think_tag:
            record["rejection_reason"] = "think_tag_in_visible_content"
        elif not content_text.startswith("<tool_call>"):
            record["rejection_reason"] = "visible_prefix_before_tool_call"
        elif not content_text.endswith("</tool_call>"):
            record["rejection_reason"] = "incomplete_or_suffixed_visible_tool_call"
        elif not _is_exact_json_action(
            content_text[len("<tool_call>"):-len("</tool_call>")].strip()
        ):
            record["rejection_reason"] = "visible_tool_call_invalid_json"
        else:
            record["rejection_reason"] = None
    elif not _is_exact_json_action(content_text):
        try:
            parsed_content = json.loads(content_text)
        except json.JSONDecodeError:
            record["rejection_reason"] = "visible_content_not_json"
        else:
            record["rejection_reason"] = "visible_json_wrong_shape"
    else:
        record["rejection_reason"] = None
    exact_split_shape = record["rejection_reason"] is None
    record["eligible"] = exact_split_shape
    if not exact_split_shape:
        return raw_content, record

    record["applied"] = True
    if carrier == DEEPSEEK_CARRIER_TOOL_CALL:
        return f"<think>{reasoning_text}</think>\n{content_text}", record
    return f"<think>{reasoning_text}</think>\n<tool_call>{content_text}</tool_call>", record


def provider_rejection_message(audit: dict[str, Any]) -> str | None:
    """Describe a rejected provider carrier without repairing its content."""
    reason = audit.get("rejection_reason")
    if not reason:
        return None
    carrier = audit.get("carrier", DEEPSEEK_CARRIER_JSON_OUTPUT)
    details = {
        "missing_reasoning_content": "native reasoning_content was empty; put the brief action reason there",
        "empty_visible_content": (
            "visible content was empty; it must contain one complete JSON action object"
        ),
        "think_tag_in_reasoning_content": (
            "native reasoning_content contained a think tag; use plain reasoning text there"
        ),
        "visible_content_not_json": (
            "visible content was not exactly one valid JSON action object; put all prose only in "
            "native reasoning_content"
        ),
        "visible_json_wrong_shape": (
            'visible JSON must contain exactly the top-level keys "tool" and "arguments"'
        ),
        "think_tag_in_visible_content": (
            "visible content contained a think tag; put reasoning only in native reasoning_content"
        ),
        "visible_prefix_before_tool_call": (
            "visible content had text before the required tool_call block"
        ),
        "incomplete_or_suffixed_visible_tool_call": (
            "visible content did not end with one complete tool_call block"
        ),
        "visible_tool_call_invalid_json": (
            "the tool_call body was not one valid JSON action object"
        ),
    }
    detail = details.get(reason, f"provider carrier was rejected: {reason}")
    if carrier == DEEPSEEK_CARRIER_TOOL_CALL:
        return (
            "DeepSeek split-response transport error: " + detail + ". On the retry, visible "
            "content must be exactly <tool_call>{\"tool\":\"...\",\"arguments\":{...}}</tool_call> "
            "with nothing before or after it; do not put reasoning prose, Markdown, or think tags "
            "in visible content."
        )
    return (
        "DeepSeek split-response transport error: " + detail + ". On the retry, visible content "
        "must be exactly {\"tool\":\"...\",\"arguments\":{...}} with nothing before or after it; "
        "do not put reasoning prose, Markdown, or XML tags in visible content."
    )
