"""Provider-specific transport adapters before the shared strict protocol parser.

Adapters map an API provider's *separate response fields* into the canonical assistant envelope.
They are deliberately narrower than parser repair: raw content is preserved, tool JSON is never
changed, and malformed XML/JSON remains a protocol error after adaptation.
"""
from __future__ import annotations

from typing import Any


DEEPSEEK_V4_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
DEEPSEEK_V4_DEFAULT_MAX_TOKENS = 2048

def is_deepseek_split_model(model: str) -> bool:
    return str(model).strip().lower() in DEEPSEEK_V4_MODELS


def provider_default_max_tokens(model: str, fallback: int) -> int:
    """Use a larger explicit budget for the known split-response provider only."""
    return DEEPSEEK_V4_DEFAULT_MAX_TOKENS if is_deepseek_split_model(model) else fallback


def provider_instruction(model: str) -> str:
    """Return an explicit transport instruction only for a provider with a known split response."""
    if is_deepseek_split_model(model):
        return (
            "\n\nDEEPSEEK SPLIT-RESPONSE OUTPUT CONTRACT (OVERRIDES ALL EARLIER FORMAT WORDING)\n"
            "This API transports your response in two fields. You MUST return exactly this shape "
            "on every turn:\n"
            "- native reasoning_content: one non-empty, brief reason for the next action; no XML tags.\n"
            "- visible content: exactly one complete <tool_call>{...}</tool_call> block and NOTHING "
            "before or after it.\n"
            "Concrete example:\n"
            "reasoning_content = Inspect the Document schema before filtering.\n"
            "content = <tool_call>{\"tool\":\"describe_table\",\"arguments\":{\"tables\":[\"Document\"]}}</tool_call>\n"
            "Never put reasoning prose, <think>, </think>, Markdown, a second tool call, or an "
            "unfinished JSON/tool-call in visible content. The client carries your existing native "
            "reasoning_content into the canonical <think> field; it will reject a missing reasoning "
            "field or any extra visible text."
        )
    return ""


def adapt_provider_response(model: str, content: str, reasoning_content: str) -> tuple[str, dict[str, Any]]:
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

    content_text = raw_content.strip()
    reasoning_text = raw_reasoning.strip()
    has_any_think_tag = "<think" in content_text.lower() or "</think>" in content_text.lower()
    reasoning_has_tag = "<think" in reasoning_text.lower() or "</think>" in reasoning_text.lower()
    record["name"] = "deepseek_flash_reasoning_content_v1"
    if not reasoning_text:
        record["rejection_reason"] = "missing_reasoning_content"
    elif not content_text:
        record["rejection_reason"] = "empty_visible_content"
    elif reasoning_has_tag:
        record["rejection_reason"] = "think_tag_in_reasoning_content"
    elif has_any_think_tag:
        record["rejection_reason"] = "think_tag_in_visible_content"
    elif not content_text.startswith("<tool_call>"):
        record["rejection_reason"] = "visible_prefix_before_tool_call"
    elif not content_text.endswith("</tool_call>"):
        record["rejection_reason"] = "incomplete_or_suffixed_visible_tool_call"
    else:
        record["rejection_reason"] = None
    exact_split_shape = record["rejection_reason"] is None
    record["eligible"] = exact_split_shape
    if not exact_split_shape:
        return raw_content, record

    record["applied"] = True
    return f"<think>{reasoning_text}</think>\n{content_text}", record
