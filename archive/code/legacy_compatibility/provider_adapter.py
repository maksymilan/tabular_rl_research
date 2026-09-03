"""Provider-specific transport adapters before the shared strict protocol parser.

Adapters map an API provider's *separate response fields* into the active assistant envelope.
They are deliberately narrower than parser repair: raw content is preserved, tool JSON is never
changed, and malformed XML/JSON remains a protocol error after adaptation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_modules.native_tool_bundle.provider_tools import (
    MAX_NATIVE_BUNDLE_CALLS,
    NATIVE_ASSISTANT_CARRIER,
    NATIVE_BUNDLE_ASSISTANT_CARRIER,
    canonical_messages_to_native,
    native_atomic_tools,
    native_tools_sha256,
)
from prompt_contract import CANONICAL_ACTION_RULE, TEACHER_ONE_ACTION_RULE


DEEPSEEK_V4_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
DEEPSEEK_V4_DEFAULT_MAX_TOKENS = 2048
DEEPSEEK_V4_REASONING_EFFORT = "high"
DEEPSEEK_CARRIER_JSON_OUTPUT = "json-output"
DEEPSEEK_CARRIER_TOOL_CALL = "tool-call"
DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS = "native-tool-calls"
DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE = "native-tool-bundle"
DEEPSEEK_CARRIER_CHOICES = (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    DEEPSEEK_CARRIER_TOOL_CALL,
    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
)
VERSION40_RESPONSE_CONTRACT_PLACEHOLDER = "{{VERSION40_RESPONSE_CONTRACT}}"
_CANONICAL_SYSTEM_RESPONSE_RULE = CANONICAL_ACTION_RULE
_SPLIT_SYSTEM_RESPONSE_RULE = (
    "1. Produce exactly one tool action per turn using the provider-specific response envelope "
    "at the end of this prompt."
)
_CANONICAL_COMPACT_RESPONSE_RULE = (
    "Each turn output exactly:\n"
    "<think>brief reason for this action</think>\n"
    '{"tool":"...","arguments":{...}}\n'
    "No other text and no tool_call tags."
)
_SPLIT_COMPACT_RESPONSE_RULE = (
    "Each turn produces exactly one tool action using the provider-specific response envelope at "
    "the end of this prompt."
)
_CANONICAL_ROLLING_COMPACT_RESPONSE_RULE = (
    "Output only <think>specific reason for the next action</think> followed by one complete "
    '{"tool":"name","arguments":{...}} JSON object. No other prose, no tool_call tags, no '
    "second action, no shorthand JSON, and no legacy tool fields."
)
_SPLIT_ROLLING_COMPACT_RESPONSE_RULE = (
    "Produce exactly one tool action using the provider-specific response envelope at the end of "
    "this prompt. Do not emit a second action, shorthand JSON, or legacy tool fields."
)
_CANONICAL_GENERATION_RESPONSE_RULE = TEACHER_ONE_ACTION_RULE
_SPLIT_GENERATION_RESPONSE_RULE = (
    "ONE REQUEST = ONE ACTION. Produce one non-empty brief action reason in the provider's native "
    "reasoning field and exactly one action in the provider's visible field, using the "
    "provider-specific response envelope at the end of this prompt. Immediately STOP after that "
    "single action: never emit a second reason, a second action, a numbered "
    "plan of calls, or a complete multi-step solution in one response. The harness will execute "
    "only this one action and return a fresh state before you choose the next action. Follow the "
    "provider-specific field placement at the end of this prompt."
)
_NATIVE_SYSTEM_RESPONSE_RULE = (
    "1. Each turn, select exactly one function through the provider's native function-calling "
    "interface. Put non-empty reasoning only in the native reasoning field and leave assistant "
    "content empty."
)
_NATIVE_GENERATION_RESPONSE_RULE = (
    "ONE REQUEST = ONE NATIVE FUNCTION CALL. Produce one non-empty, brief action reason in the "
    "provider's native reasoning field, select exactly one supplied function, and pass only that "
    "function's arguments object. Immediately stop after the call. Never emit a second function, "
    "serialize a tool/action wrapper into assistant content, or provide a complete multi-step "
    "solution. The harness executes this one call and returns fresh state before the next choice."
)
_NATIVE_BUNDLE_SYSTEM_RESPONSE_RULE = (
    "1. Each turn, use the provider's native function interface. Select one or more independent "
    "functions that are all valid from the currently visible state."
)
_NATIVE_BUNDLE_GENERATION_RESPONSE_RULE = (
    "ONE REQUEST = ONE NATIVE TOOL BUNDLE. Produce one non-empty, brief reason in the provider's "
    "native reasoning field, then select one or more supplied functions. Calls in the same "
    "response share one pre-call state: never make a later call depend on a result created by an "
    "earlier call in that response. Call answer_from_context only by itself."
)
_CANONICAL_ONE_ACTION_POLICY_RULE = (
    "2. Use only names and arguments in TOOLS. One turn contains one action."
)
_NATIVE_BUNDLE_ACTION_POLICY_RULE = (
    "2. Use only names and arguments in TOOLS. One turn contains one or more independent "
    "native calls, all chosen from the same visible pre-call state."
)
_CANONICAL_ASSISTANT_HISTORY_RE = re.compile(
    r"^\s*<think>(?P<reasoning>.*?)</think>\s*(?P<call_json>\{.*\})\s*$",
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
    native_model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
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
    elif carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        # Thinking mode currently rejects tool_choice="required". Version50 narrows auto to one
        # call client-side; version51 accepts a bounded provider-native bundle.
        options["tools"] = native_atomic_tools(native_model_arg_schema)
        options["tool_choice"] = "auto"
    return options


def provider_request_audit_options(
    model: str,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
    native_model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
) -> dict[str, Any]:
    """Return request controls without duplicating the full native schema in every turn."""
    options = provider_request_options(
        model,
        carrier=carrier,
        native_model_arg_schema=native_model_arg_schema,
    )
    if carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    } and "tools" in options:
        options = dict(options)
        options.pop("tools")
        options["native_tools_sha256"] = native_tools_sha256(
            native_model_arg_schema
        )
        options["native_tool_count"] = len(
            native_atomic_tools(native_model_arg_schema)
        )
        options["provider_assistant_carrier"] = (
            NATIVE_BUNDLE_ASSISTANT_CARRIER
            if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE
            else NATIVE_ASSISTANT_CARRIER
        )
        if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
            options["max_native_bundle_calls"] = MAX_NATIVE_BUNDLE_CALLS
    return options


def _rewrite_canonical_cookbook_for_native(prompt: str) -> str:
    """Keep the teacher's argument examples without teaching a competing JSON wrapper."""
    header = "CANONICAL CALLS (copy these argument shapes; replace names and values only)"
    suffix = "\n\nDATA GENERATION STRICTNESS\n"
    start = prompt.find(header)
    if start < 0:
        raise ValueError("native function-call prompt cannot find the canonical call cookbook")
    end = prompt.find(suffix, start)
    if end < 0:
        raise ValueError("native function-call prompt cannot find data-generation strictness")
    lines = prompt[start:end].splitlines()
    rewritten = [
        "NATIVE FUNCTION CALL EXAMPLES (select one function; replace argument values only)"
    ]
    for line in lines[1:]:
        if not line.strip():
            continue
        label, separator, raw_action = line.partition(": ")
        if not separator:
            raise ValueError(f"cannot convert canonical call example: {line!r}")
        try:
            action = json.loads(raw_action)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cannot parse canonical call example: {label}") from exc
        if (
            not isinstance(action, dict)
            or set(action) != {"tool", "arguments"}
            or not isinstance(action.get("tool"), str)
            or not isinstance(action.get("arguments"), dict)
        ):
            raise ValueError(f"canonical call example has unexpected shape: {label}")
        arguments = json.dumps(
            action["arguments"], ensure_ascii=False, separators=(",", ":")
        )
        rewritten.append(
            f"{label}: call function {action['tool']} with arguments {arguments}"
        )
    return prompt[:start] + "\n".join(rewritten) + prompt[end:]


def provider_system_prompt(
    model: str,
    canonical_prompt: str,
    *,
    example_visible_content: str | None = None,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return one unambiguous API-facing response contract for the selected provider.

    The internal trajectory protocol remains canonical ``<think>`` + raw JSON. A provider
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
    if carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        split_response = _NATIVE_SYSTEM_RESPONSE_RULE
        split_generation_rule = _NATIVE_GENERATION_RESPONSE_RULE
        if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
            split_response = _NATIVE_BUNDLE_SYSTEM_RESPONSE_RULE
            split_generation_rule = _NATIVE_BUNDLE_GENERATION_RESPONSE_RULE
    else:
        split_generation_rule = _SPLIT_GENERATION_RESPONSE_RULE
    prompt = canonical_prompt.replace(
        canonical_response,
        split_response,
        1,
    ).replace(
        _CANONICAL_GENERATION_RESPONSE_RULE,
        split_generation_rule,
        1,
    )
    if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
        if prompt.count(_CANONICAL_ONE_ACTION_POLICY_RULE) != 1:
            raise ValueError(
                "native tool-bundle prompt cannot replace the one-action policy rule"
            )
        prompt = prompt.replace(
            _CANONICAL_ONE_ACTION_POLICY_RULE,
            _NATIVE_BUNDLE_ACTION_POLICY_RULE,
            1,
        )
    if carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        prompt = _rewrite_canonical_cookbook_for_native(prompt)
    return prompt + provider_instruction(
        model,
        example_visible_content=example_visible_content,
        carrier=carrier,
    )


def provider_system_prompt_version40(
    model: str,
    prompt_template: str,
    *,
    example_visible_content: str | None = None,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Fill version40's single response-contract slot without duplicating prompt rules."""
    if prompt_template.count(VERSION40_RESPONSE_CONTRACT_PLACEHOLDER) != 1:
        raise ValueError("version40 prompt must contain exactly one response-contract placeholder")
    example = example_visible_content or (
        '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
    )
    if not is_deepseek_split_model(model):
        contract = (
            "Use one non-empty <think> block for the reasoning required above, followed directly "
            'by one raw JSON object with exactly the keys "tool" and "arguments". Emit nothing '
            "else."
        )
    elif carrier == DEEPSEEK_CARRIER_JSON_OUTPUT:
        contract = (
            "Use the API's native reasoning field for the reasoning required above. The entire "
            'visible response must be one complete JSON object with exactly the keys "tool" and '
            '"arguments"; JSON Output is enabled. Put every parameter inside arguments. Emit no '
            "Markdown, XML, explanation, or second action.\n"
            "Shape example (replace values only):\n"
            f"{example}"
        )
    elif carrier == DEEPSEEK_CARRIER_TOOL_CALL:
        contract = (
            "Use the API's native reasoning field for the reasoning required above. The entire "
            "visible response must be one complete <tool_call>{...}</tool_call> block containing "
            'a JSON object with exactly the keys "tool" and "arguments". Put every parameter '
            "inside arguments. Emit no Markdown, explanation, or second action.\n"
            "Shape example (replace values only):\n"
            f"<tool_call>{example}</tool_call>"
        )
    else:
        raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
    return prompt_template.replace(
        VERSION40_RESPONSE_CONTRACT_PLACEHOLDER,
        contract,
        1,
    )


def provider_request_messages(
    model: str,
    messages: list[dict],
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
    *,
    preserve_reasoning: bool = False,
    native_assistant_history: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[dict]:
    """Render canonical legal-history actions in the provider's API-facing carrier.

    By default DeepSeek receives prior assistant actions as one raw JSON action object and old
    reasoning is omitted, preserving the promoted contract. The isolated version40/version41
    diagnostics may instead copy the complete prior canonical reason into the provider-native
    ``reasoning_content`` field while keeping the visible action carrier unchanged.
    The returned list is a copy; canonical audit/history records are not mutated.
    """
    rendered = [dict(message) for message in messages]
    if not is_deepseek_split_model(model):
        return rendered
    if carrier not in DEEPSEEK_CARRIER_CHOICES:
        raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
    if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS:
        return canonical_messages_to_native(rendered, native_assistant_history)
    if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
        return rendered
    for message in rendered:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("DeepSeek assistant history content must be a string")
        match = _CANONICAL_ASSISTANT_HISTORY_RE.fullmatch(content)
        if match:
            if preserve_reasoning:
                reasoning = match.group("reasoning").strip()
                if not reasoning:
                    raise ValueError("DeepSeek assistant history reasoning must be non-empty")
                message["reasoning_content"] = reasoning
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
    include_client_implementation: bool = True,
) -> str:
    """Return an explicit transport instruction only for a provider with a known split response."""
    if is_deepseek_split_model(model):
        if carrier not in DEEPSEEK_CARRIER_CHOICES:
            raise ValueError(f"unknown DeepSeek carrier {carrier!r}")
        example = example_visible_content or (
            '{"tool":"describe_table","arguments":{"tables":["Document"]}}'
        )
        if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
            return (
                "\n\nDEEPSEEK NATIVE TOOL-BUNDLE CONTRACT\n"
                "Use the API's separate native reasoning field for one non-empty brief reason. "
                f"Select between one and {MAX_NATIVE_BUNDLE_CALLS} supplied functions. Every "
                "call in one response is chosen from the same currently visible state: calls may "
                "be independent, but no call may consume a handle or value produced by another "
                "call in that response. Call answer_from_context only as the sole call. Assistant "
                "content is not executable and is ignored for scoring; put arguments only in "
                "native function calls. The harness returns one tool result for every call id."
            )
        if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS:
            return (
                "\n\nDEEPSEEK NATIVE FUNCTION-CALL CONTRACT\n"
                "Use the API's separate native reasoning field for one non-empty brief reason. "
                "Keep it under 120 words and reserve output budget for the call. Select exactly "
                "one function from the supplied tools and pass only its arguments object. Leave "
                "assistant content empty: do not serialize a {\"tool\":...,\"arguments\":...} "
                "wrapper, XML/tool_call tags, Markdown, an answer, or a second call into content. "
                "The client rejects zero or multiple calls, unknown functions, malformed argument "
                "JSON, missing reasoning, and any non-empty assistant content before execution."
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
        client_clause = (
            " The client preserves the separate reason and raw JSON for audit, renders the "
            "active think-plus-JSON envelope, and rejects a missing reason or extra top-level "
            "keys."
            if include_client_implementation
            else ""
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
            "Markdown or XML tags, add a second action, or put any text before or after the JSON."
            + client_clause
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
    if carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        record["name"] = NATIVE_ASSISTANT_CARRIER
        if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
            record["name"] = NATIVE_BUNDLE_ASSISTANT_CARRIER
        record["transport_reconstructed_from_native_tool_calls"] = True
    elif carrier == DEEPSEEK_CARRIER_TOOL_CALL:
        record["name"] = "deepseek_reasoning_tool_call_v1"
    else:
        record["name"] = "deepseek_reasoning_json_content_v2"
    if carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
        record["assistant_content_ignored"] = True
        record["reasoning_contains_think_tag"] = reasoning_has_tag
        try:
            payload = json.loads(content_text)
        except json.JSONDecodeError:
            payload = None
        calls = payload.get("calls") if isinstance(payload, dict) else None
        if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_NATIVE_BUNDLE_CALLS:
            record["rejection_reason"] = "invalid_native_bundle"
        else:
            record["rejection_reason"] = None
            record["native_call_count"] = len(calls)
        record["eligible"] = record["rejection_reason"] is None
        record["applied"] = record["eligible"]
        return raw_content, record
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
        content_text = content_text[len("<tool_call>"):-len("</tool_call>")].strip()
    return f"<think>{reasoning_text}</think>\n{content_text}", record


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
    if reason.startswith("tool_call_count_"):
        count = reason.removeprefix("tool_call_count_")
        detail = (
            f"the assistant selected {count} native functions in one atomic turn; "
            "exactly one is required"
        )
    else:
        native_details = {
            "nonempty_assistant_content": (
                "assistant content was non-empty; use only reasoning_content plus one native "
                "function call"
            ),
            "missing_function_payload": "the native tool call had no function payload",
            "unknown_function": "the native tool call named a function outside the supplied set",
            "invalid_arguments_json": "the native function arguments were not valid JSON",
            "arguments_not_object_json": (
                "the native function arguments did not decode to one JSON object"
            ),
        }
        detail = details.get(
            reason,
            native_details.get(reason, f"provider carrier was rejected: {reason}"),
        )
    if carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        return (
            "DeepSeek native function-call transport error: " + detail + ". On retry, select "
            "exactly one supplied function, put only its parameters in the function arguments, "
            "keep assistant content empty, and put the brief reason only in reasoning_content."
        )
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
