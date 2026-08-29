#!/usr/bin/env python3
"""Qwen3 inline carrier for the frozen checkpoint-relalg Atomic-v24 profile.

External DeepSeek generation authors reasoning and visible JSON in separate
provider fields.  Qwen3 SFT/evaluation/RL instead uses one ordinary text
completion containing exactly one non-empty ``<think>`` block followed by one
raw JSON action.  This module is the only adapter between that local text
carrier and the shared checkpoint-relalg action/runtime contract.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from .protocol import (
    ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    CARRIER_TEXT_JSON,
    CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
    ProtocolValidationError,
    get_system_prompt,
    validate_model_action,
)


QWEN3_INLINE_CARRIER_VERSION = "checkpoint-relalg-qwen3-inline-think-json-v1"

_PROVIDER_CARRIER_CLAUSE = (
    "TEXT-JSON CARRIER (diagnostic-only)\n"
    "Keep reasoning only in the provider reasoning_content field. Visible assistant "
    "content must be exactly one raw JSON object with exactly the keys tool and "
    "arguments: {\"tool\":\"tool_name\",\"arguments\":{}}. Do not return prose, "
    "Markdown fences, XML, an array, or multiple actions. A later user message whose "
    "JSON type is checkpoint_relalg_tool_result is Harness feedback, not a new task."
)

_QWEN3_CARRIER_CLAUSE = (
    "QWEN3 INLINE ACTION CARRIER\n"
    "Each assistant turn must be exactly one non-empty <think>...</think> block "
    "followed directly by one raw JSON object with exactly the keys tool and arguments: "
    "{\"tool\":\"tool_name\",\"arguments\":{}}. Put reasoning only inside the think "
    "block. Do not return prose outside it, Markdown fences, XML tool-call tags, an "
    "array, or multiple actions. A later user message whose JSON type is "
    "checkpoint_relalg_tool_result is Harness feedback, not a new task. Prior "
    "assistant history contains its JSON action without the earlier think text."
)

_INLINE_ACTION = re.compile(
    r"\A\s*<think>(?P<reasoning>.*?)</think>\s*(?P<action>\{.*\})\s*\Z",
    flags=re.DOTALL,
)


class Qwen3ActionError(ProtocolValidationError):
    """One local Qwen3 completion violates the inline carrier or action schema."""


def provider_student_system_prompt() -> str:
    """Return the exact split-provider prompt used by source candidates."""

    return get_system_prompt(
        "atomic",
        teacher=False,
        carrier=CARRIER_TEXT_JSON,
        checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    )


def qwen3_student_system_prompt() -> str:
    """Return the same semantic contract with only the transport clause changed."""

    source = provider_student_system_prompt()
    if source.count(_PROVIDER_CARRIER_CLAUSE) != 1:
        raise RuntimeError("Atomic-v24 provider carrier clause drifted")
    result = source.replace(_PROVIDER_CARRIER_CLAUSE, _QWEN3_CARRIER_CLAUSE, 1)
    if "provider reasoning_content" in result:
        raise RuntimeError("Qwen3 system prompt still contains provider-only instructions")
    return result


def qwen3_student_prompt_sha256() -> str:
    return hashlib.sha256(qwen3_student_system_prompt().encode("utf-8")).hexdigest()


def render_qwen3_action(reasoning: str, content: str) -> str:
    """Join a verified split source envelope into one local Qwen3 completion."""

    if not isinstance(reasoning, str) or not reasoning.strip():
        raise Qwen3ActionError(
            "Qwen3 reasoning must be non-empty",
            code="invalid_qwen3_reasoning",
            path="$.reasoning",
        )
    if "<think>" in reasoning or "</think>" in reasoning:
        raise Qwen3ActionError(
            "Qwen3 reasoning must not contain nested think tags",
            code="invalid_qwen3_reasoning",
            path="$.reasoning",
        )
    if not isinstance(content, str) or not content.strip():
        raise Qwen3ActionError(
            "Qwen3 action content must be non-empty",
            code="invalid_qwen3_action_content",
            path="$.content",
        )
    return f"<think>\n{reasoning}\n</think>\n\n{content}"


def parse_qwen3_action(text: Any) -> dict[str, Any]:
    """Strictly lower one inline Qwen3 completion to the shared canonical action."""

    decoded = decode_qwen3_action(text)
    try:
        action = validate_model_action(
            decoded["raw_action"],
            mode="atomic",
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
        )
    except ProtocolValidationError as exc:
        raise Qwen3ActionError(exc.message, code=exc.code, path=exc.path) from exc
    return {
        **deepcopy(action),
        "reasoning": decoded["reasoning"],
        "action_text": decoded["action_text"],
        "assistant_text": decoded["assistant_text"],
    }


def decode_qwen3_action(text: Any) -> dict[str, Any]:
    """Decode the local carrier without validating tool arguments.

    Local evaluation needs this boundary so a syntactically valid authored
    action can reach ``CheckpointRelalgRuntime.apply`` and receive the same
    structured Harness validation feedback as the DeepSeek source rollout.
    ``parse_qwen3_action`` remains the strict public helper for callers that
    need carrier and action-schema validation in one step.
    """

    if not isinstance(text, str):
        raise Qwen3ActionError(
            "Qwen3 completion must be text",
            code="invalid_qwen3_carrier",
            path="$.",
        )
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        raise Qwen3ActionError(
            "Qwen3 completion requires exactly one think block",
            code="invalid_qwen3_carrier",
            path="$.",
        )
    match = _INLINE_ACTION.fullmatch(text)
    if match is None or not match.group("reasoning").strip():
        raise Qwen3ActionError(
            "Qwen3 completion must be one non-empty think block followed by one JSON object",
            code="invalid_qwen3_carrier",
            path="$.",
        )
    try:
        raw_action = json.loads(match.group("action"))
    except json.JSONDecodeError as exc:
        raise Qwen3ActionError(
            "Qwen3 action suffix is not one raw JSON object",
            code="invalid_qwen3_action_content",
            path="$.action",
        ) from exc
    if not isinstance(raw_action, dict) or set(raw_action) != {"tool", "arguments"}:
        raise Qwen3ActionError(
            "Qwen3 action must contain exactly tool and arguments",
            code="invalid_qwen3_action_content",
            path="$.action",
        )
    if not isinstance(raw_action.get("tool"), str) or not raw_action["tool"]:
        raise Qwen3ActionError(
            "Qwen3 action tool must be a non-empty string",
            code="invalid_qwen3_action_content",
            path="$.action.tool",
        )
    if not isinstance(raw_action.get("arguments"), dict):
        raise Qwen3ActionError(
            "Qwen3 action arguments must be an object",
            code="invalid_qwen3_action_content",
            path="$.action.arguments",
        )
    return {
        "raw_action": deepcopy(raw_action),
        "reasoning": match.group("reasoning").strip(),
        "action_text": match.group("action"),
        "assistant_text": text,
    }


def qwen3_history_assistant_text(text: Any) -> str:
    """Return the exact JSON action retained in later local-Qwen history.

    DeepSeek thinking history retains provider reasoning, but the Qwen3 SFT
    contract deliberately removes earlier think blocks.  Local evaluation and
    RL must call this helper before appending an authored assistant turn to a
    later model input.
    """

    return str(parse_qwen3_action(text)["action_text"])


__all__ = [
    "QWEN3_INLINE_CARRIER_VERSION",
    "Qwen3ActionError",
    "decode_qwen3_action",
    "parse_qwen3_action",
    "provider_student_system_prompt",
    "qwen3_history_assistant_text",
    "qwen3_student_prompt_sha256",
    "qwen3_student_system_prompt",
    "render_qwen3_action",
]
