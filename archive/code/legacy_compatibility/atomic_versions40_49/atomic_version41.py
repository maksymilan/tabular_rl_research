"""Atomic version41: version40 behavior with one focused prompt-only correction module."""
from __future__ import annotations

import hashlib
import json

from atomic_version40 import (
    HISTORY_TURNS,
    MODEL_ARG_SCHEMA,
    TOOLS,
    TOOL_SPECS,
    parse_assistant_strict as parse_assistant_strict_version40,
    tool_schema_hash,
)
from atomic_version41_prompt import PROMPT_TEMPLATE
from provider_adapter import (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    provider_system_prompt_version40,
)
from protocol import AdjacentActionGuard


PROTOCOL_VERSION = "version41"

STUDENT_SYSTEM_PROMPT = provider_system_prompt_version40(
    "canonical-non-split-model",
    PROMPT_TEMPLATE,
)


def provider_system_prompt(
    model: str,
    *,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return version41's provider-specific prompt without changing the carrier contract."""
    return provider_system_prompt_version40(
        model,
        PROMPT_TEMPLATE,
        carrier=carrier,
    )


def parse_assistant_strict(
    text: str,
    *,
    adjacent_guard: AdjacentActionGuard | None = None,
    step_id: str | None = None,
) -> tuple[str, str, dict]:
    """Use the unchanged version40 parser and public action schema."""
    return parse_assistant_strict_version40(
        text,
        adjacent_guard=adjacent_guard,
        step_id=step_id,
    )


def protocol_hash(system_prompt: str) -> str:
    """Hash the version41 prompt together with the unchanged public tool schema."""
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "tool_schema_sha256": tool_schema_hash(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
