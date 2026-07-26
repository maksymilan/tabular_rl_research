#!/usr/bin/env python3
"""Public registry for the two independently selectable table-tool schemes.

The schemes share the harness-owned atomic relational semantics. They do not share a model action
carrier, parser, prompt, trajectory identity, or training manifest. This module is deliberately a
small adapter boundary so callers never infer a scheme from record shape or experimental flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TOOL_SCHEME_REGISTRY_VERSION = "tool-scheme-registry-v1"
ATOMIC_TOOL_SCHEME = "atomic"
ACTION_BLOCK_TOOL_SCHEME = "action-block"
TOOL_SCHEME_NAMES = (
    ATOMIC_TOOL_SCHEME,
    ACTION_BLOCK_TOOL_SCHEME,
)

ATOMIC_ASSISTANT_CARRIER = "inline-think-tagged-tool-call"


@dataclass(frozen=True)
class ToolScheme:
    """One complete model-visible action protocol."""

    name: str
    protocol_version: str
    protocol_hash: str
    system_prompt: str
    assistant_carrier: str
    top_level_tools: tuple[str, ...]
    atomic_tools: tuple[str, ...]
    max_batch_calls: int | None = None

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
            "tool_scheme": self.name,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "assistant_carrier": self.assistant_carrier,
            "top_level_tools": list(self.top_level_tools),
            "atomic_tools": list(self.atomic_tools),
            "max_batch_calls": self.max_batch_calls,
        }


def require_tool_scheme(name: str) -> str:
    if name not in TOOL_SCHEME_NAMES:
        raise ValueError(
            f"unknown tool scheme {name!r}; expected one of {TOOL_SCHEME_NAMES}"
        )
    return name


def build_atomic_tool_scheme(
    *,
    system_prompt: str | None = None,
) -> ToolScheme:
    """Build the original one-model-turn/one-tool scheme."""
    from protocol import (  # imported lazily to keep the registry cycle-free
        PROTOCOL_VERSION,
        SYSTEM_PROMPT,
        TOOLS,
        protocol_hash,
    )

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    return ToolScheme(
        name=ATOMIC_TOOL_SCHEME,
        protocol_version=PROTOCOL_VERSION,
        protocol_hash=protocol_hash(prompt),
        system_prompt=prompt,
        assistant_carrier=ATOMIC_ASSISTANT_CARRIER,
        top_level_tools=tuple(sorted(TOOLS)),
        atomic_tools=tuple(sorted(TOOLS - {"plan", "answer_from_context"})),
    )


def build_action_block_tool_scheme(
    *,
    max_batch_calls: int = 8,
    assistant_carrier: str | None = None,
    protocol_version: str | None = None,
) -> ToolScheme:
    """Build the action-block scheme without modifying the atomic scheme."""
    from batch_plan_protocol import (
        BATCH_CARRIER_INLINE_THINK,
        EXECUTABLE_TOOLS,
        SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION,
        batch_plan_protocol_hash,
        build_batch_plan_system_prompt,
    )

    carrier = assistant_carrier or BATCH_CARRIER_INLINE_THINK
    version = (
        protocol_version or SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION
    )
    prompt = build_batch_plan_system_prompt(
        max_batch_calls,
        assistant_carrier=carrier,
    )
    return ToolScheme(
        name=ACTION_BLOCK_TOOL_SCHEME,
        protocol_version=version,
        protocol_hash=batch_plan_protocol_hash(
            prompt,
            max_batch_calls,
            protocol_version=version,
        ),
        system_prompt=prompt,
        assistant_carrier=carrier,
        top_level_tools=("action_block", "answer_from_context"),
        atomic_tools=tuple(EXECUTABLE_TOOLS),
        max_batch_calls=max_batch_calls,
    )


def build_tool_scheme(
    name: str,
    *,
    system_prompt: str | None = None,
    max_batch_calls: int = 8,
    assistant_carrier: str | None = None,
    protocol_version: str | None = None,
) -> ToolScheme:
    require_tool_scheme(name)
    if name == ATOMIC_TOOL_SCHEME:
        if assistant_carrier is not None or protocol_version is not None:
            raise ValueError(
                "atomic scheme carrier/version are owned by the atomic protocol"
            )
        return build_atomic_tool_scheme(system_prompt=system_prompt)
    if system_prompt is not None:
        raise ValueError(
            "action-block system prompt is derived from its carrier and batch bound"
        )
    return build_action_block_tool_scheme(
        max_batch_calls=max_batch_calls,
        assistant_carrier=assistant_carrier,
        protocol_version=protocol_version,
    )


def render_scheme_action(
    scheme: ToolScheme,
    reasoning: str,
    tool: str,
    arguments: dict,
) -> str:
    """Render a model target in the selected scheme's native training carrier."""
    if scheme.name == ATOMIC_TOOL_SCHEME:
        from protocol import assistant_message

        return assistant_message(reasoning, tool, arguments)
    if scheme.name == ACTION_BLOCK_TOOL_SCHEME:
        from batch_plan_protocol import render_batch_plan_assistant

        return render_batch_plan_assistant(reasoning, tool, arguments)
    raise ValueError(f"unsupported tool scheme: {scheme.name}")


def parse_scheme_action(
    scheme: ToolScheme,
    text: str,
) -> tuple[str, str, dict]:
    """Strictly parse one local/student-model turn for the selected scheme."""
    if scheme.name == ATOMIC_TOOL_SCHEME:
        from protocol import parse_assistant_strict

        return parse_assistant_strict(text)
    if scheme.name == ACTION_BLOCK_TOOL_SCHEME:
        from batch_plan_protocol import parse_batch_plan_assistant

        return parse_batch_plan_assistant(
            text,
            max_batch_calls=int(scheme.max_batch_calls or 0),
        )
    raise ValueError(f"unsupported tool scheme: {scheme.name}")


def assert_record_tool_scheme(
    record: dict,
    expected: str,
    *,
    allow_legacy_atomic: bool = False,
) -> None:
    """Reject cross-scheme dataset mixing before replay, export, or training."""
    require_tool_scheme(expected)
    actual = record.get("tool_scheme")
    if actual is None and allow_legacy_atomic and expected == ATOMIC_TOOL_SCHEME:
        return
    if actual != expected:
        raise ValueError(
            f"record tool_scheme {actual!r} does not match expected {expected!r}"
        )
