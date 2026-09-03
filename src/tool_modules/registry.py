#!/usr/bin/env python3
"""The single active tool-scheme registry for Atomic version26.

Historical tool schemes are stored under ``archive/code/legacy_tool_modules`` and are
intentionally not importable from the active source tree. Keeping this registry narrow makes
cross-scheme mixing fail at import/configuration time instead of silently creating a second
training route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from action_carrier import ACTIVE_ACTION_CARRIER


TOOL_SCHEME_REGISTRY_VERSION = "tool-scheme-registry-v13-atomic-only"
ATOMIC_TOOL_SCHEME = "atomic"
FORWARD_TOOL_SCHEME = ATOMIC_TOOL_SCHEME
TOOL_SCHEME_NAMES = (ATOMIC_TOOL_SCHEME,)
ATOMIC_ASSISTANT_CARRIER = ACTIVE_ACTION_CARRIER


@dataclass(frozen=True)
class ToolScheme:
    """The complete model-visible Atomic version26 action protocol."""

    name: str
    protocol_version: str
    protocol_hash: str
    system_prompt: str
    assistant_carrier: str
    top_level_tools: tuple[str, ...]
    atomic_tools: tuple[str, ...]
    max_batch_calls: int | None = None
    mode: str | None = None
    tool_schema_hash: str | None = None
    student_prompt_hash: str | None = None
    teacher_prompt_hash: str | None = None
    admission_status: str | None = None
    provider_response_envelope_version: str | None = None
    atomic_operator_profile: str | None = None

    def manifest_fields(self) -> dict[str, Any]:
        payload = {
            "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
            "tool_scheme": self.name,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "assistant_carrier": self.assistant_carrier,
            "top_level_tools": list(self.top_level_tools),
            "atomic_tools": list(self.atomic_tools),
            "max_batch_calls": self.max_batch_calls,
        }
        optional = {
            "mode": self.mode,
            "tool_schema_sha256": self.tool_schema_hash,
            "student_prompt_sha256": self.student_prompt_hash,
            "teacher_prompt_sha256": self.teacher_prompt_hash,
            "admission_status": self.admission_status,
            "provider_response_envelope_version": self.provider_response_envelope_version,
            "atomic_operator_profile": self.atomic_operator_profile,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


def require_tool_scheme(name: str) -> str:
    if name != ATOMIC_TOOL_SCHEME:
        raise ValueError(
            f"unsupported tool scheme {name!r}; the active registry only permits {ATOMIC_TOOL_SCHEME!r}"
        )
    return name


def build_atomic_tool_scheme(*, system_prompt: str | None = None) -> ToolScheme:
    """Build the frozen Atomic version26 one-tool-per-turn scheme."""
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


def build_tool_scheme(
    name: str,
    *,
    system_prompt: str | None = None,
    max_batch_calls: int | None = None,
    assistant_carrier: str | None = None,
    protocol_version: str | None = None,
    mode: str | None = None,
) -> ToolScheme:
    require_tool_scheme(name)
    if max_batch_calls is not None or assistant_carrier is not None or protocol_version is not None:
        raise ValueError("Atomic owns its carrier, protocol version, and single-action boundary")
    if mode is not None:
        raise ValueError("Atomic does not accept a mode selector")
    return build_atomic_tool_scheme(system_prompt=system_prompt)


def render_scheme_action(
    scheme: ToolScheme,
    reasoning: str,
    tool: str,
    arguments: dict,
) -> str:
    require_tool_scheme(scheme.name)
    from protocol import assistant_message

    return assistant_message(reasoning, tool, arguments)


def parse_scheme_action(scheme: ToolScheme, text: str) -> tuple[str, str, dict]:
    require_tool_scheme(scheme.name)
    from protocol import parse_assistant_strict

    return parse_assistant_strict(text)


def assert_record_tool_scheme(
    record: dict,
    expected: str,
    *,
    allow_legacy_atomic: bool = False,
) -> None:
    """Reject cross-scheme records before replay, export, or training."""
    require_tool_scheme(expected)
    actual = record.get("tool_scheme")
    if actual is None and allow_legacy_atomic and expected == ATOMIC_TOOL_SCHEME:
        return
    if actual != expected:
        raise ValueError(
            f"record tool_scheme {actual!r} does not match expected {expected!r}"
        )
