#!/usr/bin/env python3
"""Public registry for the independently selectable table-tool schemes.

The schemes share the active think-plus-raw-JSON carrier. The relational schemes also share
harness-owned atomic semantics; direct-sql-search instead shares only the immutable database,
causal loop, and hidden scorer. Schemes do not share a model-visible tool schema, action validator,
prompt, trajectory identity, or training manifest. This module is deliberately a small adapter
boundary so callers never infer a scheme from record shape or experimental flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from action_carrier import ACTIVE_ACTION_CARRIER


TOOL_SCHEME_REGISTRY_VERSION = "tool-scheme-registry-v4"
ATOMIC_TOOL_SCHEME = "atomic"
ACTION_BLOCK_TOOL_SCHEME = "action-block"
RELATIONAL_PROGRAM_TOOL_SCHEME = "relational-program"
DIRECT_SQL_SEARCH_TOOL_SCHEME = "direct-sql-search"
TOOL_SCHEME_NAMES = (
    ATOMIC_TOOL_SCHEME,
    ACTION_BLOCK_TOOL_SCHEME,
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    DIRECT_SQL_SEARCH_TOOL_SCHEME,
)

ATOMIC_ASSISTANT_CARRIER = ACTIVE_ACTION_CARRIER


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
        MAX_ACTION_BLOCK_CALLS,
        SEQUENTIAL_EXECUTABLE_TOOLS,
        SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
        TERMINAL_TOOL,
        batch_plan_protocol_hash,
        build_batch_plan_system_prompt,
    )

    if not 1 <= max_batch_calls <= MAX_ACTION_BLOCK_CALLS:
        raise ValueError(
            f"active action-block max_batch_calls must be in 1..{MAX_ACTION_BLOCK_CALLS}"
        )
    carrier = assistant_carrier or BATCH_CARRIER_INLINE_THINK
    version = (
        protocol_version or SIMPLE_SCALAR_CELL_PROTOCOL_VERSION
    )
    prompt = build_batch_plan_system_prompt(
        max_batch_calls,
        assistant_carrier=carrier,
        protocol_version=version,
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
        top_level_tools=("action_block", TERMINAL_TOOL),
        atomic_tools=tuple(SEQUENTIAL_EXECUTABLE_TOOLS),
        max_batch_calls=max_batch_calls,
    )


def build_relational_program_tool_scheme(
    *,
    max_program_calls: int = 8,
    assistant_carrier: str | None = None,
    protocol_version: str | None = None,
) -> ToolScheme:
    """Build the separate declarative-program scheme."""
    from relational_program_protocol import (
        ACTION_CARRIER,
        MAX_RELATIONAL_PROGRAM_CALLS,
        RELATIONAL_PRIMITIVE_TOOLS,
        RELATIONAL_PROGRAM_PROTOCOL_VERSION,
        TOP_LEVEL_TOOLS,
        build_relational_program_system_prompt,
        relational_program_protocol_hash,
    )

    if not 1 <= max_program_calls <= MAX_RELATIONAL_PROGRAM_CALLS:
        raise ValueError(
            "active relational-program max_program_calls must be in "
            f"1..{MAX_RELATIONAL_PROGRAM_CALLS}"
        )
    carrier = assistant_carrier or ACTION_CARRIER
    version = protocol_version or RELATIONAL_PROGRAM_PROTOCOL_VERSION
    prompt = build_relational_program_system_prompt(
        max_program_calls,
        assistant_carrier=carrier,
    )
    return ToolScheme(
        name=RELATIONAL_PROGRAM_TOOL_SCHEME,
        protocol_version=version,
        protocol_hash=relational_program_protocol_hash(
            prompt,
            max_program_calls,
            protocol_version=version,
        ),
        system_prompt=prompt,
        assistant_carrier=carrier,
        top_level_tools=tuple(TOP_LEVEL_TOOLS),
        atomic_tools=tuple(RELATIONAL_PRIMITIVE_TOOLS),
        max_batch_calls=max_program_calls,
    )


def build_direct_sql_search_tool_scheme() -> ToolScheme:
    """Build the exclusive two-tool search plus direct-SQL diagnostic scheme."""
    from direct_sql_search_protocol import (
        DIRECT_SQL_SEARCH_ASSISTANT_CARRIER,
        DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
        DIRECT_SQL_SEARCH_TOOLS,
        build_direct_sql_search_system_prompt,
        direct_sql_search_protocol_hash,
    )

    prompt = build_direct_sql_search_system_prompt()
    tools = tuple(sorted(DIRECT_SQL_SEARCH_TOOLS))
    return ToolScheme(
        name=DIRECT_SQL_SEARCH_TOOL_SCHEME,
        protocol_version=DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
        protocol_hash=direct_sql_search_protocol_hash(prompt),
        system_prompt=prompt,
        assistant_carrier=DIRECT_SQL_SEARCH_ASSISTANT_CARRIER,
        top_level_tools=tools,
        atomic_tools=tools,
    )


def build_tool_scheme(
    name: str,
    *,
    system_prompt: str | None = None,
    max_batch_calls: int | None = None,
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
            "non-atomic system prompts are derived from their carrier and call bound"
        )
    if name == ACTION_BLOCK_TOOL_SCHEME:
        return build_action_block_tool_scheme(
            max_batch_calls=max_batch_calls or 8,
            assistant_carrier=assistant_carrier,
            protocol_version=protocol_version,
        )
    if name == RELATIONAL_PROGRAM_TOOL_SCHEME:
        return build_relational_program_tool_scheme(
            max_program_calls=max_batch_calls or 8,
            assistant_carrier=assistant_carrier,
            protocol_version=protocol_version,
        )
    if max_batch_calls is not None or assistant_carrier is not None or protocol_version is not None:
        raise ValueError(
            "direct-sql-search scheme owns its carrier, version, and single-action boundary"
        )
    return build_direct_sql_search_tool_scheme()


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
    if scheme.name == RELATIONAL_PROGRAM_TOOL_SCHEME:
        from relational_program_protocol import render_relational_program_assistant

        return render_relational_program_assistant(reasoning, tool, arguments)
    if scheme.name == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        from action_carrier import render_action_carrier

        return render_action_carrier(reasoning, tool, arguments)
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
    if scheme.name == RELATIONAL_PROGRAM_TOOL_SCHEME:
        from relational_program_protocol import parse_relational_program_assistant

        return parse_relational_program_assistant(
            text,
            max_batch_calls=int(scheme.max_batch_calls or 0),
        )
    if scheme.name == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        from direct_sql_search_protocol import parse_direct_sql_search_action

        return parse_direct_sql_search_action(text)
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
