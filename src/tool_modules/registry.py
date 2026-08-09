#!/usr/bin/env python3
"""Public registry for the independently selectable table-tool schemes.

The schemes share the active think-plus-raw-JSON carrier. The relational schemes also share
harness-owned atomic semantics; the two SQL schemes instead share only the immutable database,
causal loop, and hidden scorer. Schemes do not share a model-visible tool schema, action validator,
prompt, trajectory identity, or training manifest. This module is deliberately a small adapter
boundary so callers never infer a scheme from record shape or experimental flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

from action_carrier import ACTIVE_ACTION_CARRIER


TOOL_SCHEME_REGISTRY_VERSION = "tool-scheme-registry-v12"
ATOMIC_TOOL_SCHEME = "atomic"
NATIVE_TOOL_BUNDLE_SCHEME = "native-tool-bundle"
CHECKPOINT_RELALG_TOOL_SCHEME = "checkpoint-relalg"
# All new tool-design, teacher-rollout, evaluation, and future training work starts here.
# Older schemes remain registered so active RL runs and frozen artifacts stay reproducible.
FORWARD_TOOL_SCHEME = CHECKPOINT_RELALG_TOOL_SCHEME
ACTION_BLOCK_TOOL_SCHEME = "action-block"
RELATIONAL_PROGRAM_TOOL_SCHEME = "relational-program"
DIRECT_SQL_SEARCH_TOOL_SCHEME = "direct-sql-search"
ITERATIVE_SQL_TOOL_SCHEME = "iterative-sql"
TOOL_SCHEME_NAMES = (
    ATOMIC_TOOL_SCHEME,
    NATIVE_TOOL_BUNDLE_SCHEME,
    CHECKPOINT_RELALG_TOOL_SCHEME,
    ACTION_BLOCK_TOOL_SCHEME,
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    DIRECT_SQL_SEARCH_TOOL_SCHEME,
    ITERATIVE_SQL_TOOL_SCHEME,
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
    mode: str | None = None
    tool_schema_hash: str | None = None
    student_prompt_hash: str | None = None
    teacher_prompt_hash: str | None = None
    admission_status: str | None = None

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
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


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
    from tool_modules.action_block.protocol import (
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


def build_native_tool_bundle_scheme() -> ToolScheme:
    """Build the direct DeepSeek-native multi-call experiment."""
    from tool_modules.native_tool_bundle.no_plan_protocol import (
        MODEL_ARG_SCHEMA,
        PROTOCOL_VERSION,
        PROVIDER_ASSISTANT_CARRIER,
        STUDENT_SYSTEM_PROMPT,
        protocol_hash,
    )
    from tool_modules.native_tool_bundle.provider_tools import MAX_NATIVE_BUNDLE_CALLS

    tools = tuple(sorted(MODEL_ARG_SCHEMA))
    return ToolScheme(
        name=NATIVE_TOOL_BUNDLE_SCHEME,
        protocol_version=PROTOCOL_VERSION,
        protocol_hash=protocol_hash(STUDENT_SYSTEM_PROMPT),
        system_prompt=STUDENT_SYSTEM_PROMPT,
        assistant_carrier=PROVIDER_ASSISTANT_CARRIER,
        top_level_tools=tools,
        atomic_tools=tuple(
            tool for tool in tools if tool not in {"plan", "answer_from_context"}
        ),
        max_batch_calls=MAX_NATIVE_BUNDLE_CALLS,
    )


def build_checkpoint_relalg_tool_scheme(*, mode: str) -> ToolScheme:
    """Build the forward checkpointed Direct/Atomic/Hybrid scheme.

    ``mode`` is deliberately mandatory: the three capability surfaces share one state
    protocol but never appear together accidentally.
    """
    from tool_modules.checkpoint_relalg.protocol import (
        ADMISSION_STATUS,
        MODE_TOOLS,
        NATIVE_ASSISTANT_CARRIER,
        PROTOCOL_VERSION,
        get_system_prompt,
        normalize_mode,
        prompt_hash,
        tool_schema_hash,
    )

    active_mode = normalize_mode(mode)
    student_prompt = get_system_prompt(active_mode, teacher=False)
    student_hash = prompt_hash(active_mode, teacher=False)
    teacher_hash = prompt_hash(active_mode, teacher=True)
    schema_hash = tool_schema_hash(active_mode)
    identity = hashlib.sha256(json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "mode": active_mode,
            "student_prompt_sha256": student_hash,
            "tool_schema_sha256": schema_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    tools = tuple(MODE_TOOLS[active_mode])
    return ToolScheme(
        name=CHECKPOINT_RELALG_TOOL_SCHEME,
        protocol_version=PROTOCOL_VERSION,
        protocol_hash=identity,
        system_prompt=student_prompt,
        assistant_carrier=NATIVE_ASSISTANT_CARRIER,
        top_level_tools=tools,
        atomic_tools=tuple(
            tool
            for tool in tools
            if tool not in {
                "describe_table",
                "inspect_column",
                "read_rows",
                "execute_sql",
                "commit_checkpoint",
                "restore_checkpoint",
                "answer",
            }
        ),
        max_batch_calls=1,
        mode=active_mode,
        tool_schema_hash=schema_hash,
        student_prompt_hash=student_hash,
        teacher_prompt_hash=teacher_hash,
        admission_status=ADMISSION_STATUS,
    )


def build_relational_program_tool_scheme(
    *,
    max_program_calls: int = 8,
    assistant_carrier: str | None = None,
    protocol_version: str | None = None,
) -> ToolScheme:
    """Build the separate declarative-program scheme."""
    from tool_modules.relational_program.protocol import (
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
    from tool_modules.direct_sql_search.protocol import (
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


def build_iterative_sql_tool_scheme() -> ToolScheme:
    """Build the direct SQL exploration plus recoverable final-submission scheme."""
    from tool_modules.iterative_sql.protocol import (
        ITERATIVE_SQL_ASSISTANT_CARRIER,
        ITERATIVE_SQL_PROTOCOL_VERSION,
        ITERATIVE_SQL_TOOLS,
        build_iterative_sql_system_prompt,
        iterative_sql_protocol_hash,
    )

    prompt = build_iterative_sql_system_prompt()
    tools = tuple(sorted(ITERATIVE_SQL_TOOLS))
    return ToolScheme(
        name=ITERATIVE_SQL_TOOL_SCHEME,
        protocol_version=ITERATIVE_SQL_PROTOCOL_VERSION,
        protocol_hash=iterative_sql_protocol_hash(prompt),
        system_prompt=prompt,
        assistant_carrier=ITERATIVE_SQL_ASSISTANT_CARRIER,
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
    mode: str | None = None,
) -> ToolScheme:
    require_tool_scheme(name)
    if name == ATOMIC_TOOL_SCHEME:
        if mode is not None:
            raise ValueError("atomic scheme does not accept checkpoint-relalg mode")
        if assistant_carrier is not None or protocol_version is not None:
            raise ValueError(
                "atomic scheme carrier/version are owned by the atomic protocol"
            )
        return build_atomic_tool_scheme(system_prompt=system_prompt)
    if name == NATIVE_TOOL_BUNDLE_SCHEME:
        if mode is not None:
            raise ValueError("native-tool-bundle does not accept checkpoint-relalg mode")
        if system_prompt is not None or max_batch_calls is not None:
            raise ValueError(
                "native-tool-bundle owns its prompt and bounded provider call count"
            )
        if assistant_carrier is not None or protocol_version is not None:
            raise ValueError(
                "native-tool-bundle carrier/version are owned by version54"
            )
        return build_native_tool_bundle_scheme()
    if name == CHECKPOINT_RELALG_TOOL_SCHEME:
        if system_prompt is not None or max_batch_calls is not None:
            raise ValueError(
                "checkpoint-relalg owns its prompt and single-call boundary"
            )
        if assistant_carrier is not None or protocol_version is not None:
            raise ValueError(
                "checkpoint-relalg carrier/version are owned by checkpoint-relalg-v1"
            )
        if mode is None:
            raise ValueError("checkpoint-relalg requires mode=direct|atomic|hybrid")
        return build_checkpoint_relalg_tool_scheme(mode=mode)
    if mode is not None:
        raise ValueError(f"{name} does not accept checkpoint-relalg mode")
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
            "SQL schemes own their carrier, version, and single-action boundary"
        )
    if name == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        return build_direct_sql_search_tool_scheme()
    return build_iterative_sql_tool_scheme()


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
    if scheme.name == NATIVE_TOOL_BUNDLE_SCHEME:
        raise ValueError(
            "native-tool-bundle targets are structured provider messages, not text actions"
        )
    if scheme.name == CHECKPOINT_RELALG_TOOL_SCHEME:
        raise ValueError(
            "checkpoint-relalg targets are structured provider messages, not text actions"
        )
    if scheme.name == ACTION_BLOCK_TOOL_SCHEME:
        from tool_modules.action_block.protocol import render_batch_plan_assistant

        return render_batch_plan_assistant(reasoning, tool, arguments)
    if scheme.name == RELATIONAL_PROGRAM_TOOL_SCHEME:
        from tool_modules.relational_program.protocol import render_relational_program_assistant

        return render_relational_program_assistant(reasoning, tool, arguments)
    if scheme.name == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        from action_carrier import render_action_carrier

        return render_action_carrier(reasoning, tool, arguments)
    if scheme.name == ITERATIVE_SQL_TOOL_SCHEME:
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
    if scheme.name == NATIVE_TOOL_BUNDLE_SCHEME:
        raise ValueError(
            "native-tool-bundle actions must be read from structured provider tool_calls"
        )
    if scheme.name == CHECKPOINT_RELALG_TOOL_SCHEME:
        raise ValueError(
            "checkpoint-relalg actions must be read from structured provider tool_calls"
        )
    if scheme.name == ACTION_BLOCK_TOOL_SCHEME:
        from tool_modules.action_block.protocol import parse_batch_plan_assistant

        return parse_batch_plan_assistant(
            text,
            max_batch_calls=int(scheme.max_batch_calls or 0),
        )
    if scheme.name == RELATIONAL_PROGRAM_TOOL_SCHEME:
        from tool_modules.relational_program.protocol import parse_relational_program_assistant

        return parse_relational_program_assistant(
            text,
            max_batch_calls=int(scheme.max_batch_calls or 0),
        )
    if scheme.name == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        from tool_modules.direct_sql_search.protocol import parse_direct_sql_search_action

        return parse_direct_sql_search_action(text)
    if scheme.name == ITERATIVE_SQL_TOOL_SCHEME:
        from tool_modules.iterative_sql.protocol import parse_iterative_sql_action

        return parse_iterative_sql_action(text)
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
