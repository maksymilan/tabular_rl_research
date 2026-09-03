"""Version54: remove public plan from version53 and keep all other semantics fixed."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

import protocol as version39
from tool_modules.native_tool_bundle.no_plan_prompt import (
    NATIVE_STUDENT_SYSTEM_PROMPT,
    NATIVE_TEACHER_SYSTEM_PROMPT,
    PROMPT_PROFILE,
)
from tool_modules.native_tool_bundle.provider_tools import (
    MAX_NATIVE_BUNDLE_CALLS,
    NATIVE_BUNDLE_ASSISTANT_CARRIER,
    native_tools_sha256,
)
from tool_modules.native_tool_bundle.reviewed_protocol import HISTORY_TURNS


PROTOCOL_VERSION = "version54"
REMOVED_MODEL_TOOLS = frozenset({"plan"})
STUDENT_SYSTEM_PROMPT = NATIVE_STUDENT_SYSTEM_PROMPT
TEACHER_SYSTEM_PROMPT = NATIVE_TEACHER_SYSTEM_PROMPT
TOOLS = set(version39.TOOLS) - REMOVED_MODEL_TOOLS
MODEL_ARG_SCHEMA = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in version39.MODEL_ARG_SCHEMA.items()
    if tool in TOOLS
}
PROVIDER_ASSISTANT_CARRIER = NATIVE_BUNDLE_ASSISTANT_CARRIER


def validate_model_action(tool: str, args: dict) -> None:
    if tool not in TOOLS:
        raise version39.ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}",
            code="unknown_tool",
            details={"legal_tools": sorted(TOOLS)},
            attempted_tool=tool,
            attempted_arguments=deepcopy(args),
        )
    version39.validate_model_action(tool, args)


def tool_schema_hash() -> str:
    arguments = {
        tool: {
            "required": sorted(required),
            "optional": sorted(optional),
        }
        for tool, (required, optional) in sorted(MODEL_ARG_SCHEMA.items())
    }
    tool_specs = {tool: version39.TOOL_SPECS[tool] for tool in sorted(TOOLS)}
    payload = json.dumps(
        {"tools": tool_specs, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "student_system": STUDENT_SYSTEM_PROMPT,
            "prompt_profile": PROMPT_PROFILE,
            "tool_schema_sha256": tool_schema_hash(),
            "provider_assistant_carrier": PROVIDER_ASSISTANT_CARRIER,
            "native_tools_sha256": native_tools_sha256(MODEL_ARG_SCHEMA),
            "removed_model_tools": sorted(REMOVED_MODEL_TOOLS),
            "max_native_bundle_calls": MAX_NATIVE_BUNDLE_CALLS,
            "soft_default_bundle_calls": 1,
            "soft_normal_max_bundle_calls": 3,
            "terminal_bundle_calls": 1,
            "history_turns": HISTORY_TURNS,
            "history_unit": "provider-native-assistant-turn",
            "preexecution_reference_state": "bundle-pre-state",
            "error_feedback_policy": "preserve-error-turn-and-target-later-correction",
            "data_instruction_boundary": "task-data-cannot-alter-system-or-tool-policy",
            "join_cardinality_policy": "inspect-when-grain-sensitive-or-anomalous",
            "independent_filter_bundle_policy": "jointly-required-distinct-handles",
            "representation_policy": "stored-source-versus-tool-produced-derived",
            "teacher_harness_implementation_prose": False,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
