"""Version53 prompt-only review over unchanged native-bundle execution semantics."""
from __future__ import annotations

import hashlib
import json

from tool_modules.native_tool_bundle.protocol import (
    HISTORY_TURNS,
    MAX_NATIVE_BUNDLE_CALLS,
    MODEL_ARG_SCHEMA,
    PROVIDER_ASSISTANT_CARRIER,
    TOOLS,
    tool_schema_hash,
    validate_model_action,
)
from tool_modules.native_tool_bundle.provider_tools import native_tools_sha256
from tool_modules.native_tool_bundle.reviewed_prompt import (
    NATIVE_STUDENT_SYSTEM_PROMPT,
    NATIVE_TEACHER_SYSTEM_PROMPT,
    PROMPT_PROFILE,
)


PROTOCOL_VERSION = "version53"
STUDENT_SYSTEM_PROMPT = NATIVE_STUDENT_SYSTEM_PROMPT
TEACHER_SYSTEM_PROMPT = NATIVE_TEACHER_SYSTEM_PROMPT


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "student_system": STUDENT_SYSTEM_PROMPT,
            "prompt_profile": PROMPT_PROFILE,
            "tool_schema_sha256": tool_schema_hash(),
            "provider_assistant_carrier": PROVIDER_ASSISTANT_CARRIER,
            "native_tools_sha256": native_tools_sha256(),
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
