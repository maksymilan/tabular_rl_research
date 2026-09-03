"""Version52 prompt profile over the unchanged version51 native bundle semantics."""
from __future__ import annotations

import hashlib
import json

from tool_modules.native_tool_bundle.compact_prompt import (
    NATIVE_STUDENT_SYSTEM_PROMPT,
    NATIVE_TEACHER_SYSTEM_PROMPT,
    PROMPT_PROFILE,
)
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


PROTOCOL_VERSION = "version52"
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
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
