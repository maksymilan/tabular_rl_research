"""Atomic version50: version39 semantics with DeepSeek-native function-call transport."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import protocol as version39
from tool_modules.native_tool_bundle.provider_tools import (
    NATIVE_ASSISTANT_CARRIER,
    native_tools_sha256,
)


PROTOCOL_VERSION = "version50"
HISTORY_TURNS = 4
STUDENT_SYSTEM_PROMPT = version39.STUDENT_SYSTEM_PROMPT
TOOLS = version39.TOOLS
MODEL_ARG_SCHEMA = version39.MODEL_ARG_SCHEMA
PROVIDER_ASSISTANT_CARRIER = NATIVE_ASSISTANT_CARRIER

parse_assistant_strict = version39.parse_assistant_strict
validate_model_action = version39.validate_model_action
tool_schema_hash = version39.tool_schema_hash


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "tool_schema_sha256": tool_schema_hash(),
            "provider_assistant_carrier": PROVIDER_ASSISTANT_CARRIER,
            "native_tools_sha256": native_tools_sha256(),
            "history_turns": HISTORY_TURNS,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
