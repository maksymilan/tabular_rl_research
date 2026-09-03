"""Version51: bounded DeepSeek-native bundles over the version39 atomic tools.

The public primitive functions, harness execution, resident state, and terminal denotation are
unchanged. The model-turn boundary is new: one provider assistant turn may author one to eight
independent primitive calls from the same visible pre-state. Primitive results are returned
together using the provider's native assistant/tool message protocol.
"""
from __future__ import annotations

import hashlib
import json

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

import protocol as version39
from tool_modules.native_tool_bundle.provider_tools import (
    MAX_NATIVE_BUNDLE_CALLS,
    NATIVE_BUNDLE_ASSISTANT_CARRIER,
    native_tools_sha256,
)


PROTOCOL_VERSION = "version51"
HISTORY_TURNS = 4
STUDENT_SYSTEM_PROMPT = version39.STUDENT_SYSTEM_PROMPT
TOOLS = version39.TOOLS
MODEL_ARG_SCHEMA = version39.MODEL_ARG_SCHEMA
PROVIDER_ASSISTANT_CARRIER = NATIVE_BUNDLE_ASSISTANT_CARRIER

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
            "max_native_bundle_calls": MAX_NATIVE_BUNDLE_CALLS,
            "history_turns": HISTORY_TURNS,
            "history_unit": "provider-native-assistant-turn",
            "preexecution_reference_state": "bundle-pre-state",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
