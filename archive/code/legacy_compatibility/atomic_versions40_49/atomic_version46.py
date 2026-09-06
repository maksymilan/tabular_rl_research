"""Atomic version46: version39 behavior with Harness-authored relation handle cards."""
from __future__ import annotations

import hashlib
import json

import protocol as version39


PROTOCOL_VERSION = "version46"
HISTORY_TURNS = 4
RESIDENT_STATE_PROFILE = version39.RESIDENT_STATE_PROFILE_HANDLE_CARDS
LATEST_OBSERVATION_FULL = False
STUDENT_SYSTEM_PROMPT = version39.STUDENT_SYSTEM_PROMPT
TOOLS = version39.TOOLS
MODEL_ARG_SCHEMA = version39.MODEL_ARG_SCHEMA

parse_assistant_strict = version39.parse_assistant_strict
validate_model_action = version39.validate_model_action
tool_schema_hash = version39.tool_schema_hash


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "tool_schema_sha256": tool_schema_hash(),
            "resident_state_profile": RESIDENT_STATE_PROFILE,
            "latest_observation_full": LATEST_OBSERVATION_FULL,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
