"""Atomic version47: handle cards plus transient full row observations."""
from __future__ import annotations

import hashlib
import json

import atomic_version46 as version46
from protocol import RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS


PROTOCOL_VERSION = "version47"
HISTORY_TURNS = version46.HISTORY_TURNS
RESIDENT_STATE_PROFILE = RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS
LATEST_OBSERVATION_FULL = True
STUDENT_SYSTEM_PROMPT = version46.STUDENT_SYSTEM_PROMPT
TOOLS = version46.TOOLS
MODEL_ARG_SCHEMA = version46.MODEL_ARG_SCHEMA

parse_assistant_strict = version46.parse_assistant_strict
validate_model_action = version46.validate_model_action
tool_schema_hash = version46.tool_schema_hash


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
