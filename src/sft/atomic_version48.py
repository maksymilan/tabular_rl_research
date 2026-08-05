"""Atomic version48: version47 context plus brief interpret-before-act guidance."""
from __future__ import annotations

import hashlib
import json

import atomic_version47 as version47


PROTOCOL_VERSION = "version48"
HISTORY_TURNS = version47.HISTORY_TURNS
RESIDENT_STATE_PROFILE = version47.RESIDENT_STATE_PROFILE
LATEST_OBSERVATION_FULL = version47.LATEST_OBSERVATION_FULL
STUDENT_SYSTEM_PROMPT = version47.STUDENT_SYSTEM_PROMPT
TOOLS = version47.TOOLS
MODEL_ARG_SCHEMA = version47.MODEL_ARG_SCHEMA

INTERPRET_BEFORE_ACT_SUFFIX = (
    "\n\nOBSERVATION CONTINUITY\n"
    "After the first tool result, begin the next reason with one brief statement of what the "
    "latest Harness output established and how it changes the next action. Continue from earlier "
    "reasoning and tool feedback, but do not copy the full output or treat your own summary as "
    "factual evidence. If an exact row value is archived, observe it again before reuse."
)

parse_assistant_strict = version47.parse_assistant_strict
validate_model_action = version47.validate_model_action
tool_schema_hash = version47.tool_schema_hash


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "tool_schema_sha256": tool_schema_hash(),
            "resident_state_profile": RESIDENT_STATE_PROFILE,
            "latest_observation_full": LATEST_OBSERVATION_FULL,
            "interpret_before_act": True,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
