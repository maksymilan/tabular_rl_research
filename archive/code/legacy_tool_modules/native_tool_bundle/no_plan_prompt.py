"""Version54 prompt profile: version53 runtime semantics without the removed plan tool."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from tool_modules.native_tool_bundle.reviewed_prompt import (
    NATIVE_STUDENT_SYSTEM_PROMPT as VERSION53_STUDENT_SYSTEM_PROMPT,
    TEACHER_GENERATION_RULES as VERSION53_TEACHER_GENERATION_RULES,
    TEACHER_SEMANTIC_QUALITY,
)


PROMPT_PROFILE = "native-schema-role-separated-no-plan-v1"
_PLAN_GUIDANCE = (
    "Plan evidence must be grounded; otherwise omit it or use null, never an empty string."
)
if VERSION53_TEACHER_GENERATION_RULES.count(_PLAN_GUIDANCE) != 1:
    raise RuntimeError("version53 teacher plan guidance changed; review version54 no-plan prompt")

TEACHER_GENERATION_RULES = VERSION53_TEACHER_GENERATION_RULES.replace(
    " After an error, name the failure and feedback-supported correction. " + _PLAN_GUIDANCE,
    " After an error, name the failure and feedback-supported correction.",
)

NATIVE_STUDENT_SYSTEM_PROMPT = VERSION53_STUDENT_SYSTEM_PROMPT
NATIVE_TEACHER_SYSTEM_PROMPT = (
    NATIVE_STUDENT_SYSTEM_PROMPT
    + "\n\n"
    + TEACHER_SEMANTIC_QUALITY
    + "\n\n"
    + TEACHER_GENERATION_RULES
)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def static_request_size_audit(native_tools: list[dict[str, Any]]) -> dict[str, Any]:
    schema_json = json.dumps(
        native_tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "profile": PROMPT_PROFILE,
        "student_prompt_chars": len(NATIVE_STUDENT_SYSTEM_PROMPT),
        "teacher_provider_prompt_chars": len(NATIVE_TEACHER_SYSTEM_PROMPT),
        "native_tools_json_chars": len(schema_json),
        "static_request_chars": len(NATIVE_TEACHER_SYSTEM_PROMPT) + len(schema_json),
        "student_prompt_sha256": prompt_sha256(NATIVE_STUDENT_SYSTEM_PROMPT),
        "teacher_provider_prompt_sha256": prompt_sha256(NATIVE_TEACHER_SYSTEM_PROMPT),
    }
