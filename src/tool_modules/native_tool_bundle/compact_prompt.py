"""Compact role-separated prompts for provider-native atomic tool bundles.

The provider already receives the complete typed function schema on every request.  Repeating
tool signatures, nested argument grammar, long per-tool descriptions, and a JSON call cookbook in
the system message spends context without adding a second source of truth.  This module keeps only
semantics that are not represented by the native schema: resident-state authority, relational and
terminal invariants, bundle scheduling, and causal recovery from environment errors.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from prompt_contract import (
    STUDENT_CONTEXT_CONTRACT,
    TEACHER_SEMANTIC_DECISION_DISCIPLINE,
)


PROMPT_PROFILE = "native-schema-semantic-only-v1"

SOFT_BUNDLE_POLICY = (
    "BUNDLE SCHEDULING (soft policy)\n"
    "1. Default to one function call. Normally use no more than three calls in one assistant turn, "
    "and bundle only calls that are independently justified by the same visible pre-call state.\n"
    "2. Prefer a bundle for independent perception work such as describing several source tables, "
    "inspecting several known columns, or reading several already-resident tables.\n"
    "3. Parallel condition_filter calls are appropriate only for explicit competing hypotheses "
    "grounded in the question, external knowledge, or observed values; do not invent speculative "
    "branches merely to increase coverage.\n"
    "4. Normally issue join_tables, group_aggregate, scalar_compute, project, "
    "extreme_value_select, and set_op in a single-call turn because the next relational decision "
    "usually depends on observing that result. A call in a bundle must never consume a handle, "
    "step id, value, or schema produced by another call in the same bundle."
)

ERROR_RECOVERY_CONTRACT = (
    "ERROR FEEDBACK AND RECOVERY\n"
    "Every tool result, including a structured error, is authoritative feedback from the "
    "environment. After an error, use its type, code, message, details, attempted function, and "
    "attempted arguments to correct the next call; do not repeat the same invalid arguments. "
    "An errored call is not data evidence and its rejected state change cannot be assumed. The "
    "errored assistant turn and tool feedback remain in causal history so a later corrected call "
    "can learn recovery from the observed failure."
)

EXECUTION_RULES = (
    "EXECUTION AND ANSWER RULES\n"
    "The API-supplied native function schemas are the sole authority for function names, required "
    "fields, optional fields, and nested argument shapes. Put executable arguments only in native "
    "function calls; assistant content is audit-only and cannot supply evidence.\n"
    "Resolve schemas before using unknown columns. Use resident handles and exact logical columns; "
    "after joins, dotted identifiers are relation.column, while a bare downstream name is valid "
    "only when it resolves uniquely. A value_ref cites the producing scalar or one-row metric step, "
    "not a perception, read, or plan step. Relational operators must preserve the requested "
    "population and row grain.\n"
    "Terminal evidence is scored from the cited table only. Before answering, read and derive "
    "exactly the requested rows, separate output fields, column order, and stored representation; "
    "remove helper columns and do not substitute labels for IDs/codes. answer_from_context must be "
    "the sole function call in its assistant turn. This is a hard harness rule, including after an "
    "earlier error."
)

ROLLING_HISTORY_RULE = (
    "ROLLING CAUSAL HISTORY\n"
    "The request may contain only the most recent provider assistant turns and all matching tool "
    "results. Continue from that bounded suffix without assuming it is complete. CURRENT "
    "ENVIRONMENT STATE is authoritative; LAST TOOL ERROR is the latest rejected action. Earlier "
    "assistant reasoning is a decision trace, not factual evidence."
)

TEACHER_GENERATION_RULES = (
    "TEACHER DATA GENERATION\n"
    "Give one non-empty brief native reasoning field that is specific to the current question, "
    "visible state or latest error, and the calls being selected; keep it under 80 words. Leave "
    "assistant content empty because it is audit-only, while any provider-emitted non-empty content "
    "is retained but never executed. Do not restate the question after the first turn. Do not "
    "repeat an already-visible read_subtable call; change its columns, "
    "conditions, order, offset, or limit when different rows are needed. A plan item with no "
    "evidence must omit evidence or use null, never an empty string. The harness returns one "
    "role=tool result per call id before the next assistant decision."
)


NATIVE_STUDENT_SYSTEM_PROMPT = (
    "You are a relational table-tool agent. Answer the question by interacting with the database "
    "through the supplied native functions.\n\n"
    "CONTEXT\n"
    + STUDENT_CONTEXT_CONTRACT
    + "\n\n"
    + EXECUTION_RULES
    + "\n\n"
    + SOFT_BUNDLE_POLICY
    + "\n\n"
    + ERROR_RECOVERY_CONTRACT
    + "\n\n"
    + ROLLING_HISTORY_RULE
)

NATIVE_TEACHER_SYSTEM_PROMPT = (
    NATIVE_STUDENT_SYSTEM_PROMPT
    + "\n\nTEACHER-ONLY SEMANTIC QUALITY CONTROL\n"
    + TEACHER_SEMANTIC_DECISION_DISCIPLINE
    + "\n\n"
    + TEACHER_GENERATION_RULES
)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def static_request_size_audit(native_tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic character counts for prompt/schema regression tests and manifests."""
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
        "teacher_provider_prompt_sha256": prompt_sha256(
            NATIVE_TEACHER_SYSTEM_PROMPT
        ),
    }
