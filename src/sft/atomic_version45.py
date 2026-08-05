"""Atomic version45: bounded SQL candidate recall for version44 value search."""
from __future__ import annotations

import hashlib
import json

import atomic_version44 as version44
from public_tool_contract import (
    VERSION45_PUBLIC_TOOL_ARGUMENTS,
    VERSION45_PUBLIC_TOOL_CONTRACTS,
)


PROTOCOL_VERSION = "version45"
HISTORY_TURNS = version44.HISTORY_TURNS


def _tool_signature(
    tool: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> str:
    arguments = [*required, *(f"{name}?" for name in optional)]
    return f"{tool}({', '.join(arguments)})"


TOOL_SPECS: dict[str, str] = {
    tool: (
        f"{_tool_signature(tool, required, optional)} -> "
        f"{VERSION45_PUBLIC_TOOL_CONTRACTS[tool].semantics}"
    )
    for tool, (required, optional) in VERSION45_PUBLIC_TOOL_ARGUMENTS.items()
}
TOOLS = frozenset(TOOL_SPECS)
MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in VERSION45_PUBLIC_TOOL_ARGUMENTS.items()
}

_OLD_RUNTIME_SEARCH_RULE = (
    "Its table is mandatory, omitted column means every column in that table, and offset paginates "
    "the same deterministic relevance ordering. A search observes values but never filters rows "
    "or creates a relation."
)
_NEW_RUNTIME_SEARCH_RULE = (
    "Its table is mandatory, omitted column means every column in that table, and offset paginates "
    "one stable bounded candidate-pool relevance ordering. candidate_truncated=true means recall "
    "saturated: narrow table, column, or query instead of treating the result as exhaustive. A "
    "case-insensitive exact hit ends broader fuzzy recall. A search observes values but never "
    "filters rows or creates a relation."
)

STUDENT_SYSTEM_PROMPT = version44.STUDENT_SYSTEM_PROMPT.replace(
    version44.TOOL_SPECS["search_values"],
    TOOL_SPECS["search_values"],
).replace(
    _OLD_RUNTIME_SEARCH_RULE,
    _NEW_RUNTIME_SEARCH_RULE,
)
if STUDENT_SYSTEM_PROMPT == version44.STUDENT_SYSTEM_PROMPT:
    raise RuntimeError("version45 student search contract replacement did not apply")

_OLD_TEACHER_SEARCH_GUIDANCE = version44._teacher_tool_guidance()["search_values"]
_NEW_TEACHER_SEARCH_GUIDANCE = (
    "search_values(table, query, column=None, limit=20, offset=0) -> search a deterministic "
    "SQL-recalled candidate pool in one mandatory table, then lexically rank exact stored values. "
    "Omit column to search every column in that table; provide it whenever the target column is "
    "known. The response contains at most 20 matches and next_offset when another candidate-pool "
    "page exists. candidate_truncated=true means recall saturated; narrow table, column, or query "
    "rather than assuming the response is exhaustive. Exact or case-insensitive exact hits suppress "
    "broader fuzzy alternatives. Copy a returned exact value into a later filter only when its "
    "table/column and question semantics match the intended predicate. Similar candidates are "
    "alternatives, not instructions to broaden an exact predicate. This tool observes values but "
    "does not filter, project, rank database rows, or produce terminal evidence."
)


def teacher_system_prompt(student_prompt: str = STUDENT_SYSTEM_PROMPT) -> str:
    prompt = version44.teacher_system_prompt(student_prompt)
    replaced = prompt.replace(
        _OLD_TEACHER_SEARCH_GUIDANCE,
        _NEW_TEACHER_SEARCH_GUIDANCE,
    )
    if replaced == prompt:
        raise RuntimeError("version45 teacher search guidance replacement did not apply")
    return replaced


def provider_system_prompt(
    model: str,
    canonical_prompt: str,
    *,
    carrier: str = "json-output",
) -> str:
    return version44.provider_system_prompt(model, canonical_prompt, carrier=carrier)


validate_model_action = version44.validate_model_action
parse_assistant_strict = version44.parse_assistant_strict


def tool_schema_hash() -> str:
    arguments = {
        tool: {
            "required": sorted(required),
            "optional": sorted(optional),
        }
        for tool, (required, optional) in sorted(MODEL_ARG_SCHEMA.items())
    }
    payload = json.dumps(
        {"tools": TOOL_SPECS, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_hash(system_prompt: str) -> str:
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt,
            "tool_schema_sha256": tool_schema_hash(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
