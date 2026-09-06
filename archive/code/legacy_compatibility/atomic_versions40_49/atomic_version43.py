"""Atomic version43: explicit terminal columns with deterministic unique-bare resolution."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from atomic_version41 import HISTORY_TURNS, MODEL_ARG_SCHEMA
from atomic_version42 import (
    parse_assistant_strict,
    validate_model_action,
)
from atomic_version43_prompt import PROMPT_TEMPLATE, TOOL_SPECS
from protocol import ProtocolError
from provider_adapter import (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    provider_system_prompt_version40,
)


PROTOCOL_VERSION = "version43"
TOOLS = frozenset(TOOL_SPECS)

STUDENT_SYSTEM_PROMPT = provider_system_prompt_version40(
    "canonical-non-split-model",
    PROMPT_TEMPLATE,
)


def provider_system_prompt(
    model: str,
    *,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return version43's provider-specific prompt with the unchanged carrier."""
    return provider_system_prompt_version40(
        model,
        PROMPT_TEMPLATE,
        carrier=carrier,
    )


def _resolve_terminal_column(
    requested: str,
    available: list[str],
) -> tuple[str | None, str | None, list[str]]:
    """Resolve exact names first, then one unique dotted suffix for a bare name."""
    requested_fold = requested.casefold()
    exact = [column for column in available if column.casefold() == requested_fold]
    if len(exact) == 1:
        return exact[0], "exact", exact
    if not exact and "." not in requested:
        suffix = [
            column
            for column in available
            if "." in column
            and column.rsplit(".", 1)[-1].casefold() == requested_fold
        ]
        if len(suffix) == 1:
            return suffix[0], "unique-bare", suffix
        return None, None, suffix
    return None, None, exact


def lower_terminal_evidence(h, args: dict) -> tuple[dict, dict]:
    """Project declared columns using only exact or unique-bare table-local resolution."""
    evidence = args["evidence"]
    table = evidence["table"]
    requested = list(evidence["columns"])
    available = list(h._cols(table))
    resolved: list[str] = []
    resolution: list[dict[str, str]] = []
    for index, column in enumerate(requested):
        match, mode, candidates = _resolve_terminal_column(column, available)
        if match is None or mode is None:
            raise ProtocolError(
                (
                    f"answer_from_context.evidence.columns[{index}] {column!r} is neither one "
                    f"exact column nor one uniquely resolvable bare column of {table!r}; "
                    f"candidate columns: {candidates}; available columns: {available}"
                ),
                code="unknown_column",
                details={
                    "argument_path": (
                        f"answer_from_context.evidence.columns[{index}]"
                    ),
                    "requested_column": column,
                    "candidate_columns": candidates,
                    "available_columns": available,
                    "allowed_resolution": "exact-or-unique-bare-suffix",
                },
                attempted_tool="answer_from_context",
                attempted_arguments=deepcopy(args),
            )
        resolved.append(match)
        resolution.append({
            "requested": column,
            "resolved": match,
            "mode": mode,
        })
    try:
        h.validate_project(table, resolved, False)
        projected = h.project(table, resolved, False)
    except Exception as exc:
        raise ProtocolError(
            f"answer_from_context terminal projection failed: {exc}",
            code="terminal_projection_error",
            attempted_tool="answer_from_context",
            attempted_arguments=deepcopy(args),
        ) from exc
    projected_table = projected["table_name"]
    lowered = {
        "evidence": {"table": projected_table},
        "reason": args.get("reason", ""),
    }
    audit = {
        "schema": "terminal-column-projection-v2",
        "source_table": table,
        "requested_columns": requested,
        "resolved_columns": resolved,
        "column_resolution": resolution,
        "projected_table": projected_table,
        "row_count": projected.get("row_count"),
        "columns": projected.get("columns"),
    }
    return lowered, audit


def tool_schema_hash() -> str:
    """Hash version43's terminal resolution semantics and unchanged other tools."""
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
    """Hash the version43 prompt and public tool schema."""
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
