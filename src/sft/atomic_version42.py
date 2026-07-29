"""Atomic version42: explicit terminal columns with deterministic harness projection."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from atomic_version41 import HISTORY_TURNS, MODEL_ARG_SCHEMA
from atomic_version42_prompt import PROMPT_TEMPLATE, TOOL_SPECS
from protocol import (
    AdjacentActionGuard,
    ProtocolError,
    parse_assistant_structure_strict,
    validate_version40_model_arguments,
)
from provider_adapter import (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    provider_system_prompt_version40,
)


PROTOCOL_VERSION = "version42"
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
    """Return version42's provider-specific prompt with the unchanged carrier."""
    return provider_system_prompt_version40(
        model,
        PROMPT_TEMPLATE,
        carrier=carrier,
    )


def _validate_terminal_action(args: dict) -> None:
    if not isinstance(args, dict):
        raise ProtocolError(
            "answer_from_context arguments must be an object",
            code="argument_validation_error",
        )
    keys = set(args)
    required, optional = MODEL_ARG_SCHEMA["answer_from_context"]
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise ProtocolError(
            "answer_from_context requires evidence and optional reason only",
            code="argument_validation_error",
            details={
                "missing_arguments": sorted(missing),
                "unknown_arguments": sorted(unknown),
            },
        )
    evidence = args.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"table", "columns"}:
        raise ProtocolError(
            'answer_from_context.evidence must be exactly {"table":handle,"columns":[...]}',
            code="argument_validation_error",
        )
    table = evidence.get("table")
    columns = evidence.get("columns")
    if not isinstance(table, str) or not table.strip():
        raise ProtocolError(
            "answer_from_context.evidence.table must be a non-empty handle",
            code="argument_validation_error",
        )
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) and column.strip() for column in columns)
    ):
        raise ProtocolError(
            "answer_from_context.evidence.columns must be a non-empty list of column strings",
            code="argument_validation_error",
        )
    folded = [column.casefold() for column in columns]
    if len(folded) != len(set(folded)):
        raise ProtocolError(
            "answer_from_context.evidence.columns must not repeat a column",
            code="argument_validation_error",
        )


def validate_model_action(tool: str, args: dict) -> None:
    """Validate the version42 action surface without widening other version40/41 tools."""
    if tool not in TOOLS:
        raise ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}",
            code="unknown_tool",
            details={"legal_tools": sorted(TOOLS)},
            attempted_tool=tool,
            attempted_arguments=args,
        )
    try:
        if tool == "answer_from_context":
            _validate_terminal_action(args)
        else:
            validate_version40_model_arguments(tool, args)
    except ProtocolError as exc:
        if exc.code == "protocol_error":
            exc.code = "argument_validation_error"
        exc.failure_type = exc.failure_type or "argument_validation_error"
        exc.attempted_tool = tool
        exc.attempted_arguments = deepcopy(args)
        required, optional = MODEL_ARG_SCHEMA[tool]
        exc.details = {
            **exc.details,
            "expected_arguments": {
                "required": sorted(required),
                "optional": sorted(optional),
            },
        }
        raise


def parse_assistant_strict(
    text: str,
    *,
    adjacent_guard: AdjacentActionGuard | None = None,
    step_id: str | None = None,
) -> tuple[str, str, dict]:
    """Parse one version42 action and require explicit terminal columns."""
    try:
        think, tool, args = parse_assistant_structure_strict(text)
    except ProtocolError:
        if adjacent_guard is not None:
            adjacent_guard.clear()
        raise
    if adjacent_guard is not None:
        if not step_id:
            raise ValueError("step_id is required when adjacent_guard is provided")
        adjacent_guard.observe(tool, args, step_id=step_id)
    try:
        validate_model_action(tool, args)
    except ProtocolError:
        if adjacent_guard is not None:
            adjacent_guard.mark_last("rejected")
        raise
    return think, tool, args


def lower_terminal_evidence(h, args: dict) -> tuple[dict, dict]:
    """Project declared terminal columns without consulting the question, gold, or model reason."""
    evidence = args["evidence"]
    table = evidence["table"]
    requested = list(evidence["columns"])
    available = list(h._cols(table))
    by_fold: dict[str, list[str]] = {}
    for column in available:
        by_fold.setdefault(column.casefold(), []).append(column)
    resolved: list[str] = []
    for index, column in enumerate(requested):
        matches = by_fold.get(column.casefold(), [])
        if len(matches) != 1:
            raise ProtocolError(
                (
                    f"answer_from_context.evidence.columns[{index}] {column!r} is not one exact "
                    f"column of {table!r}; available columns: {available}"
                ),
                code="unknown_column",
                details={
                    "argument_path": (
                        f"answer_from_context.evidence.columns[{index}]"
                    ),
                    "requested_column": column,
                    "available_columns": available,
                },
                attempted_tool="answer_from_context",
                attempted_arguments=deepcopy(args),
            )
        resolved.append(matches[0])
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
        "schema": "terminal-column-projection-v1",
        "source_table": table,
        "requested_columns": requested,
        "resolved_columns": resolved,
        "projected_table": projected_table,
        "row_count": projected.get("row_count"),
        "columns": projected.get("columns"),
    }
    return lowered, audit


def tool_schema_hash() -> str:
    """Hash version42's explicit terminal semantics and unchanged nonterminal tools."""
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
    """Hash the version42 prompt and public tool schema."""
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
