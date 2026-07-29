"""Isolated atomic version40 prompt and public action contract.

Version40 is a diagnostic profile, not the promoted default. It keeps version39 execution
semantics and join arguments unchanged while testing four model-facing changes together:

- remove ``plan`` from the public action surface;
- rename the read-only row observer to ``inspect_rows``;
- use one layered prompt without duplicated teacher/runtime policy prose;
- retain complete reasoning and observations for the recent four successful action pairs.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from protocol import (
    AdjacentActionGuard,
    ProtocolError,
    parse_assistant_structure_strict,
    validate_version40_model_arguments,
)
from provider_adapter import (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    VERSION40_RESPONSE_CONTRACT_PLACEHOLDER,
    provider_system_prompt_version40,
)
from public_tool_contract import (
    VERSION40_PUBLIC_TOOL_ARGUMENTS,
    VERSION40_PUBLIC_TOOL_CONTRACTS,
)


PROTOCOL_VERSION = "version40"
HISTORY_TURNS = 4


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
        f"{VERSION40_PUBLIC_TOOL_CONTRACTS[tool].semantics}"
    )
    for tool, (required, optional) in VERSION40_PUBLIC_TOOL_ARGUMENTS.items()
}
TOOLS = frozenset(TOOL_SPECS)
MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in VERSION40_PUBLIC_TOOL_ARGUMENTS.items()
}


PROMPT_TEMPLATE = (
    "BACKGROUND\n"
    "You are a relational table-tool agent working through a deterministic harness. The harness "
    "executes one typed action, updates grounded state, and returns factual feedback before the "
    "next decision. Model-authored reasoning is a working hypothesis, never database evidence.\n\n"
    "TASK\n"
    "Answer the user question by deriving one grounded result table. The terminal evidence table "
    "must itself contain exactly the requested rows, columns, column order, and representation; "
    "reasoning text cannot repair its data or shape.\n\n"
    "ENVIRONMENT\n"
    "- DATABASE CATALOG lists table names, row counts, and relations, but not unresolved schemas.\n"
    "- CURRENT ENVIRONMENT STATE is authoritative for known schemas, inspected values and rows, "
    "derived handles, exact logical columns, and scalar-producing step ids. A handle exposes "
    "metadata until rows are observed.\n"
    "- EXTERNAL KNOWLEDGE, when present, is a binding part of the question. Follow every stated "
    "mapping, literal, field, operator, formula, aggregation, restriction, and output requirement "
    "exactly, even when another interpretation seems more natural.\n"
    "- LEGAL HISTORY contains at most the four most recent harness-successful pairs. Each retained "
    "pair includes the complete prior reasoning, exact action, and unabridged tool result. LAST "
    "TOOL ERROR is the latest rejected action's structured feedback. Rejected text is not evidence.\n\n"
    "TOOLS\n"
    + "\n".join(TOOL_SPECS.values())
    + "\n\nARGUMENT-SHAPE EXAMPLES\n"
    "These examples clarify nested JSON only; they do not prescribe tool order.\n"
    '{"tool":"inspect_rows","arguments":{"table":"transactions","columns":["id","amount"],'
    '"conditions":{"column":"posted_at","op":"on_date","value":"2024-01-31"},'
    '"order_by":["id"],"limit":5}}\n'
    '{"tool":"condition_filter","arguments":{"table":"people","conditions":{"and":['
    '{"column":"city","op":"=","value":"Paris"},'
    '{"column":"age","op":">=","value":18}]}}}\n'
    '{"tool":"join_tables","arguments":{"base":"orders","joins":['
    '{"table":"customers","on":[{"left":"orders.customer_id","right":"id"}]},'
    '{"table":"regions","on":[{"left":"customers.region_id","right":"id"}]}]}}\n'
    '{"tool":"group_aggregate","arguments":{"table":"eligible_people","group_by":[],'
    '"aggregations":[{"op":"count","column":"*","as":"female_count","where":'
    '{"column":"gender","op":"=","value":"F"}},{"op":"count","column":"*",'
    '"as":"male_count","where":{"column":"gender","op":"=","value":"M"}}]}}\n'
    '{"tool":"answer_from_context","arguments":{"evidence":{"table":"project_003"},'
    '"reason":"The cited table has the requested result shape."}}\n\n'
    "REASONING CONTINUITY\n"
    "Reason in a focused, medium-length continuation, normally two to six substantive sentences. "
    "Use the retained reasoning to preserve useful hypotheses and unfinished decisions, then update "
    "them from the latest tool result, current state, or error. Do not restart from the original "
    "question each turn, copy the whole history, continue at excessive length, or jump to an "
    "unsupported terse conclusion.\n\n"
    "RULES\n"
    "1. Use only exact visible table handles and columns. Resolve an unknown schema before using "
    "its columns; after a join, dotted identifiers are exact relation.column names and a bare "
    "downstream name is valid only when unique.\n"
    "2. Keep the eligible population and row grain fixed unless grounded feedback changes them. "
    "Do not invent earliest/latest/current/top-1, deduplication, aggregation, or other restrictions "
    "merely to obtain a plausible or singleton result.\n"
    "3. Treat unexpected empty results, multiplicity, NULLs, dates, or arithmetic as signals to "
    "inspect the relevant schema, values, rows, or relationship and revise only the unsupported "
    "assumption.\n"
    "4. A value_ref cites the step that produced a resident scalar or one-row metric table. "
    "Observation steps are not scalar producers; use value_ref+column for one named metric.\n"
    "5. Row observation never filters, projects, or creates a new handle. Use relational tools to "
    "change rows or columns, and remove helper columns before citing terminal evidence.\n\n"
    "RESPONSE CONTRACT\n"
    + VERSION40_RESPONSE_CONTRACT_PLACEHOLDER
)

STUDENT_SYSTEM_PROMPT = provider_system_prompt_version40(
    "canonical-non-split-model",
    PROMPT_TEMPLATE,
)


def provider_system_prompt(
    model: str,
    *,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Return the single provider-specific response contract for version40."""
    return provider_system_prompt_version40(
        model,
        PROMPT_TEMPLATE,
        carrier=carrier,
    )


def validate_model_action(tool: str, args: dict) -> None:
    """Validate one version40 model action and attach its public schema on failure."""
    if tool not in TOOLS:
        raise ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}",
            code="unknown_tool",
            details={"legal_tools": sorted(TOOLS)},
            attempted_tool=tool,
            attempted_arguments=args,
        )
    try:
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
    """Parse one active version40 action without widening the shared carrier."""
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


def tool_schema_hash() -> str:
    """Hash the exact version40 public names, arguments, and semantics."""
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
    """Hash the version40 prompt and public tool schema."""
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
