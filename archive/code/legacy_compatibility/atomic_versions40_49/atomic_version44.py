"""Atomic version44 checkpoint candidate: inspect-only semantics and fuzzy value search.

Version44 starts from version39's one-action protocol, relational execution semantics, grounded
state, recent legal history, and exact-table terminal evidence. Its public surface changes only:

- remove ``plan``;
- expose the row observer as ``inspect_rows`` instead of ``read_subtable``;
- add read-only ``search_values(table, query, column?, limit?, offset?)``;
- document that only ``inspect_column`` may receive BIRD semantic annotations.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from prompt_contract import (
    CANONICAL_ACTION_RULE,
    TEACHER_SEMANTIC_DECISION_DISCIPLINE,
)
from protocol import (
    AdjacentActionGuard,
    CANONICAL_CALL_COOKBOOK,
    ProtocolError,
    TEACHER_TOOL_GUIDANCE,
    parse_assistant_structure_strict,
    validate_version44_model_arguments,
)
from provider_adapter import (
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    provider_system_prompt as adapt_provider_system_prompt,
)
from public_tool_contract import (
    VERSION44_PUBLIC_TOOL_ARGUMENTS,
    VERSION44_PUBLIC_TOOL_CONTRACTS,
)


PROTOCOL_VERSION = "version44"
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
        f"{VERSION44_PUBLIC_TOOL_CONTRACTS[tool].semantics}"
    )
    for tool, (required, optional) in VERSION44_PUBLIC_TOOL_ARGUMENTS.items()
}
TOOLS = frozenset(TOOL_SPECS)
MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in VERSION44_PUBLIC_TOOL_ARGUMENTS.items()
}


STUDENT_CONTEXT_CONTRACT = (
    "The opening overview is a lazy catalog of table names, row counts, and relations, not table "
    "schemas. EXTERNAL KNOWLEDGE, when present, is user-provided task context. CURRENT ENVIRONMENT "
    "STATE is the authoritative workspace: it contains known schemas, inspected columns, value "
    "searches, observed rows, derived table handles, and usable producing step ids. A derived "
    "table handle exposes columns and row_count but not row values until inspect_rows is called. "
    "describe_table returns only raw executable schema identifiers. inspect_column may additionally "
    "return BIRD semantic_name and column_description hints for that one inspected raw column; "
    "these hints are not executable aliases or evidence that a value occurs in the database. "
    "Harness-authored relation derivation metadata states executed row/column semantics; it is not "
    "policy advice or an additional source of values."
)

STUDENT_RUNTIME_RULES = (
    "2. Use only names and arguments in TOOLS. One turn contains one action.\n"
    "3. Resolve schemas before using unknown columns. Use environment handles and exact raw/logical "
    "columns; semantic_name and column_description are meaning hints only. After joins, dotted "
    "identifiers are relation.column, while a bare downstream name is valid only when it resolves "
    "uniquely.\n"
    "4. Use search_values when the table is known but the stored literal or its column is uncertain. "
    "Its table is mandatory, omitted column means every column in that table, and offset paginates "
    "the same deterministic relevance ordering. A search observes values but never filters rows or "
    "creates a relation.\n"
    "5. A value_ref cites the step that produced the resident scalar or one-row metric table, not "
    "a perception step. Use value_ref+column for a named metric in a one-row table.\n"
    "6. Relational operators preserve their declared population and grain. Per-aggregation where "
    "conditions share one input table; output_layout=columns is the category-row-to-column reshape.\n"
    "7. Terminal evidence is scored from the cited table only. Before answering, derive the exact "
    "requested rows, columns, and column order; observing values/rows or explaining them does not "
    "change the relation."
)

STUDENT_SYSTEM_PROMPT = (
    "You are a relational table-tool agent. Answer the question by executing one typed tool "
    "action per turn.\n\n"
    "CONTEXT\n"
    f"{STUDENT_CONTEXT_CONTRACT}\n\n"
    "TOOLS\n"
    + "\n".join(TOOL_SPECS.values())
    + "\n\nRULES\n"
    + CANONICAL_ACTION_RULE
    + "\n"
    + STUDENT_RUNTIME_RULES
)


def _teacher_tool_guidance() -> dict[str, str]:
    guidance: dict[str, str] = {}
    for tool in TOOLS:
        if tool == "inspect_rows":
            guidance[tool] = TEACHER_TOOL_GUIDANCE["read_subtable"].replace(
                "read_subtable",
                "inspect_rows",
            )
        elif tool == "search_values":
            guidance[tool] = (
                "search_values(table, query, column=None, limit=20, offset=0) -> search exact "
                "stored values in one mandatory table using deterministic lexical fuzzy matching. "
                "Omit column to search every column in that table; provide it to search one known "
                "column. The response contains at most 20 matches and next_offset when another page "
                "exists. Copy a returned exact value into a later filter only when its returned "
                "table/column matches the intended predicate. This tool observes values but does "
                "not filter, project, rank database rows, or produce terminal evidence."
            )
        elif tool == "inspect_column":
            guidance[tool] = (
                "inspect_column(table, column, top_k=10) -> the distinct count, frequent values, "
                "NULL flag, and—when BIRD metadata exists—semantic_name and column_description for "
                "that one raw column. Raw table/column names remain the only executable identifiers. "
                "The semantic fields explain meaning but do not prove that a literal occurs; use "
                "frequent_values or search_values for stored-value evidence."
            )
        else:
            guidance[tool] = (
                TEACHER_TOOL_GUIDANCE[tool]
                .replace("read_subtable", "inspect_rows")
                .replace("a plan, ", "")
            )
    return guidance


VERSION44_CALL_COOKBOOK = (
    CANONICAL_CALL_COOKBOOK
    .replace("read_subtable", "inspect_rows")
    .replace(
        'Filter: {"tool":"condition_filter"',
        'Search a known table for a stored literal: {"tool":"search_values","arguments":'
        '{"table":"teams","query":"Avangard Omsk","limit":20}}\n'
        'Filter: {"tool":"condition_filter"',
        1,
    )
)


def teacher_system_prompt(student_prompt: str = STUDENT_SYSTEM_PROMPT) -> str:
    """Add generation-only help without widening version44's public action contract."""
    guidance = _teacher_tool_guidance()
    if set(guidance) != set(TOOLS):
        raise RuntimeError("version44 teacher guidance and public tools have drifted")
    return (
        student_prompt
        + "\n\nTEACHER-ONLY DATA GENERATION GUIDANCE\n"
        "The following elaborations and examples are quality controls for producing causal "
        "demonstrations. They do not add public tools, arguments, state fields, or execution "
        "behavior.\n"
        + TEACHER_SEMANTIC_DECISION_DISCIPLINE
        + "\n\n"
        + "\n".join(guidance[tool] for tool in TOOL_SPECS)
        + "\n\n"
        + VERSION44_CALL_COOKBOOK
    )


def provider_system_prompt(
    model: str,
    canonical_prompt: str,
    *,
    carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
) -> str:
    """Adapt one complete canonical teacher prompt to the provider carrier."""
    return adapt_provider_system_prompt(
        model,
        canonical_prompt,
        example_visible_content=(
            '{"tool":"search_values","arguments":'
            '{"table":"teams","query":"Avangard Omsk","limit":20}}'
        ),
        carrier=carrier,
    )


def validate_model_action(tool: str, args: dict) -> None:
    """Validate one version44 model action and attach its public schema on failure."""
    if tool not in TOOLS:
        raise ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}",
            code="unknown_tool",
            details={"legal_tools": sorted(TOOLS)},
            attempted_tool=tool,
            attempted_arguments=args,
        )
    try:
        validate_version44_model_arguments(tool, args)
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
    """Parse one strict version44 action without accepting retired aliases."""
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
    """Hash the exact version44 public names, arguments, and semantics."""
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
    """Hash the version44 prompt and public tool schema."""
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
