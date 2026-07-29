"""Prompt-only additions for the atomic version41 diagnostic.

Version40 deliberately compressed the external-teacher prompt. Its paired Gate50 showed that the
generic instruction to return an exact table did not preserve several concrete output-slot rules,
and that five tools needed one short correction example for errors observed in that gate.

This module is the single owner of those additions. It removes the overlapping output-shape
sentences from the frozen version40 template, inserts one consolidated output contract, and renders
only verifier-observed error corrections. Tool schemas and execution semantics remain elsewhere.
"""
from __future__ import annotations

import json
from copy import deepcopy

from atomic_version40 import PROMPT_TEMPLATE as VERSION40_PROMPT_TEMPLATE
from provider_adapter import VERSION40_RESPONSE_CONTRACT_PLACEHOLDER


OUTPUT_CONTRACT = (
    "OUTPUT CONTRACT\n"
    "Identify the requested output slots and their order before relational commitments, then keep "
    "them fixed unless the QUESTION or EXTERNAL KNOWLEDGE explicitly changes them. Immediately "
    "before answer_from_context, cite a table containing exactly those slots in exactly that order.\n"
    "- Preserve stored output representation. Keep separate source fields in separate columns "
    "unless formatting is explicitly requested. Do not concatenate names, replace an ID/code with "
    "a label/name, translate or case-normalize text, or round a number merely because another "
    "representation seems more natural.\n"
    "- Remove filter, join, count, and ranking helper columns. For ranked results, use "
    "return_columns; otherwise use project to create the exact terminal columns and order. "
    "inspect_rows and reasoning only observe or discuss a relation and cannot repair its shape."
)


ERROR_CORRECTION_ACTIONS: tuple[dict, ...] = (
    {
        "tool": "condition_filter",
        "error": "table membership used value_ref or an empty in_table handle name",
        "guidance": (
            "Use op=in with a non-empty resident table-handle name in in_table. value_ref is only "
            "for a scalar-producing step."
        ),
        "arguments": {
            "table": "orders",
            "conditions": {
                "column": "customer_id",
                "op": "in",
                "in_table": "eligible_customer_ids",
            },
            "return_columns": ["order_id"],
        },
    },
    {
        "tool": "condition_filter",
        "error": "scalar comparison cited a perception or multi-row step",
        "guidance": (
            "For a scalar comparison, value_ref must cite the step that produced a resident 1x1 "
            "or one-row metric table."
        ),
        "arguments": {
            "table": "orders",
            "conditions": {
                "column": "amount",
                "op": ">",
                "value_ref": "step_5",
            },
        },
    },
    {
        "tool": "extreme_value_select",
        "error": "order_by was empty or split DESC into a second list item",
        "guidance": (
            'order_by is a non-empty list; keep the direction in the same string as its column.'
        ),
        "arguments": {
            "table": "group_004",
            "order_by": ["cnt DESC"],
            "top_k": 3,
            "return_columns": ["word", "word_id"],
        },
    },
    {
        "tool": "group_aggregate",
        "error": "category_values/output_columns were supplied without columns layout",
        "guidance": (
            "Category-to-column reshape requires output_layout=columns, one group column, one "
            "aggregation, and ordered category_values. If output_columns is supplied, it must have "
            "the same length and order."
        ),
        "arguments": {
            "table": "eligible_people",
            "group_by": ["gender"],
            "aggregations": [
                {"op": "count", "column": "*", "as": "person_count"},
            ],
            "output_layout": "columns",
            "category_values": ["M", "F"],
            "output_columns": ["male_count", "female_count"],
        },
    },
    {
        "tool": "scalar_compute",
        "error": "operation used a SQL/function spelling outside the public enum",
        "guidance": (
            "Use the exact public operation and cite producing steps; date_diff_days operand order "
            "is start, then end."
        ),
        "arguments": {
            "operation": "date_diff_days",
            "operands": [
                {"value_ref": "step_4", "column": "start_date"},
                {"value_ref": "step_4", "column": "end_date"},
            ],
            "result_name": "duration_days",
        },
    },
    {
        "tool": "inspect_rows",
        "error": "limit exceeded 20 or offset pagination omitted deterministic ordering",
        "guidance": (
            "limit is 1..20. A positive offset requires order_by and requests the next deterministic "
            "page without creating a new table."
        ),
        "arguments": {
            "table": "transactions",
            "columns": ["id", "amount"],
            "order_by": ["id"],
            "limit": 20,
            "offset": 20,
        },
    },
)


def correction_action(action: dict) -> dict:
    """Return the exact model-visible action object for one correction record."""
    return {
        "tool": action["tool"],
        "arguments": deepcopy(action["arguments"]),
    }


def render_error_correction_examples() -> str:
    """Render concise, valid corrections for only the errors observed in version40 Gate50."""
    lines = [
        "ERROR-CORRECTION EXAMPLES",
        "Use the matching correction after structured tool feedback; replace visible handles, "
        "columns, steps, and values only.",
    ]
    for action in ERROR_CORRECTION_ACTIONS:
        lines.append(
            f"- {action['tool']} — error: {action['error']}. {action['guidance']}"
        )
        lines.append(
            json.dumps(
                correction_action(action),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines)


_VERSION40_TASK_OUTPUT = (
    "Answer the user question by deriving one grounded result table. The terminal evidence table "
    "must itself contain exactly the requested rows, columns, column order, and representation; "
    "reasoning text cannot repair its data or shape."
)
_VERSION41_TASK = "Answer the user question by deriving one grounded result table."
_VERSION40_RULE5 = (
    "5. Row observation never filters, projects, or creates a new handle. Use relational tools to "
    "change rows or columns, and remove helper columns before citing terminal evidence."
)
_VERSION41_RULE5 = (
    "5. Row observation never filters, projects, or creates a new handle. Use relational tools to "
    "change rows or columns."
)
_RESPONSE_MARKER = (
    "\n\nRESPONSE CONTRACT\n" + VERSION40_RESPONSE_CONTRACT_PLACEHOLDER
)


def build_prompt_template() -> str:
    """Build version41 by replacing version40's overlapping output clauses exactly once."""
    prompt = VERSION40_PROMPT_TEMPLATE
    for old, new in (
        (_VERSION40_TASK_OUTPUT, _VERSION41_TASK),
        (_VERSION40_RULE5, _VERSION41_RULE5),
    ):
        if prompt.count(old) != 1:
            raise ValueError("version40 prompt changed; version41 output consolidation is unsafe")
        prompt = prompt.replace(old, new, 1)
    if prompt.count(_RESPONSE_MARKER) != 1:
        raise ValueError("version40 response marker changed; cannot insert version41 prompt module")
    module = OUTPUT_CONTRACT + "\n\n" + render_error_correction_examples()
    return prompt.replace(_RESPONSE_MARKER, "\n\n" + module + _RESPONSE_MARKER, 1)


PROMPT_TEMPLATE = build_prompt_template()
