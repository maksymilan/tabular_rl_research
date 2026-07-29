"""Role-separated model-visible prompts for the table-tool protocol.

The student runtime contract is the shared semantic authority used by SFT export, evaluation,
and RL.  A teacher receives that same contract plus generation-only guidance and examples.  The
teacher additions may improve data quality, but they must never add tools, arguments, state fields,
or execution semantics that are absent from the shared contract.
"""
from __future__ import annotations

import hashlib

from public_tool_contract import (
    ACTION_BLOCK_PUBLIC_TOOL_ARGUMENTS,
    ACTION_BLOCK_PUBLIC_TOOL_CONTRACTS,
    PUBLIC_TOOL_ARGUMENTS,
    PUBLIC_TOOL_CONTRACTS,
    render_action_grammar,
)

CANONICAL_ACTION_RULE = (
    '1. Each turn, output exactly: <think>brief reasoning</think> then one raw JSON object '
    '{"tool": "<name>", "arguments": {...}}. Nothing else; do not use tool_call tags.'
)

TEACHER_ONE_ACTION_RULE = (
    "ONE REQUEST = ONE ACTION. Emit exactly one non-empty <think> block and exactly one "
    'raw JSON object with only "tool" and "arguments". Immediately STOP after that JSON object: '
    "never emit a second <think>, a second action, a numbered plan of calls, or a complete "
    "multi-step solution in one response. The harness will execute only this one action and "
    "return a fresh state before you choose the next action. Your <think> block must be non-empty "
    "on every turn. Put the reason inside <think> tags and do not use tool_call tags."
)

TEACHER_SEMANTIC_DECISION_DISCIPLINE = (
    "SEMANTIC DECISION DISCIPLINE\n"
    "Treat the QUESTION and EXTERNAL KNOWLEDGE as the binding answer specification. Follow every "
    "explicit phrase-to-column, value, operator, aggregation, formula, identifier, or output-field "
    "mapping exactly. Do not replace it with a more natural label, field, COUNT DISTINCT, proxy, "
    "or formatted value. You may inspect the stored spelling of a literal, but never silently "
    "redefine the requested target.\n"
    "Before the first filter, join, aggregation, or ranking commitment, identify from visible "
    "evidence: (a) the requested answer entity or measurement unit, (b) what one row of the current "
    "input represents, (c) the conditions that define the eligible population, and (d) the exact "
    "requested output fields and their order. Preserve that population and grain unless a later "
    "observation from the harness proves the assumption wrong.\n"
    "Do not invent a selector or restriction merely to obtain one row or a plausible answer. In "
    "particular, do not add earliest, latest, current, active, first, top-1, mean, or same-year "
    "semantics unless the question, external knowledge, or observed schema establishes it. If "
    "several rows satisfy every stated condition, retain them unless the specification supplies a "
    "grounded disambiguator.\n"
    "Match aggregation semantics to the requested unit exactly. COUNT, COUNT DISTINCT, row count, "
    "and entity count are different. Numerators and denominators must use the same eligible "
    "population and grain unless the specification explicitly defines otherwise. Ranking before "
    "a required join may rank the wrong population; establish all answer-eligibility relations "
    "before aggregation or ranking.\n"
    "Treat zero rows, unexpected multiplicity, NULLs, impossible dates, or implausible arithmetic "
    "as evidence that an assumption needs inspection. Do not make an empty join nonempty with a "
    "left join, choose an arbitrary matching row, switch from an ID to a label, or accept malformed "
    "numeric coercion just because it yields an answer. Inspect the relevant schema, values, rows, "
    "or relationship, then revise only what the new evidence supports; normalize every arithmetic "
    "operand consistently when stored numeric text requires normalization.\n"
    "Immediately before answer_from_context, compare the evidence table to the question one slot "
    "at a time. It must contain exactly the requested rows, columns, column order, and representation: "
    "remove ranking/count/join helper columns, preserve separate source fields unless formatting is "
    "explicitly requested, and never rely on the reason text to repair the cited table."
)


def _tool_signature(
    tool: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> str:
    arguments = [*required, *(f"{name}?" for name in optional)]
    return f"{tool}({', '.join(arguments)})"


SHARED_TOOL_SPECS: dict[str, str] = {
    tool: f"{_tool_signature(tool, required, optional)} -> {PUBLIC_TOOL_CONTRACTS[tool].semantics}"
    for tool, (required, optional) in PUBLIC_TOOL_ARGUMENTS.items()
}

ACTION_BLOCK_TOOL_SPECS: dict[str, str] = {
    tool: (
        f"{_tool_signature(tool, required, optional)} -> "
        f"{ACTION_BLOCK_PUBLIC_TOOL_CONTRACTS[tool].semantics}"
    )
    for tool, (required, optional) in ACTION_BLOCK_PUBLIC_TOOL_ARGUMENTS.items()
}


STUDENT_CONTEXT_CONTRACT = (
    "The opening overview is a catalog of table names, row counts, and relations, not table "
    "schemas. EXTERNAL KNOWLEDGE, when present, is user-provided task context. CURRENT ENVIRONMENT "
    "STATE is the authoritative workspace: it contains resident control state, known schemas, "
    "inspected values, derived table handles, row reads, and usable producing step ids. A derived "
    "table handle exposes columns and row_count but not row values until read_subtable is called. "
    "Harness-authored relation derivation metadata states executed row/column semantics; it is not "
    "policy advice or an additional source of values."
)

STUDENT_RUNTIME_RULES = (
    "2. Use only names and arguments in TOOLS. One turn contains one action.\n"
    "3. Resolve schemas before using unknown columns. Use environment handles and exact logical "
    "columns; after joins, dotted identifiers are relation.column, while a bare downstream name is "
    "valid only when it resolves uniquely.\n"
    "4. A value_ref cites the step that produced the resident scalar or one-row metric table, not "
    "a perception or plan step. Use value_ref+column for a named metric in a one-row table.\n"
    "5. Relational operators preserve their declared population and grain. Per-aggregation where "
    "conditions share one input table; output_layout=columns is the category-row-to-column reshape.\n"
    "6. Terminal evidence is scored from the cited table only. Before answering, derive the exact "
    "requested rows, columns, and column order; observing rows or explaining them does not change "
    "the relation."
)

ROLLING_HISTORY_CONTRACT = (
    "\n\nROLLING LEGAL HISTORY\n"
    "The context may include a bounded transcript suffix of earlier harness-successful assistant "
    "actions paired with observations. Continue from it without assuming it is complete. CURRENT "
    "ENVIRONMENT STATE remains authoritative and LAST TOOL ERROR is the record of the latest "
    "rejected action. Rejected assistant text and unexecuted plan text are not factual evidence."
)


def build_student_system_prompt(*, include_action_grammar: bool = False) -> str:
    """Build a student prompt from the shared semantic and structural contract."""
    prompt = (
        "You are a relational table-tool agent. Answer the question by executing one typed tool "
        "action per turn.\n\n"
        "CONTEXT\n"
        f"{STUDENT_CONTEXT_CONTRACT}\n\n"
        "TOOLS\n"
        + "\n".join(SHARED_TOOL_SPECS.values())
        + "\n\nRULES\n"
        + CANONICAL_ACTION_RULE
        + "\n"
        + STUDENT_RUNTIME_RULES
    )
    if include_action_grammar:
        prompt += "\n\n" + render_action_grammar()
    return prompt


def add_teacher_guidance(
    student_prompt: str,
    *,
    tool_guidance: dict[str, str],
    call_cookbook: str,
) -> str:
    """Append generation-only help without changing the student's public action contract."""
    if set(tool_guidance) != set(SHARED_TOOL_SPECS):
        raise ValueError("teacher guidance and shared tool contract must cover the same tools")
    return (
        student_prompt
        + "\n\nTEACHER-ONLY DATA GENERATION GUIDANCE\n"
        "The following elaborations and examples are quality controls for producing causal "
        "demonstrations. They are not part of the student runtime prompt and do not add any public "
        "tool, argument, state field, or execution behavior.\n"
        + TEACHER_SEMANTIC_DECISION_DISCIPLINE
        + "\n\n"
        + "\n".join(tool_guidance.values())
        + "\n\n"
        + call_cookbook
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
