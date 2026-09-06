"""Prompt-only surface for version42's explicit terminal-column diagnostic."""
from __future__ import annotations

from atomic_version41 import PROMPT_TEMPLATE as VERSION41_PROMPT_TEMPLATE
from atomic_version41 import TOOL_SPECS as VERSION41_TOOL_SPECS
from atomic_version41_prompt import OUTPUT_CONTRACT as VERSION41_OUTPUT_CONTRACT


VERSION42_ANSWER_SPEC = (
    "answer_from_context(evidence, reason?) -> terminal. evidence is exactly "
    '{"table":handle,"columns":[exact_column,...]}. The harness deterministically projects those '
    "existing grounded columns in the listed order and scores that projection. Use at least one "
    "column; this call cannot compute, rename, concatenate, aggregate, deduplicate, or invent values."
)

TOOL_SPECS = dict(VERSION41_TOOL_SPECS)
TOOL_SPECS["answer_from_context"] = VERSION42_ANSWER_SPEC

OUTPUT_CONTRACT = (
    "OUTPUT CONTRACT\n"
    "Identify the requested output slots and their order before relational commitments, then keep "
    "them fixed unless the QUESTION or EXTERNAL KNOWLEDGE explicitly changes them.\n"
    "- In answer_from_context, evidence.columns must list exactly those existing grounded columns "
    "in the requested order. Do not include filter, join, count, or ranking helper columns. The "
    "harness performs only this final column selection; reasoning text cannot change it.\n"
    "- Preserve stored output representation. Keep separate source fields in separate columns "
    "unless formatting is explicitly requested. Do not concatenate names, replace an ID/code with "
    "a label/name, translate or case-normalize text, or round a number merely because another "
    "representation seems more natural.\n"
    "- If the answer requires computation, aliasing, concatenation, aggregation, or distinct rows, "
    "derive that relation before the terminal call. inspect_rows only observes rows."
)

_VERSION41_ANSWER_EXAMPLE = (
    '{"tool":"answer_from_context","arguments":{"evidence":{"table":"project_003"},'
    '"reason":"The cited table has the requested result shape."}}'
)
_VERSION42_ANSWER_EXAMPLE = (
    '{"tool":"answer_from_context","arguments":{"evidence":{"table":"top_002",'
    '"columns":["label"]},"reason":"label is the only requested output slot."}}'
)


def build_prompt_template() -> str:
    """Replace only version41's terminal contract, example, and consolidated output module."""
    prompt = VERSION41_PROMPT_TEMPLATE
    replacements = (
        (
            VERSION41_TOOL_SPECS["answer_from_context"],
            VERSION42_ANSWER_SPEC,
        ),
        (VERSION41_OUTPUT_CONTRACT, OUTPUT_CONTRACT),
        (_VERSION41_ANSWER_EXAMPLE, _VERSION42_ANSWER_EXAMPLE),
    )
    for old, new in replacements:
        if prompt.count(old) != 1:
            raise ValueError(
                "version41 prompt changed; version42 terminal-column replacement is unsafe"
            )
        prompt = prompt.replace(old, new, 1)
    return prompt


PROMPT_TEMPLATE = build_prompt_template()
