"""Prompt-only surface for version43's unique-bare terminal-column diagnostic."""
from __future__ import annotations

from atomic_version42_prompt import PROMPT_TEMPLATE as VERSION42_PROMPT_TEMPLATE
from atomic_version42_prompt import TOOL_SPECS as VERSION42_TOOL_SPECS


VERSION43_ANSWER_SPEC = (
    "answer_from_context(evidence, reason?) -> terminal. evidence is exactly "
    '{"table":handle,"columns":[column,...]}. Each column is either an exact existing logical '
    "column or a bare name that matches exactly one dotted logical column by suffix. The harness "
    "deterministically resolves and projects those grounded columns in the listed order. Use at "
    "least one column; this call cannot compute, rename, concatenate, aggregate, deduplicate, or "
    "invent values."
)

TOOL_SPECS = dict(VERSION42_TOOL_SPECS)
TOOL_SPECS["answer_from_context"] = VERSION43_ANSWER_SPEC


def build_prompt_template() -> str:
    """Replace only version42's terminal column-name resolution sentence."""
    prompt = VERSION42_PROMPT_TEMPLATE
    old = VERSION42_TOOL_SPECS["answer_from_context"]
    if prompt.count(old) != 1:
        raise ValueError(
            "version42 prompt changed; version43 terminal-column replacement is unsafe"
        )
    return prompt.replace(old, VERSION43_ANSWER_SPEC, 1)


PROMPT_TEMPLATE = build_prompt_template()
