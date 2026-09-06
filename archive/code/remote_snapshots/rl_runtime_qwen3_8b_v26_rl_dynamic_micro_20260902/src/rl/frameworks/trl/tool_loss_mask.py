#!/usr/bin/env python3
"""Exact response-token loss masks for the think + raw-JSON action carrier."""
from __future__ import annotations

from typing import Sequence


class ToolMaskUnavailable(ValueError):
    """The sampled response contains no valid raw-JSON tool suffix to train."""


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def tool_token_loss_mask(tokenizer, response_ids: Sequence[int]) -> tuple[int, ...]:
    """Mask reasoning tokens while retaining the direct raw-JSON action suffix.

    Prefix decoding is deliberate: the response IDs are the exact vLLM sample,
    and re-tokenizing decoded text could move a BPE boundary. A token is active
    when decoding through it reveals at least one character at or after the
    first raw-JSON ``{`` following ``</think>``.
    """
    if not response_ids:
        raise ToolMaskUnavailable("tool-only masking requires a non-empty response")
    text = _decode(tokenizer, response_ids)
    think_end = text.find("</think>")
    if think_end < 0:
        raise ToolMaskUnavailable(
            "tool-only masking requires a closing </think> marker"
        )
    json_start = text.find("{", think_end + len("</think>"))
    if json_start < 0:
        raise ToolMaskUnavailable(
            "tool-only masking requires a raw JSON action object"
        )
    if text[think_end + len("</think>"):json_start].strip():
        raise ToolMaskUnavailable(
            "non-whitespace content appears between </think> and JSON"
        )

    mask = []
    previous_length = 0
    for end in range(1, len(response_ids) + 1):
        prefix_length = len(_decode(tokenizer, response_ids[:end]))
        if prefix_length < previous_length:
            raise ValueError("tokenizer prefix decoding is not monotonic")
        mask.append(int(prefix_length > json_start))
        previous_length = prefix_length
    if not any(mask):
        raise ToolMaskUnavailable("tool-only masking produced no trainable tokens")
    return tuple(mask)
