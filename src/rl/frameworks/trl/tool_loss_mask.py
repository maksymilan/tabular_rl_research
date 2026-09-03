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


def _prefix_decoded_lengths(tokenizer, response_ids: Sequence[int]) -> list[int]:
    """Return exact per-prefix decoded lengths with a fast-tokenizer batch path.

    The original implementation intentionally decodes every prefix so BPE
    boundaries are never reconstructed by hand.  Fast Hugging Face tokenizers
    can perform the same individual decodes in bounded batches, which removes
    thousands of Python↔Rust calls for a long sampled response while retaining
    the original fallback for slow/custom tokenizers.
    """

    batch_decode = getattr(tokenizer, "batch_decode", None)
    if not getattr(tokenizer, "is_fast", False) or not callable(batch_decode):
        return [
            len(_decode(tokenizer, response_ids[:end]))
            for end in range(1, len(response_ids) + 1)
        ]

    lengths: list[int] = []
    chunk_size = 256
    for chunk_start in range(0, len(response_ids), chunk_size):
        chunk_end = min(len(response_ids), chunk_start + chunk_size)
        prefixes = [
            list(response_ids[:end])
            for end in range(chunk_start + 1, chunk_end + 1)
        ]
        try:
            decoded = batch_decode(
                prefixes,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except (AttributeError, TypeError):
            # A tokenizer may advertise ``is_fast`` while exposing an older or
            # customized batch API.  Preserve the exact legacy behavior.
            return [
                len(_decode(tokenizer, response_ids[:end]))
                for end in range(1, len(response_ids) + 1)
            ]
        if len(decoded) != len(prefixes):
            return [
                len(_decode(tokenizer, response_ids[:end]))
                for end in range(1, len(response_ids) + 1)
            ]
        lengths.extend(len(text) for text in decoded)
    return lengths


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
    for prefix_length in _prefix_decoded_lengths(tokenizer, response_ids):
        if prefix_length < previous_length:
            raise ValueError("tokenizer prefix decoding is not monotonic")
        mask.append(int(prefix_length > json_start))
        previous_length = prefix_length
    if not any(mask):
        raise ToolMaskUnavailable("tool-only masking produced no trainable tokens")
    return tuple(mask)
