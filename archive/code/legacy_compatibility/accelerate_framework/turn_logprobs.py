#!/usr/bin/env python3
"""Token-exact log probabilities for complete assistant-turn actions.

The process objective is defined over assistant turns, not over an appended multi-turn transcript.
Each training item therefore contains the exact prefix token IDs seen at rollout time and the exact
assistant response token IDs sampled from that prefix.  This is the transition-level aggregation
used by the active backend; retaining token IDs avoids chat-template re-tokenization drift.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


TurnTokens = tuple[list[int], list[int]]


def turn_padding_key(turn: TurnTokens) -> tuple[int, int]:
    """Order turns by the two dimensions that drive padding in ``build_turn_batch``."""
    prompt_ids, response_ids = turn
    return len(prompt_ids) + max(0, len(response_ids) - 1), len(response_ids)


def build_turn_batch(
    tokenizer: Any,
    turns: Sequence[TurnTokens],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad exact rollout prefixes and align next-token response targets."""
    if not turns:
        raise ValueError("at least one assistant turn is required")
    if any(not response_ids for _, response_ids in turns):
        raise ValueError("every assistant turn must contain at least one response token")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be defined")

    max_response_len = max(len(response_ids) for _, response_ids in turns)
    sequences = [prompt_ids + response_ids[:-1] for prompt_ids, response_ids in turns]
    max_sequence_len = max(len(sequence) for sequence in sequences)

    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    for sequence, (_, response_ids) in zip(sequences, turns, strict=True):
        sequence_pad = max_sequence_len - len(sequence)
        response_pad = max_response_len - len(response_ids)
        input_rows.append([tokenizer.pad_token_id] * sequence_pad + sequence)
        attention_rows.append([0] * sequence_pad + [1] * len(sequence))
        target_rows.append([-100] * response_pad + response_ids)

    return (
        torch.tensor(input_rows, device=device, dtype=torch.long),
        torch.tensor(attention_rows, device=device, dtype=torch.long),
        torch.tensor(target_rows, device=device, dtype=torch.long),
    )


def model_logits_for_response(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """Return only response-prediction logits when the model supports that optimization."""
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        ).logits
    except TypeError:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    return logits[:, -logits_to_keep:, :]


def response_token_logprobs_batched(
    model: Any,
    tokenizer: Any,
    turns: Sequence[TurnTokens],
    device: torch.device,
) -> list[torch.Tensor]:
    """Return one differentiable token-log-probability vector per assistant turn."""
    input_ids, attention_mask, targets = build_turn_batch(tokenizer, turns, device)
    max_response_len = targets.shape[1]
    logits = model_logits_for_response(
        model,
        input_ids,
        attention_mask,
        max_response_len,
    )
    token_losses = F.cross_entropy(
        logits.float().transpose(1, 2),
        targets,
        ignore_index=-100,
        reduction="none",
    )
    target_mask = targets.ne(-100)
    return [
        -token_losses[index][target_mask[index]]
        for index in range(len(turns))
    ]


def response_logprobs_batched(
    model: Any,
    tokenizer: Any,
    turns: Sequence[TurnTokens],
    device: torch.device,
) -> list[torch.Tensor]:
    """Return ``log pi(a_t | s_t)`` as the sum over every token in each complete turn."""
    return [
        token_logprobs.sum()
        for token_logprobs in response_token_logprobs_batched(
            model,
            tokenizer,
            turns,
            device,
        )
    ]
