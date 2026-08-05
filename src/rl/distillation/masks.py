"""Exact response-token weights for dense teacher scoring.

Only model-authored response tokens can receive weight.  Prompt, schema,
environment observations, and padding never enter this function and therefore
cannot accidentally become policy targets.
"""
from __future__ import annotations

from typing import Sequence


class PolicyTokenMaskUnavailable(ValueError):
    """The response does not satisfy the think + direct-JSON carrier."""


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def policy_token_weights(
    tokenizer,
    response_ids: Sequence[int],
    *,
    think_weight: float = 0.0,
    action_weight: float = 1.0,
) -> tuple[float, ...]:
    """Return weights aligned to the exact sampled response IDs.

    The JSON object includes the action name, arguments, and terminal evidence,
    so all of those tokens receive ``action_weight``.  ``<think>`` text is
    configurable but disabled in the first experiment.  Prefix decoding is
    used instead of re-tokenization so BPE boundaries cannot move.
    """
    if not response_ids:
        raise PolicyTokenMaskUnavailable("policy masking requires a response")
    if think_weight < 0.0 or action_weight <= 0.0:
        raise ValueError("think_weight must be non-negative and action_weight positive")

    text = _decode(tokenizer, response_ids)
    think_open = text.find("<think>")
    think_end = text.find("</think>")
    if think_open < 0 or think_end < think_open:
        raise PolicyTokenMaskUnavailable(
            "policy masking requires one visible <think>...</think> block"
        )
    after_think = think_end + len("</think>")
    json_start = text.find("{", after_think)
    if json_start < 0:
        raise PolicyTokenMaskUnavailable(
            "policy masking requires a direct raw JSON action object"
        )
    if text[after_think:json_start].strip():
        raise PolicyTokenMaskUnavailable(
            "non-whitespace content appears between </think> and JSON"
        )

    weights: list[float] = []
    previous_length = 0
    for end in range(1, len(response_ids) + 1):
        prefix_length = len(_decode(tokenizer, response_ids[:end]))
        if prefix_length < previous_length:
            raise ValueError("tokenizer prefix decoding is not monotonic")
        contributed_characters = prefix_length > previous_length
        if contributed_characters and prefix_length > json_start:
            weight = action_weight
        elif contributed_characters and prefix_length > think_open:
            weight = think_weight
        else:
            weight = 0.0
        weights.append(float(weight))
        previous_length = prefix_length

    if not any(weight > 0.0 for weight in weights):
        raise PolicyTokenMaskUnavailable("policy masking produced no active tokens")
    return tuple(weights)
