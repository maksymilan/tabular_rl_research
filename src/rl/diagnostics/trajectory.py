"""Reusable trajectory semantics shared by offline RL diagnostics.

The online trainer and several historical probes classify the same transition
features into the same dense-outcome buckets.  Keeping that precedence in one
place prevents a new diagnostic from silently assigning a different meaning to
an observation or local error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


POSITIVE_DENSE_CATEGORIES = frozenset(
    {
        "correct_key_evidence",
        "correct_key_backslice",
        "correct_terminal",
    }
)
NEGATIVE_DENSE_CATEGORIES = frozenset({"severe_local_error"})


def dense_reward_category(
    transition: Mapping[str, Any], feature_row: Mapping[str, Any]
) -> str:
    """Return the canonical dense-outcome category for one transition.

    The precedence is intentionally explicit: a severe local Harness event is
    an error even when the trajectory eventually reaches a correct terminal;
    otherwise correct trajectories receive the most specific evidence bucket.
    """

    features = feature_row.get("features") or {}
    if bool(features.get("dense_severe_local_bad_event")):
        return "severe_local_error"
    if bool(transition.get("trajectory_correct")):
        if bool(features.get("dense_operator_backslice_bonus")):
            return "correct_key_backslice"
        if bool(features.get("dense_observation_support_bonus")):
            return "correct_key_evidence"
        if bool(features.get("is_terminal")):
            return "correct_terminal"
        return "correct_other_clean"
    return "incorrect_other_clean"


def is_positive_dense_category(category: str) -> bool:
    """Whether a category contributes the positive routed branch."""

    return category in POSITIVE_DENSE_CATEGORIES


def is_negative_dense_category(category: str) -> bool:
    """Whether a category contributes the severe local-error branch."""

    return category in NEGATIVE_DENSE_CATEGORIES


__all__ = [
    "NEGATIVE_DENSE_CATEGORIES",
    "POSITIVE_DENSE_CATEGORIES",
    "dense_reward_category",
    "is_negative_dense_category",
    "is_positive_dense_category",
]
