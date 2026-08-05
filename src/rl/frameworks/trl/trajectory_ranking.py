#!/usr/bin/env python3
"""Trajectory-level positive/negative pairing for ranking objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


RANK_SCORE_TOKENS = {"all", "tool_only"}
RANK_SCORE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}
RANK_SCORE_REDUCTIONS = {"sum_tokens", "mean_action"}
RANK_UPDATE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}


@dataclass(frozen=True)
class TrajectoryPair:
    positive_index: int
    negative_index: int


def rank_transition_selected(
    update_scope: str,
    *,
    trajectory_correct: bool,
    legal_success: bool,
    local_penalty: float,
) -> bool:
    """Return whether one turn receives the detached trajectory-rank coefficient.

    ``full_trajectory`` is the frozen historical behavior.  The conservative
    ablation rewards every legal, locally unpenalized action in a verified
    successful trajectory, penalizes only deterministic local errors in a
    failed trajectory, and leaves legal failed-trajectory exploration neutral.
    ``dense_outcome`` keeps every failed-trajectory turn and every clean legal
    successful-trajectory turn, matching the dense action-level process target.
    """
    if update_scope not in RANK_UPDATE_SCOPES:
        raise ValueError(f"unsupported rank update scope: {update_scope}")
    if update_scope == "full_trajectory":
        return True
    if update_scope == "dense_outcome":
        if trajectory_correct:
            return bool(legal_success and local_penalty <= 0.0)
        return True
    if trajectory_correct:
        return bool(legal_success and local_penalty <= 0.0)
    return bool(local_penalty > 0.0)


def build_trajectory_pairs(
    example_indices: Sequence[int],
    correct: Sequence[bool],
) -> list[TrajectoryPair]:
    """Build all correct×incorrect pairs within each question."""
    if len(example_indices) != len(correct):
        raise ValueError("trajectory metadata vectors must align")
    pairs = []
    for example_index in sorted(set(example_indices)):
        positives = [
            index
            for index, (current_example, is_correct) in enumerate(
                zip(example_indices, correct, strict=True)
            )
            if current_example == example_index and is_correct
        ]
        negatives = [
            index
            for index, (current_example, is_correct) in enumerate(
                zip(example_indices, correct, strict=True)
            )
            if current_example == example_index and not is_correct
        ]
        pairs.extend(
            TrajectoryPair(positive_index, negative_index)
            for positive_index in positives
            for negative_index in negatives
        )
    return pairs


def filter_pairs_with_score_support(
    pairs: Sequence[TrajectoryPair],
    trajectory_has_score_support: Sequence[bool],
) -> list[TrajectoryPair]:
    """Drop pairs whose positive or negative trajectory has no scored action."""
    support = list(trajectory_has_score_support)
    retained = []
    for pair in pairs:
        if pair.positive_index >= len(support) or pair.negative_index >= len(support):
            raise ValueError("trajectory pair index is outside score-support metadata")
        if support[pair.positive_index] and support[pair.negative_index]:
            retained.append(pair)
    return retained


def build_pairs_from_transition_metadata(
    trajectory_indices: Sequence[int],
    example_indices: Sequence[int],
    correct: Sequence[bool],
) -> tuple[list[int], list[TrajectoryPair]]:
    """Recover trajectory pairs after TRL has shuffled transition-aligned fields."""
    if not (
        len(trajectory_indices) == len(example_indices) == len(correct)
    ):
        raise ValueError("transition metadata vectors must align")
    trajectory_ids = sorted(set(int(value) for value in trajectory_indices))
    if trajectory_ids != list(range(len(trajectory_ids))):
        raise ValueError("trajectory indices must be contiguous from zero")
    metadata: dict[int, tuple[int, bool]] = {}
    for trajectory_id, example_index, is_correct in zip(
        trajectory_indices,
        example_indices,
        correct,
        strict=True,
    ):
        value = (int(example_index), bool(is_correct))
        previous = metadata.setdefault(int(trajectory_id), value)
        if previous != value:
            raise ValueError(
                "one trajectory has inconsistent example/correct metadata"
            )
    pairs = build_trajectory_pairs(
        [metadata[trajectory_id][0] for trajectory_id in trajectory_ids],
        [metadata[trajectory_id][1] for trajectory_id in trajectory_ids],
    )
    return trajectory_ids, pairs
