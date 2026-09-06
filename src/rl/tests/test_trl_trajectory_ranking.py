from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.trajectory_ranking import (
    TrajectoryPair,
    build_pairs_from_transition_metadata,
    build_trajectory_pairs,
    filter_pairs_with_score_support,
    rank_transition_selected,
)


def test_pairs_are_only_built_within_the_same_problem() -> None:
    pairs = build_trajectory_pairs(
        [10, 10, 10, 20, 20],
        [True, False, False, True, True],
    )
    assert pairs == [
        TrajectoryPair(0, 1),
        TrajectoryPair(0, 2),
    ]


def test_homogeneous_groups_produce_no_ranking_pairs() -> None:
    assert build_trajectory_pairs([1, 1, 2, 2], [True, True, False, False]) == []


def test_transition_aligned_metadata_survives_arbitrary_shuffle() -> None:
    trajectory_ids, pairs = build_pairs_from_transition_metadata(
        [2, 0, 1, 2, 0, 1],
        [20, 10, 10, 20, 10, 10],
        [True, True, False, True, True, False],
    )
    assert trajectory_ids == [0, 1, 2]
    assert pairs == [TrajectoryPair(0, 1)]


def test_transition_metadata_rejects_inconsistent_trajectory_labels() -> None:
    try:
        build_pairs_from_transition_metadata(
            [0, 0],
            [10, 20],
            [True, True],
        )
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("expected inconsistent trajectory metadata to fail")


def test_full_trajectory_scope_selects_every_turn() -> None:
    assert rank_transition_selected(
        "full_trajectory",
        trajectory_correct=False,
        legal_success=True,
        local_penalty=0.0,
    )


def test_conservative_scope_rewards_only_clean_legal_success_turns() -> None:
    assert rank_transition_selected(
        "conservative_legal",
        trajectory_correct=True,
        legal_success=True,
        local_penalty=0.0,
    )
    assert not rank_transition_selected(
        "conservative_legal",
        trajectory_correct=True,
        legal_success=False,
        local_penalty=0.08,
    )
    assert not rank_transition_selected(
        "conservative_legal",
        trajectory_correct=True,
        legal_success=True,
        local_penalty=0.03,
    )


def test_conservative_scope_penalizes_only_explicit_failed_turn_errors() -> None:
    assert rank_transition_selected(
        "conservative_legal",
        trajectory_correct=False,
        legal_success=False,
        local_penalty=0.08,
    )
    assert not rank_transition_selected(
        "conservative_legal",
        trajectory_correct=False,
        legal_success=True,
        local_penalty=0.0,
    )


def test_pairs_without_conservative_score_support_are_dropped() -> None:
    pairs = [TrajectoryPair(0, 1), TrajectoryPair(0, 2)]
    assert filter_pairs_with_score_support(pairs, [True, False, True]) == [
        TrajectoryPair(0, 2)
    ]


def test_dense_outcome_selects_clean_positive_and_every_negative_turn() -> None:
    assert rank_transition_selected(
        "dense_outcome",
        trajectory_correct=True,
        legal_success=True,
        local_penalty=0.0,
    )
    assert not rank_transition_selected(
        "dense_outcome",
        trajectory_correct=True,
        legal_success=False,
        local_penalty=2.0,
    )
    assert rank_transition_selected(
        "dense_outcome",
        trajectory_correct=False,
        legal_success=True,
        local_penalty=0.0,
    )
    assert rank_transition_selected(
        "dense_outcome",
        trajectory_correct=False,
        legal_success=False,
        local_penalty=2.0,
    )
