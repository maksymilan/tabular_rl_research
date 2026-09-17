from __future__ import annotations

import pytest

from rl.frameworks.trl.rollout import RolloutSettings, adaptive_extension_counts


def test_zero_signal_groups_are_extended_by_one_step() -> None:
    targets = adaptive_extension_counts(
        outcomes={0: (8, 0), 1: (8, 8), 2: (8, 3), 3: (8, 5)},
        initial_group_size=8,
        max_group_size=32,
    )
    assert targets == {0: 8, 1: 8}


def test_mixed_groups_are_never_extended() -> None:
    targets = adaptive_extension_counts(
        outcomes={0: (16, 7), 1: (16, 9), 2: (16, 0)},
        initial_group_size=8,
        max_group_size=32,
    )
    assert targets == {2: 8}


def test_extension_stops_at_the_cap() -> None:
    targets = adaptive_extension_counts(
        outcomes={0: (32, 0), 1: (24, 0)},
        initial_group_size=8,
        max_group_size=32,
    )
    assert targets == {1: 8}
    assert adaptive_extension_counts(
        outcomes={0: (32, 32)},
        initial_group_size=8,
        max_group_size=32,
    ) == {}


def test_incomplete_groups_are_left_alone() -> None:
    """A group smaller than the requested K is incomplete evidence, not signal."""
    targets = adaptive_extension_counts(
        outcomes={0: (3, 0)},
        initial_group_size=8,
        max_group_size=32,
    )
    assert targets == {}


def test_invalid_bounds_fail_closed() -> None:
    with pytest.raises(ValueError):
        adaptive_extension_counts(
            outcomes={}, initial_group_size=1, max_group_size=32
        )
    with pytest.raises(ValueError):
        adaptive_extension_counts(
            outcomes={}, initial_group_size=8, max_group_size=4
        )


def test_settings_validate_adaptive_cap() -> None:
    RolloutSettings(adaptive_group_size_max=32).validate()
    RolloutSettings(adaptive_group_size_max=None).validate()
    with pytest.raises(ValueError):
        RolloutSettings(adaptive_group_size_max=1).validate()
