from __future__ import annotations

import pytest

from rl.scenarios.diagnostics.run_routed_coupled_reward_smoke import (
    normalized_loss_weights,
)


def row(trajectory_id: str, turn_index: int, category: str) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "turn_index": turn_index,
        "reward_category": category,
    }


def test_active_mean_preserves_equal_question_mass() -> None:
    first = [row("a", 0, "positive"), row("a", 1, "negative")]
    second = [row("b", 0, "positive")]
    weights = normalized_loss_weights([first, second], "active_mean")
    assert weights[("a", 0)] == pytest.approx(0.25)
    assert weights[("a", 1)] == pytest.approx(0.25)
    assert weights[("b", 0)] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_category_mean_preserves_equal_category_mass() -> None:
    first = [row("a", 0, "positive"), row("a", 1, "negative")]
    second = [row("b", 0, "positive")]
    weights = normalized_loss_weights([first, second], "category_mean")
    assert weights[("a", 0)] == pytest.approx(0.25)
    assert weights[("b", 0)] == pytest.approx(0.25)
    assert weights[("a", 1)] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_category_mean_accepts_explicit_category_mass() -> None:
    first = [row("a", 0, "key"), row("a", 1, "terminal")]
    second = [row("b", 0, "key"), row("b", 1, "negative")]
    weights = normalized_loss_weights(
        [first, second],
        "category_mean",
        {"key": 2.0, "terminal": 0.25, "negative": 1.0},
    )
    assert weights[("a", 0)] + weights[("b", 0)] == pytest.approx(2.0 / 3.25)
    assert weights[("a", 1)] == pytest.approx(0.25 / 3.25)
    assert weights[("b", 1)] == pytest.approx(1.0 / 3.25)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="active transitions"):
        normalized_loss_weights([[row("a", 0, "positive")], []], "active_mean")
