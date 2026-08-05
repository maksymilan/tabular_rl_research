from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.transition_grpo import TransitionGRPOTrainer


def _ranking_fixture(reduction: str):
    trainer = object.__new__(TransitionGRPOTrainer)
    trainer.rank_loss_coefficient = 0.5
    trainer.rank_beta = 0.1
    trainer.rank_score_scope = "conservative_legal"
    trainer.rank_score_reduction = reduction
    trainer.rank_update_scope = "conservative_legal"
    trainer._metrics = defaultdict(lambda: defaultdict(list))

    # Correct trajectory: two clean legal actions. Failed trajectory: one
    # legal exploration followed by one deterministic local error.
    transition_scores = torch.tensor([-1.0, -3.0, -100.0, -2.0])

    def sequence_logps(self, model, inputs):
        return transition_scores

    trainer._transition_sequence_logps = MethodType(sequence_logps, trainer)
    inputs = {
        "transition_trajectory_indices": torch.tensor([0, 0, 1, 1]),
        "transition_example_indices": torch.tensor([10, 10, 10, 10]),
        "transition_trajectory_correct": torch.tensor([True, True, False, False]),
        "transition_legal_success": torch.tensor([True, True, True, False]),
        "transition_local_penalty": torch.tensor([0.0, 0.0, 0.0, 0.08]),
        "rank_completion_mask": torch.ones((4, 1), dtype=torch.long),
    }
    return trainer, inputs


def test_conservative_score_scope_excludes_legal_failed_exploration() -> None:
    trainer, inputs = _ranking_fixture("sum_tokens")
    coefficients, _ = trainer._ranking_coefficients(None, inputs)

    # The -100 legal-exploration score is absent. Correct score=-4, failed
    # score=-2, so the measured gap is exactly -2 rather than +98.
    assert trainer._metrics["train"]["rank/score_gap"] == pytest.approx([-2.0])
    assert coefficients[2].item() == pytest.approx(0.0)
    assert coefficients[0].item() < 0
    assert coefficients[1].item() < 0
    assert coefficients[3].item() > 0


def test_action_mean_normalizes_each_selected_action_and_trajectory() -> None:
    trainer, inputs = _ranking_fixture("mean_action")
    coefficients, _ = trainer._ranking_coefficients(None, inputs)

    # mean(-1,-3) == -2 for the correct trajectory and the only explicitly
    # bad action is -2, so action-mean scoring has zero gap.
    assert trainer._metrics["train"]["rank/score_gap"] == pytest.approx([0.0])
    assert coefficients[0].item() == pytest.approx(coefficients[1].item())
    assert coefficients[0].item() == pytest.approx(-0.0125)
    assert coefficients[2].item() == pytest.approx(0.0)
    assert coefficients[3].item() == pytest.approx(0.025)
