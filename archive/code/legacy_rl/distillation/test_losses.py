from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.rl.distillation.losses import (  # noqa: E402
    branch_dpo_loss,
    branch_logprob_from_turns,
    masked_mopd_loss,
    weighted_logprob,
)


def test_mopd_masks_context_clips_advantage_and_only_updates_current():
    current = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    old = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
    teacher = torch.tensor([9.0, -3.0, -0.5], requires_grad=True)
    weights = torch.tensor([0.0, 1.0, 1.0])
    loss = masked_mopd_loss(current, old, teacher, weights, advantage_clip=1.0)
    # Active advantages are [-1, +0.5].
    expected = -((-1.0 * -2.0) + (0.5 * -3.0)) / 2.0
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert current.grad.tolist() == pytest.approx([0.0, 0.5, -0.25])
    assert old.grad is None
    assert teacher.grad is None


def test_mopd_rejects_zero_policy_weight():
    with pytest.raises(ValueError, match="active policy token"):
        masked_mopd_loss([0.0], [0.0], [0.0], [0.0])


def test_branch_dpo_rewards_larger_reference_adjusted_margin():
    chosen = torch.tensor(-1.0, requires_grad=True)
    rejected = torch.tensor(-2.0, requires_grad=True)
    loss = branch_dpo_loss(chosen, rejected, -1.5, -1.5, beta=0.2)
    assert loss.item() == pytest.approx(-math.log(1.0 / (1.0 + math.exp(-0.2))))
    loss.backward()
    assert chosen.grad.item() < 0.0
    assert rejected.grad.item() > 0.0


def test_multi_turn_branch_score_uses_only_policy_tokens():
    score = branch_logprob_from_turns(
        [torch.tensor([-10.0, -1.0]), torch.tensor([-2.0])],
        [torch.tensor([0.0, 1.0]), torch.tensor([1.0])],
    )
    assert score.item() == pytest.approx(-1.5)
    assert weighted_logprob(torch.tensor([-1.0]), torch.tensor([1.0])).item() == -1.0
