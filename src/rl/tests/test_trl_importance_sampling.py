from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from rl.frameworks.trl.transition_grpo import TransitionGRPOTrainer


def _trainer(mode: str = "token_truncate", cap: float = 3.0):
    trainer = object.__new__(TransitionGRPOTrainer)
    trainer.vllm_importance_sampling_mode = mode
    trainer.vllm_importance_sampling_cap = cap
    return trainer


def test_token_importance_diagnostics_exclude_padding_and_report_cap() -> None:
    trainer = _trainer(cap=3.0)
    old = torch.tensor([[math.log(2.0), math.log(4.0), 99.0]])
    sampled = torch.zeros_like(old)
    mask = torch.tensor([[1, 1, 0]])
    ratio = trainer._importance_sampling_ratio(old, sampled, mask)
    result = trainer._importance_sampling_diagnostics(
        old, sampled, mask, ratio
    )
    assert result["applied_ratio_min"] == pytest.approx(2.0)
    assert result["applied_ratio_mean"] == pytest.approx(2.5)
    assert result["applied_ratio_max"] == pytest.approx(3.0)
    assert result["cap_exceeded_fraction"] == pytest.approx(0.5)
    assert result["log_ratio_abs_max"] == pytest.approx(math.log(4.0))


def test_sequence_importance_diagnostics_use_one_ratio_per_trajectory() -> None:
    trainer = _trainer(mode="sequence_truncate", cap=10.0)
    old = torch.tensor([[math.log(2.0), math.log(3.0)], [7.0, 7.0]])
    sampled = torch.zeros_like(old)
    mask = torch.tensor([[1, 1], [0, 0]])
    ratio = trainer._importance_sampling_ratio(old, sampled, mask)
    result = trainer._importance_sampling_diagnostics(
        old, sampled, mask, ratio
    )
    assert result["applied_ratio_min"] == pytest.approx(6.0)
    assert result["applied_ratio_mean"] == pytest.approx(6.0)
    assert result["applied_ratio_max"] == pytest.approx(6.0)
    assert result["cap_exceeded_fraction"] == 0.0
