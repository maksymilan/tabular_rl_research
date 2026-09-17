from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.mechanism import RLMechanism
from rl.frameworks.trl.transition_batch import TransitionUpdate


def _update(advantage: float, trajectory_id: str = "t") -> TransitionUpdate:
    return TransitionUpdate(
        prompt_ids=(1,),
        response_ids=(2,),
        sampling_logprobs=(0.0,),
        advantage=advantage,
        trajectory_id=trajectory_id,
        turn_index=0,
        example_index=0,
        trajectory_correct=advantage > 0,
    )


def test_mechanism_from_legacy_args_and_reduction():
    mechanism = RLMechanism.from_legacy_args(
        reward_mode="result-only",
        policy_reduction="transition_mean",
        credit_assignment="trajectory",
        error_penalty=1.0,
    )
    assert mechanism.reward_mode == "result-only"
    assert mechanism.effective_advantages(
        [_update(1.0), _update(-1.0, "u")],
        transition_count=2,
        trajectory_count=2,
    ) == [1.0, -1.0]


def test_trajectory_credit_is_a_noop_and_does_not_mutate_updates():
    mechanism = RLMechanism(credit_assignment="trajectory")
    updates = [_update(1.0)]
    result, audit = mechanism.apply_credit([], updates)
    assert result == updates
    assert result is not updates
    assert audit is None


def test_advantage_magnitude_cap_is_symmetric_after_reduction():
    mechanism = RLMechanism(credit_assignment="trajectory", advantage_magnitude_cap=1.0)
    assert mechanism.effective_advantages(
        [_update(3.0), _update(-2.0, "u"), _update(0.25, "v")],
        transition_count=3,
        trajectory_count=3,
    ) == [1.0, -1.0, 0.25]
