"""Small, framework-independent IQL building blocks.

The project uses this module for an offline feasibility diagnostic first. It
does not define a reward model and it does not change the actor objective. The
reward passed to these functions must already come from the Atomic Harness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light audit environments.
    torch = None


@dataclass(frozen=True)
class IQLConfig:
    """Registered defaults for the diagnostic, not the formal RL arm."""

    gamma: float = 0.99
    expectile_tau: float = 0.7
    inverse_temperature: float = 3.0
    max_advantage_weight: float = 20.0

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.5 < self.expectile_tau < 1.0:
            raise ValueError("expectile_tau must be in (0.5, 1)")
        if self.inverse_temperature <= 0.0:
            raise ValueError("inverse_temperature must be positive")
        if self.max_advantage_weight <= 0.0:
            raise ValueError("max_advantage_weight must be positive")


def expectile_weight(delta: float, tau: float) -> float:
    """Return the scalar expectile weight for ``delta = target - prediction``."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    return tau if delta > 0.0 else 1.0 - tau


def expectile_loss_value(prediction: float, target: float, tau: float) -> float:
    """Dependency-free scalar expectile loss used by tests and audits."""
    delta = float(target) - float(prediction)
    return expectile_weight(delta, tau) * delta * delta


def expectile_loss(prediction: Any, target: Any, tau: float) -> Any:
    """Torch mean expectile loss for a value head."""
    if torch is None:
        raise RuntimeError("expectile_loss requires PyTorch")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    delta = target - prediction
    weights = torch.where(
        delta > 0,
        torch.as_tensor(tau, device=delta.device),
        torch.as_tensor(1.0 - tau, device=delta.device),
    )
    return (weights * delta.square()).mean()


def td_target(reward: float, next_value: float, done: bool, gamma: float) -> float:
    """One-step SMDP target; terminal transitions do not bootstrap."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    return float(reward) if bool(done) else float(reward) + gamma * float(next_value)


def advantage_weight(
    q_value: float,
    value: float,
    *,
    inverse_temperature: float,
    max_weight: float,
) -> float:
    """Bounded IQL actor weight ``exp((Q-V)/beta)``."""
    if inverse_temperature <= 0.0:
        raise ValueError("inverse_temperature must be positive")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    exponent = min(
        (float(q_value) - float(value)) / inverse_temperature,
        math.log(max_weight),
    )
    return math.exp(exponent)


def discounted_terminal_returns(
    rewards: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
) -> list[float]:
    """Compute episode-local return-to-go for sparse terminal rewards."""
    if len(rewards) != len(dones):
        raise ValueError("rewards and dones must have equal length")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if bool(dones[index]):
            running = float(rewards[index])
        else:
            running = float(rewards[index]) + gamma * running
        result[index] = running
    return result
