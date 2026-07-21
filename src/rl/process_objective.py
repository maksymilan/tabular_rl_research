#!/usr/bin/env python3
"""Tensor objective for step rewards with an explicit fixed-reference KL term."""
from __future__ import annotations

from collections.abc import Sequence


def process_policy_loss(
    episode_step_logprobs: Sequence[Sequence],
    episode_step_rewards: Sequence[Sequence[float]],
    *,
    episode_step_kls: Sequence[Sequence] | None = None,
    beta: float = 0.0,
):
    """Return ``-(1/M) sum_i,t r_it log pi + beta/M sum_i,t KL_it``.

    The caller must compute ``KL_it`` against a frozen SFT-2 reference policy. Keeping that
    requirement explicit prevents accidentally treating the current rollout policy or base model
    as ``pi_SFT-2``. Tensor type/device are inherited from the provided log-probabilities.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not episode_step_logprobs:
        raise ValueError("at least one episode is required")
    if len(episode_step_logprobs) != len(episode_step_rewards):
        raise ValueError("episode log-probability and reward groups must align")
    if beta > 0 and episode_step_kls is None:
        raise ValueError("a positive beta requires KL values from a frozen SFT-2 reference")
    if episode_step_kls is not None and len(episode_step_kls) != len(episode_step_logprobs):
        raise ValueError("episode KL groups must align")

    policy_terms = []
    kl_terms = []
    for episode_index, (logps, rewards) in enumerate(
        zip(episode_step_logprobs, episode_step_rewards, strict=True)
    ):
        if len(logps) != len(rewards):
            raise ValueError(f"episode {episode_index} step log-probabilities and rewards must align")
        policy_terms.extend(-float(reward) * logp for logp, reward in zip(logps, rewards, strict=True))
        if episode_step_kls is not None:
            kls = episode_step_kls[episode_index]
            if len(kls) != len(logps):
                raise ValueError(f"episode {episode_index} KL values must align with actions")
            kl_terms.extend(kls)
    if not policy_terms:
        raise ValueError("at least one action is required")

    total = sum(policy_terms)
    if kl_terms:
        total = total + float(beta) * sum(kl_terms)
    return total / len(episode_step_logprobs)

