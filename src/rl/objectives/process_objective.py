#!/usr/bin/env python3
"""Tensor objective for step rewards with an explicit fixed-reference KL term."""
from __future__ import annotations

import math
from collections.abc import Sequence


def sampled_forward_kl(current_logp, reference_logp, *, max_value: float = 10.0):
    """Stable non-negative k3 estimator of ``KL(current || reference)``.

    Inputs may be scalars or aligned token-log-probability tensors.  Clamping is applied per token,
    before a caller sums a complete turn.  Applying k3 to a sequence-summed log-ratio instead would
    exponentiate the product of all token ratios and becomes unstable for long assistant turns.
    """
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    log_ratio = current_logp - reference_logp
    if hasattr(log_ratio, "clamp"):
        safe_log_ratio = log_ratio.float().clamp(min=-20.0, max=20.0)
        estimate = (-safe_log_ratio).exp() + safe_log_ratio - 1.0
        return estimate.clamp(min=0.0, max=float(max_value))
    safe_log_ratio = min(20.0, max(-20.0, float(log_ratio)))
    estimate = math.exp(-safe_log_ratio) + safe_log_ratio - 1.0
    return min(float(max_value), max(0.0, estimate))


def sampled_turn_forward_kl(current_token_logps, reference_token_logps):
    """Sum conditional token KL estimates to obtain one complete-turn action KL."""
    if getattr(current_token_logps, "shape", None) != getattr(reference_token_logps, "shape", None):
        raise ValueError("current and reference token log-probabilities must align")
    token_kls = sampled_forward_kl(current_token_logps, reference_token_logps)
    if hasattr(token_kls, "sum"):
        return token_kls.sum()
    return sum(token_kls)


def process_policy_loss(
    episode_step_logprobs: Sequence[Sequence],
    episode_step_rewards: Sequence[Sequence[float]],
    *,
    episode_step_kls: Sequence[Sequence] | None = None,
    episode_update_mask: Sequence[bool] | None = None,
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
    if episode_update_mask is not None and len(episode_update_mask) != len(episode_step_logprobs):
        raise ValueError("episode update mask must align")

    policy_terms = []
    kl_terms = []
    for episode_index, (logps, rewards) in enumerate(
        zip(episode_step_logprobs, episode_step_rewards, strict=True)
    ):
        if episode_update_mask is not None and not episode_update_mask[episode_index]:
            continue
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
    # M is the number of assistant turns that actually enter the process update, not episodes.
    return total / len(policy_terms)
