#!/usr/bin/env python3
"""Pure transition accounting shared by the TRL rollout and trainer adapters."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except ImportError:  # Lightweight analysis/test environments may omit Torch.
    torch = None


@dataclass(frozen=True)
class PolicyTurn:
    """One authored assistant action under its exact rollout prefix."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]

    def validate(self) -> None:
        if not self.prompt_ids:
            raise ValueError("a policy turn must retain a non-empty rollout prompt")
        if not self.response_ids:
            raise ValueError("a policy turn must retain a non-empty assistant response")
        if len(self.response_ids) != len(self.sampling_logprobs):
            raise ValueError(
                "response ids and sampling logprobs must align exactly: "
                f"{len(self.response_ids)} != {len(self.sampling_logprobs)}"
            )
        if not all(math.isfinite(value) for value in self.sampling_logprobs):
            raise ValueError("sampling logprobs must be finite")


@dataclass
class PolicyEpisode:
    """A scored harness episode plus policy-distribution evidence for every turn."""

    sample: Any
    policy_turns: list[PolicyTurn]

    def validate(self) -> None:
        if len(self.sample.turns) != len(self.policy_turns):
            raise ValueError(
                "scored and policy turns must align: "
                f"{len(self.sample.turns)} != {len(self.policy_turns)}"
            )
        for scored, policy in zip(self.sample.turns, self.policy_turns, strict=True):
            policy.validate()
            prompt_ids, response_ids = scored
            if list(policy.prompt_ids) != list(prompt_ids):
                raise ValueError("scored prompt ids differ from policy rollout ids")
            if list(policy.response_ids) != list(response_ids):
                raise ValueError("scored response ids differ from policy rollout ids")


@dataclass(frozen=True)
class TransitionUpdate:
    """One transition consumed by TRL's tokenwise clipped policy loss."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    advantage: float
    trajectory_id: str
    turn_index: int
    example_index: int
    trajectory_correct: bool
    trajectory_turn_weight: float = 1.0
    trajectory_token_weight: float = 1.0
    legal_success: bool = False
    local_penalty: float = 0.0


def standardized_group_advantages(
    rewards: Sequence[float],
    eligible: Sequence[bool],
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    """Match the historical population-std result-only group normalization."""
    if len(rewards) != len(eligible):
        raise ValueError("reward and eligibility vectors must have equal length")
    indices = [index for index, keep in enumerate(eligible) if keep]
    advantages = [0.0] * len(rewards)
    if not indices:
        return advantages
    values = [float(rewards[index]) for index in indices]
    # Identical decimal rewards such as eight copies of 0.2 can acquire a tiny
    # nonzero variance from floating-point summation. GRPO defines this group as
    # exactly homogeneous, so fail closed to a strict zero advantage before the
    # mean/std computation. Besides avoiding numerical policy noise, this lets the
    # trainer skip the otherwise null causal-prefix forwards and backwards.
    if all(value == values[0] for value in values[1:]):
        return advantages
    # The author implementation normalizes a float32 Torch reward tensor.  Do the
    # same operation here rather than computing in Python float64 and rounding
    # only the final coefficient: the latter differs by one or more float32 ULPs
    # for some K=8 compositions and can also turn an exact categorical zero into
    # a pseudo-signal.  Every training runtime has Torch; fsum below is only the
    # dependency-light diagnostic fallback.
    if torch is not None:
        reward_tensor = torch.tensor(values, dtype=torch.float32)
        std_tensor = reward_tensor.std(unbiased=False)
        if float(std_tensor) == 0.0:
            return advantages
        normalized = (
            reward_tensor - reward_tensor.mean()
        ) / (std_tensor + epsilon)
        for source_index, value in zip(indices, normalized.tolist(), strict=True):
            advantages[source_index] = float(value)
        return advantages

    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return advantages
    for index in indices:
        advantages[index] = (float(rewards[index]) - mean) / (std + epsilon)
    return advantages


def policy_reduction_advantages(
    updates: Sequence[TransitionUpdate],
    *,
    reduction: str,
    normalization_transition_count: int | None = None,
    normalization_trajectory_count: int | None = None,
) -> list[float]:
    """Return transition coefficients for transition- or trajectory-equal GRPO.

    The trainer still consumes exact causal turns. ``trajectory_mean`` makes turns
    equally weighted inside each trajectory. ``trajectory_token_mean`` instead
    weights a turn by its authored-token count, exactly matching the per-sample token
    mean used by TRUST-SQL without concatenating rebuilt rolling-state prefixes.
    """
    if reduction not in {
        "transition_mean",
        "trajectory_mean",
        "trajectory_token_mean",
    }:
        raise ValueError(f"unsupported policy reduction: {reduction}")
    if reduction == "transition_mean" or not updates:
        return [float(update.advantage) for update in updates]
    transition_count = normalization_transition_count or len(updates)
    trajectory_count = normalization_trajectory_count or len(
        {update.trajectory_id for update in updates}
    )
    if transition_count < 1 or trajectory_count < 1:
        raise ValueError("policy reduction normalization counts must be positive")
    mean_turns = transition_count / trajectory_count
    weight_field = (
        "trajectory_turn_weight"
        if reduction == "trajectory_mean"
        else "trajectory_token_weight"
    )
    return [
        float(update.advantage) * float(getattr(update, weight_field)) * mean_turns
        for update in updates
    ]


def retain_policy_contributing_updates(
    updates: Sequence[TransitionUpdate],
    *,
    policy_loss_coefficient: float,
    rank_loss_coefficient: float,
    kl_beta: float,
) -> tuple[list[TransitionUpdate], int]:
    """Drop mathematically null transitions from a policy-only GRPO batch.

    A homogeneous reward group has zero standardized advantage. When neither a
    ranking loss nor a KL term consumes the transitions, evaluating every causal
    prefix cannot affect gradients or optimizer state. One zero-advantage update is
    retained so the batch still occupies its required gradient-accumulation slot.
    """
    retained = list(updates)
    if (
        policy_loss_coefficient == 0.0
        or rank_loss_coefficient != 0.0
        or kl_beta != 0.0
    ):
        return retained, 0
    contributing = [update for update in retained if update.advantage != 0.0]
    if not contributing and retained:
        contributing = retained[:1]
    return contributing, len(retained) - len(contributing)


def build_transition_updates(
    episodes: Sequence[PolicyEpisode],
    *,
    reward_mode: str,
    train_turns: str = "all",
) -> list[TransitionUpdate]:
    """Flatten causal turns while preserving trajectory- or step-local credit."""
    if reward_mode not in {"result-only", "process"}:
        raise ValueError(f"unsupported reward mode: {reward_mode}")
    if train_turns not in {"all", "last"}:
        raise ValueError(f"unsupported train_turns value: {train_turns}")

    for episode in episodes:
        episode.validate()

    trajectory_advantages = [0.0] * len(episodes)
    result_groups: dict[Any, list[int]] = defaultdict(list)
    for episode_index, episode in enumerate(episodes):
        audit = episode.sample.audit_record
        if "example_index" not in audit:
            raise ValueError("rollout audit is missing the GRPO example_index")
        result_groups[audit["example_index"]].append(episode_index)
    for indices in result_groups.values():
        group_advantages = standardized_group_advantages(
            [float(episodes[index].sample.reward) for index in indices],
            [bool(episodes[index].sample.process_update) for index in indices],
        )
        for index, advantage in zip(indices, group_advantages, strict=True):
            trajectory_advantages[index] = advantage
    updates: list[TransitionUpdate] = []
    for episode_index, episode in enumerate(episodes):
        sample = episode.sample
        if not sample.process_update:
            continue
        indexed_turns = list(enumerate(episode.policy_turns))
        if train_turns == "last" and indexed_turns:
            indexed_turns = indexed_turns[-1:]
        trajectory_token_count = sum(
            len(turn.response_ids) for _, turn in indexed_turns
        )
        if indexed_turns and trajectory_token_count < 1:
            raise ValueError("a trainable trajectory must contain response tokens")

        if reward_mode == "process":
            if sample.step_rewards is None:
                raise ValueError("process episode is missing step-local rewards")
            if len(sample.step_rewards) != len(episode.policy_turns):
                raise ValueError(
                    "step rewards and policy turns must align: "
                    f"{len(sample.step_rewards)} != {len(episode.policy_turns)}"
                )

        trajectory_id = str(sample.audit_record["trajectory_id"])
        example_index = int(sample.audit_record["example_index"])
        process_steps = (
            (sample.audit_record.get("process_reward") or {}).get("steps") or []
        )
        if reward_mode == "process" and len(process_steps) != len(episode.policy_turns):
            raise ValueError(
                "process reward audit steps and policy turns must align: "
                f"{len(process_steps)} != {len(episode.policy_turns)}"
            )
        for turn_index, turn in indexed_turns:
            advantage = (
                float(sample.step_rewards[turn_index])
                if reward_mode == "process"
                else trajectory_advantages[episode_index]
            )
            updates.append(
                TransitionUpdate(
                    prompt_ids=turn.prompt_ids,
                    response_ids=turn.response_ids,
                    sampling_logprobs=turn.sampling_logprobs,
                    advantage=advantage,
                    trajectory_id=trajectory_id,
                    turn_index=turn_index,
                    example_index=example_index,
                    trajectory_correct=bool(sample.correct),
                    trajectory_turn_weight=1.0 / len(indexed_turns),
                    trajectory_token_weight=(
                        len(turn.response_ids) / trajectory_token_count
                    ),
                    legal_success=bool(
                        ((process_steps[turn_index].get("features") or {}).get(
                            "legal_success",
                            False,
                        ))
                    ) if process_steps else False,
                    local_penalty=float(
                        process_steps[turn_index].get("p_local") or 0.0
                    ) if process_steps else 0.0,
                )
            )
    return updates
