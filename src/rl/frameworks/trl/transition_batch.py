#!/usr/bin/env python3
"""Pure transition accounting shared by the TRL rollout and trainer adapters."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


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
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return advantages
    for index in indices:
        advantages[index] = (float(rewards[index]) - mean) / (std + epsilon)
    return advantages


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
