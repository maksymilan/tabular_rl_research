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
    turn_has_harness_error: bool | None = None


def build_transition_microbatch_ranges(
    prompt_lengths: Sequence[int],
    completion_lengths: Sequence[int],
    *,
    max_rows: int,
    token_budget: int = 0,
) -> list[tuple[int, int]]:
    """Pack adjacent transitions under a padded-token budget.

    The trainer pads prompts and completions independently, then concatenates
    them for the model forward.  A batch containing ``n`` rows therefore has a
    conservative padded-token footprint of ``n * (max_prompt + max_completion)``.
    ``token_budget=0`` preserves the historical fixed-row behavior.  A single
    row is always admitted even when it exceeds the budget; otherwise an
    individual long transition could never make progress.

    The caller is responsible for ordering rows by length when padding
    efficiency matters.  Keeping this helper pure Python makes the packing
    contract testable without importing Torch or TRL.
    """

    if len(prompt_lengths) != len(completion_lengths):
        raise ValueError("prompt and completion length vectors must align")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    total = len(prompt_lengths)
    if total == 0:
        return []
    if token_budget == 0:
        return [
            (start, min(total, start + max_rows))
            for start in range(0, total, max_rows)
        ]

    ranges: list[tuple[int, int]] = []
    start = 0
    max_prompt = 0
    max_completion = 0
    for index, (prompt_length, completion_length) in enumerate(
        zip(prompt_lengths, completion_lengths, strict=True)
    ):
        prompt_length = int(prompt_length)
        completion_length = int(completion_length)
        if prompt_length < 1 or completion_length < 1:
            raise ValueError("transition lengths must be positive")
        candidate_rows = index - start + 1
        candidate_max_prompt = max(max_prompt, prompt_length)
        candidate_max_completion = max(max_completion, completion_length)
        candidate_cost = candidate_rows * (
            candidate_max_prompt + candidate_max_completion
        )
        exceeds_rows = candidate_rows > max_rows
        exceeds_tokens = candidate_cost > token_budget
        # A long transition is allowed to stand alone even if its own padded
        # footprint exceeds the configured budget.
        if index > start and (exceeds_rows or exceeds_tokens):
            ranges.append((start, index))
            start = index
            max_prompt = prompt_length
            max_completion = completion_length
        else:
            max_prompt = candidate_max_prompt
            max_completion = candidate_max_completion
    ranges.append((start, total))
    return ranges


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


def _episode_has_harness_error(episode: PolicyEpisode) -> bool:
    audit = episode.sample.audit_record
    # Keep the cleanliness bit aligned with runtime.rollout_scoring.  A length
    # truncation can have no Harness error event because no legal action was
    # emitted, but it is still a policy-visible failure in the four-level
    # result contract.
    if audit.get("failure_type") in {"generation_length", "timeout_error"}:
        return True
    if audit.get("errors") or audit.get("error_events"):
        return True
    return any(
        isinstance(turn, dict)
        and (
            turn.get("error_event")
            or turn.get("execution_error")
            or turn.get("execution_error_type")
        )
        for turn in (audit.get("turns") or [])
    )


def _turn_has_harness_error(turn: Any) -> bool | None:
    """Classify this turn's outcome, never its incoming recovery context."""

    if not isinstance(turn, dict):
        return None
    return bool(
        isinstance(turn.get("error_event"), dict)
        or turn.get("execution_error_type")
    )


def correctness_primary_clean_secondary_advantages(
    episodes: Sequence[PolicyEpisode],
    *,
    clean_weight: float = 0.25,
) -> list[float]:
    """Make denotation correctness primary and Harness cleanliness bounded.

    The correctness bit determines the sign after group standardization.  The
    clean/error bit can only scale that sign by ``1 +/- clean_weight``; it can
    never turn a wrong trajectory positive or a correct trajectory negative.
    This is intentionally a group-level transform, not a new model-authored or
    gold-SQL reward.
    """
    if not 0.0 <= clean_weight < 1.0:
        raise ValueError("clean_weight must be in [0, 1)")
    eligible = [bool(episode.sample.process_update) for episode in episodes]
    primary = standardized_group_advantages(
        [1.0 if bool(episode.sample.correct) else 0.0 for episode in episodes],
        eligible,
    )
    result: list[float] = []
    for episode, advantage in zip(episodes, primary, strict=True):
        if advantage == 0.0:
            result.append(0.0)
            continue
        clean = not _episode_has_harness_error(episode)
        if bool(episode.sample.correct):
            multiplier = 1.0 + clean_weight if clean else 1.0 - clean_weight
        else:
            multiplier = 1.0 - clean_weight if clean else 1.0 + clean_weight
        result.append(float(advantage) * multiplier)
    return result


def class_conditional_routing_advantages(
    episodes: Sequence[PolicyEpisode],
    *,
    clean_weight: float = 0.25,
) -> list[float]:
    """Route result advantages by the outcome composition of each GRPO group.

    Mixed groups use denotation correctness for the sign.  Cleanliness only
    changes the magnitude of correct trajectories; clean and erroneous wrong
    trajectories remain equally negative so legality cannot become a proxy for
    correctness.  All-correct groups use cleanliness as a bounded efficiency
    signal, while all-wrong groups receive no result-only advantage because
    they contain no correct reference from which to infer a preferred path.
    """
    if not 0.0 <= clean_weight < 1.0:
        raise ValueError("clean_weight must be in [0, 1)")
    eligible = [bool(episode.sample.process_update) for episode in episodes]
    correct = [bool(episode.sample.correct) for episode in episodes]
    eligible_correct = [ok and keep for ok, keep in zip(correct, eligible, strict=True)]
    eligible_wrong = [not ok and keep for ok, keep in zip(correct, eligible, strict=True)]
    has_correct = any(eligible_correct)
    has_wrong = any(eligible_wrong)
    if has_wrong and not has_correct:
        return [0.0] * len(episodes)
    if not has_wrong:
        clean = [
            1.0 if not _episode_has_harness_error(episode) else 0.0
            for episode in episodes
        ]
        return [
            clean_weight * advantage
            for advantage in standardized_group_advantages(clean, eligible)
        ]

    primary = standardized_group_advantages(
        [1.0 if ok else 0.0 for ok in correct],
        eligible,
    )
    result: list[float] = []
    for episode, is_correct, advantage in zip(
        episodes, correct, primary, strict=True
    ):
        if advantage == 0.0:
            result.append(0.0)
            continue
        if not is_correct:
            result.append(float(advantage))
            continue
        clean = not _episode_has_harness_error(episode)
        multiplier = 1.0 + clean_weight if clean else 1.0 - clean_weight
        result.append(float(advantage) * multiplier)
    return result


def smc_mode_concentration_advantages(
    episodes: Sequence[PolicyEpisode],
) -> list[float]:
    """Select the highest-probability correct/wrong modes in a mixed group.

    This is an isolated diagnostic profile for the proposed SMC-GRPO idea.  It
    does not use Harness-authored explanations or gold SQL: the Harness only
    supplies the terminal correctness bit, while the rollout stores the old
    policy's sampled token log-probabilities.  A trajectory score is the mean
    sampled log-probability over all authored response tokens.  In a mixed
    group, only the highest-scoring correct trajectory receives ``+1`` and only
    the highest-scoring wrong trajectory receives ``-1``.  Homogeneous groups
    receive no advantage because they do not provide a within-question
    correctness contrast.

    The sparse +/-1 coefficients intentionally match the scale of a binary
    standardized GRPO group with one positive and one negative member.  The
    score is used only for selection, not as a reward, so an unusually long or
    low-probability trajectory cannot create an unbounded policy coefficient.
    """

    result = [0.0] * len(episodes)
    eligible = [bool(episode.sample.process_update) for episode in episodes]
    correct = [bool(episode.sample.correct) for episode in episodes]

    def trajectory_score(episode: PolicyEpisode) -> float:
        values = [
            float(logprob)
            for turn in episode.policy_turns
            for logprob in turn.sampling_logprobs
        ]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(
                "smc-mode-concentration requires finite sampled logprobs for "
                "every eligible trajectory"
            )
        return math.fsum(values) / len(values)

    correct_indices = [
        index for index, keep in enumerate(eligible)
        if keep and correct[index]
    ]
    wrong_indices = [
        index for index, keep in enumerate(eligible)
        if keep and not correct[index]
    ]
    if not correct_indices or not wrong_indices:
        return result

    best_correct = max(correct_indices, key=lambda index: trajectory_score(episodes[index]))
    best_wrong = max(wrong_indices, key=lambda index: trajectory_score(episodes[index]))
    result[best_correct] = 1.0
    result[best_wrong] = -1.0
    return result


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
    result_advantage_profile: str = "stored",
    clean_advantage_weight: float = 0.25,
    require_turn_error_audit: bool = False,
) -> list[TransitionUpdate]:
    """Flatten causal turns while preserving trajectory- or step-local credit."""
    if reward_mode not in {"result-only", "process"}:
        raise ValueError(f"unsupported reward mode: {reward_mode}")
    if train_turns not in {"all", "last"}:
        raise ValueError(f"unsupported train_turns value: {train_turns}")
    if result_advantage_profile not in {
        "stored",
        "correctness-primary-clean-secondary",
        "class-conditional-routing",
        "smc-mode-concentration",
    }:
        raise ValueError(
            f"unsupported result advantage profile: {result_advantage_profile}"
        )

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
        group = [episodes[index] for index in indices]
        if result_advantage_profile == "correctness-primary-clean-secondary":
            group_advantages = correctness_primary_clean_secondary_advantages(
                group,
                clean_weight=clean_advantage_weight,
            )
        elif result_advantage_profile == "class-conditional-routing":
            group_advantages = class_conditional_routing_advantages(
                group,
                clean_weight=clean_advantage_weight,
            )
        elif result_advantage_profile == "smc-mode-concentration":
            group_advantages = smc_mode_concentration_advantages(group)
        else:
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
        audit_turns = sample.audit_record.get("turns") or []
        if require_turn_error_audit and (
            not isinstance(audit_turns, list)
            or len(audit_turns) != len(episode.policy_turns)
            or any(
                not isinstance(audit_turn, dict)
                or audit_turn.get("turn_index", index) != index
                for index, audit_turn in enumerate(audit_turns)
            )
        ):
            raise ValueError(
                "hybrid span routing requires aligned per-turn Harness audit: "
                f"{trajectory_id}"
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
                    turn_has_harness_error=(
                        _turn_has_harness_error(audit_turns[turn_index])
                        if turn_index < len(audit_turns)
                        else None
                    ),
                )
            )
    return updates
