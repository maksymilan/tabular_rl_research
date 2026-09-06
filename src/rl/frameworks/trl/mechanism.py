"""Composable RL objective mechanisms.

The trainer owns batching and optimization; this module owns the policy objective
choices.  New reward/credit/reduction mechanisms should implement this small
interface instead of adding another branch to the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rl.frameworks.trl.state_action_ambiguity import (
    apply_asymmetric_error_credit,
    apply_state_action_ambiguity_mask,
)
from rl.frameworks.trl.transition_batch import (
    PolicyEpisode,
    TransitionUpdate,
    build_transition_updates,
    policy_reduction_advantages,
)


CreditHandler = Callable[["RLMechanism", Sequence[PolicyEpisode], Sequence[TransitionUpdate]], tuple[list[TransitionUpdate], Any | None]]


def _trajectory_credit(
    mechanism: "RLMechanism",
    episodes: Sequence[PolicyEpisode],
    updates: Sequence[TransitionUpdate],
) -> tuple[list[TransitionUpdate], Any | None]:
    del mechanism, episodes
    return list(updates), None


def _strict_credit(
    mechanism: "RLMechanism",
    episodes: Sequence[PolicyEpisode],
    updates: Sequence[TransitionUpdate],
) -> tuple[list[TransitionUpdate], Any | None]:
    return apply_state_action_ambiguity_mask(
        episodes, updates, credit_assignment=mechanism.credit_assignment
    )


def _asymmetric_credit(
    mechanism: "RLMechanism",
    episodes: Sequence[PolicyEpisode],
    updates: Sequence[TransitionUpdate],
) -> tuple[list[TransitionUpdate], Any | None]:
    return apply_asymmetric_error_credit(
        episodes,
        updates,
        error_penalty=mechanism.error_penalty,
        use_ambiguity_mask=mechanism.credit_assignment == "saam-asymmetric-error",
    )


# Registering a new credit component extends this table; the trainer does not
# gain another mechanism-specific branch.
CREDIT_HANDLERS: dict[str, CreditHandler] = {
    "trajectory": _trajectory_credit,
    "saam-strict": _strict_credit,
    "saam-asymmetric-error": _asymmetric_credit,
    "saam-asymmetric-error-no-mask": _asymmetric_credit,
}


def register_credit_handler(name: str, handler: CreditHandler) -> None:
    if not name or name in CREDIT_HANDLERS:
        raise ValueError(f"credit handler name is empty or already registered: {name!r}")
    CREDIT_HANDLERS[name] = handler


@dataclass(frozen=True)
class RLMechanism:
    """A serializable composition of reward, credit and policy reduction.

    ``TransitionGRPOTrainer`` accepts this object as a strategy component.  The
    legacy scalar constructor arguments remain supported and are converted to
    this object, so existing launchers keep the same behavior.
    """

    reward_mode: str = "result-only"
    result_advantage_profile: str = "stored"
    clean_advantage_weight: float = 0.25
    policy_reduction: str = "transition_mean"
    credit_assignment: str = "trajectory"
    error_penalty: float = 1.0
    span_balance_alpha: float | None = None
    kl_beta: float = 0.0

    @classmethod
    def from_legacy_args(cls, **kwargs: Any) -> "RLMechanism":
        fields = {
            "reward_mode",
            "result_advantage_profile",
            "clean_advantage_weight",
            "policy_reduction",
            "credit_assignment",
            "error_penalty",
            "span_balance_alpha",
            "kl_beta",
        }
        return cls(**{key: kwargs[key] for key in fields if key in kwargs})

    def build_updates(
        self,
        episodes: Sequence[PolicyEpisode],
        *,
        train_turns: str,
    ) -> list[TransitionUpdate]:
        return build_transition_updates(
            episodes,
            reward_mode=self.reward_mode,
            train_turns=train_turns,
            result_advantage_profile=self.result_advantage_profile,
            clean_advantage_weight=self.clean_advantage_weight,
        )

    def apply_credit(
        self,
        episodes: Sequence[PolicyEpisode],
        updates: Sequence[TransitionUpdate],
    ) -> tuple[list[TransitionUpdate], Any | None]:
        try:
            handler = CREDIT_HANDLERS[self.credit_assignment]
        except KeyError as exc:
            raise ValueError(f"unsupported credit assignment: {self.credit_assignment}") from exc
        return handler(self, episodes, updates)

    def effective_advantages(
        self,
        updates: Sequence[TransitionUpdate],
        *,
        transition_count: int,
        trajectory_count: int,
    ) -> list[float]:
        return policy_reduction_advantages(
            updates,
            reduction=self.policy_reduction,
            normalization_transition_count=transition_count,
            normalization_trajectory_count=trajectory_count,
        )
