#!/usr/bin/env python3
"""Gold-free state-action ambiguity masking for result-only trajectory credit.

SAAM keeps the ordinary group-standardized terminal GRPO advantage, but removes
that coefficient from a decision when the same question, deterministic executed
prefix, and complete current tool call occur in both successful and unsuccessful
eligible trajectories.  It does not estimate a local value, invent a step reward,
or renormalize the surviving coefficients.

The state identity deliberately excludes model-authored reasoning and gold fields.
It contains only prior parsed tool calls and their Harness-owned structured
observations/errors.  JSON object key order is ignored.  The only list normalized
as unordered is ``describe_table.arguments.tables``; every other list keeps its
authored order.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Sequence

from frameworks.trl.transition_batch import PolicyEpisode, TransitionUpdate

try:
    # Keep lineage resolution in the audited replay implementation so the online
    # and offline identities cannot silently diverge.  The trainer's source lock
    # includes this module when the lineage-aware credit mode is enabled.
    from diagnostics.audit_saam_lineage_replay import LineageReplay
except ImportError:  # pragma: no cover - dependency-light unit tests may omit it
    LineageReplay = None


SCHEMA_VERSION = "saam-grpo-credit-v2"
CREDIT_ASSIGNMENTS = frozenset(
    {"trajectory", "saam-strict", "saam-asymmetric-error"}
)
DETERMINISTIC_ERROR_KINDS = frozenset(
    {
        "argument_validation_error",
        "execution_error",
        "no_progress_repeat",
        "protocol_error",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _canonical_action(
    parsed: Any,
    *,
    normalize_describe_tables: bool,
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
        return None
    # Round-trip through strict stable JSON so later mutation of the audit record
    # cannot alter an already-derived identity.
    normalized_arguments = json.loads(_stable_json(arguments))
    if normalize_describe_tables and tool == "describe_table":
        tables = normalized_arguments.get("tables")
        if isinstance(tables, list) and all(
            isinstance(table, str) for table in tables
        ):
            normalized_arguments["tables"] = sorted(tables)
    return {"tool": tool, "arguments": normalized_arguments}


def _harness_outcome(turn: dict[str, Any]) -> Any:
    """Return only environment-owned evidence needed by the next state."""
    if "tool_output" in turn:
        return {"kind": "tool_output", "value": turn["tool_output"]}
    error_event = turn.get("error_event")
    if isinstance(error_event, dict) or "execution_error_type" in turn:
        event = error_event if isinstance(error_event, dict) else {}
        return {
            "kind": "tool_error",
            "error_type": turn.get("execution_error_type")
            or event.get("error_type"),
            "error_code": event.get("error_code"),
            "details": event.get("details"),
            # The public error message is part of the next model-visible state.
            # It is emitted by the Harness/protocol, not used as reward authority.
            "message": event.get("message") or turn.get("execution_error"),
        }
    # A legal terminal answer has no following observation.  This marker is also
    # safe for a terminal wrong answer because both have already ended.
    return {"kind": "no_observation"}


@dataclass(frozen=True)
class _DecisionIdentity:
    trajectory_id: str
    example_index: int
    turn_index: int
    depth: int
    correct: bool
    state_signature: str
    action_signature: str | None


@dataclass(frozen=True)
class SAAMMaskAudit:
    """Batch-local, appendable evidence for exactly what SAAM changed."""

    schema_version: str
    credit_assignment: str
    eligible_episodes: int
    eligible_transitions: int
    matchable_transitions: int
    unmatchable_transitions: int
    repeated_state_action_groups: int
    ambiguous_state_action_groups: int
    ambiguous_transitions: int
    newly_zeroed_transitions: int
    newly_zeroed_response_tokens: int
    newly_zeroed_initial_transitions: int
    newly_zeroed_noninitial_transitions: int
    original_absolute_advantage_mass: float
    removed_absolute_advantage_mass: float
    fully_zeroed_trajectories: int
    deterministic_error_transitions: int = 0
    deterministic_error_transitions_by_kind: dict[str, int] | None = None
    infrastructure_timeout_transitions: int = 0
    timeout_penalized_transitions: int = 0
    shared_success_correct_kept: int = 0
    shared_success_wrong_suppressed: int = 0
    correct_error_positive_flips: int = 0
    correct_error_positive_mass_flipped: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        if result["deterministic_error_transitions_by_kind"] is None:
            result["deterministic_error_transitions_by_kind"] = {}
        return result


def _local_error_kind(turn: dict[str, Any]) -> str | None:
    """Classify only Harness-owned errors for the asymmetric local override."""

    event = turn.get("error_event")
    if not isinstance(event, dict) and "execution_error_type" not in turn:
        return None
    event = event if isinstance(event, dict) else {}
    error_type = str(
        event.get("error_type") or turn.get("execution_error_type") or ""
    ).strip()
    error_code = str(event.get("error_code") or "").strip()
    message = str(event.get("message") or turn.get("execution_error") or "")
    if error_type == "timeout_error" or error_code == "tool_execution_timeout":
        return "infrastructure_timeout"
    lower_message = message.casefold()
    if (
        error_type == "no_progress_error"
        or error_code == "no_progress_error"
        or "no_progress" in lower_message
    ):
        return "no_progress_repeat"
    if error_type == "argument_validation_error":
        return "argument_validation_error"
    if error_type in {"protocol_error", "carrier_error", "invalid_response"}:
        return "protocol_error"
    # The frozen Harness emits deterministic execution failures as structured
    # non-timeout errors.  Keep unknown structured errors conservative and local.
    return "execution_error"


def _decision_identities(
    episodes: Sequence[PolicyEpisode],
) -> dict[tuple[str, int], _DecisionIdentity]:
    identities: dict[tuple[str, int], _DecisionIdentity] = {}
    seen_trajectory_ids: set[str] = set()
    for episode in episodes:
        episode.validate()
        sample = episode.sample
        if not sample.process_update:
            continue
        audit = sample.audit_record
        trajectory_id = str(audit["trajectory_id"])
        if trajectory_id in seen_trajectory_ids:
            raise ValueError(f"duplicate trajectory_id in SAAM batch: {trajectory_id}")
        seen_trajectory_ids.add(trajectory_id)
        example_index = int(audit["example_index"])
        turns = audit.get("turns")
        if not isinstance(turns, list) or len(turns) != len(episode.policy_turns):
            raise ValueError(
                "SAAM requires one audited Harness turn per policy turn for "
                f"{trajectory_id}: {len(turns) if isinstance(turns, list) else None} "
                f"!= {len(episode.policy_turns)}"
            )

        if LineageReplay is None:
            raise RuntimeError(
                "lineage-aware SAAM requires diagnostics.audit_saam_lineage_replay"
            )
        # LineageReplay is Harness-only and excludes model reasoning.  It replaces
        # anonymous derived handles (e.g. filter_003) by their recursive
        # relation-derivation identity before hashing state/action.
        lineage_events, _ = LineageReplay(audit).replay()
        if len(lineage_events) != len(turns):
            raise ValueError(
                f"{trajectory_id}: lineage replay returned {len(lineage_events)} "
                f"events for {len(turns)} audited turns"
            )
        for depth, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"{trajectory_id}: audited turn {depth} is not an object")
            recorded_index = turn.get("turn_index")
            if recorded_index is not None and int(recorded_index) != depth:
                raise ValueError(
                    f"{trajectory_id}: audited turn index {recorded_index} != {depth}"
                )
            lineage_event = lineage_events[depth]
            state_signature = lineage_event.get("state_digest")
            action_signature = (
                lineage_event.get("action_digest")
                if lineage_event.get("matched")
                else None
            )
            identities[(trajectory_id, depth)] = _DecisionIdentity(
                trajectory_id=trajectory_id,
                example_index=example_index,
                turn_index=depth,
                depth=depth,
                correct=bool(sample.correct),
                state_signature=state_signature,
                action_signature=action_signature,
            )
    return identities


def _identity_errors(
    episodes: Sequence[PolicyEpisode],
) -> dict[tuple[str, int], str | None]:
    result: dict[tuple[str, int], str | None] = {}
    for episode in episodes:
        sample = episode.sample
        if not sample.process_update:
            continue
        trajectory_id = str(sample.audit_record["trajectory_id"])
        turns = sample.audit_record.get("turns") or []
        if len(turns) != len(episode.policy_turns):
            raise ValueError(
                "SAAM error audit requires one Harness turn per policy turn: "
                f"{trajectory_id}"
            )
        for depth, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"{trajectory_id}: audited turn {depth} is not an object")
            result[(trajectory_id, depth)] = _local_error_kind(turn)
    return result


def apply_state_action_ambiguity_mask(
    episodes: Sequence[PolicyEpisode],
    updates: Sequence[TransitionUpdate],
    *,
    credit_assignment: str,
) -> tuple[list[TransitionUpdate], SAAMMaskAudit]:
    """Zero contradictory trajectory coefficients without changing normalization.

    ``updates`` must still contain the original GRPO advantages.  The caller must
    compute any trajectory/transition reduction with the pre-mask transition and
    trajectory counts; this function intentionally returns no replacement counts.
    """
    if credit_assignment not in CREDIT_ASSIGNMENTS:
        raise ValueError(f"unsupported credit assignment: {credit_assignment}")

    original = list(updates)
    original_mass = sum(abs(float(update.advantage)) for update in original)
    if credit_assignment == "trajectory":
        return original, SAAMMaskAudit(
            schema_version=SCHEMA_VERSION,
            credit_assignment=credit_assignment,
            eligible_episodes=sum(
                bool(episode.sample.process_update) for episode in episodes
            ),
            eligible_transitions=len(original),
            matchable_transitions=0,
            unmatchable_transitions=0,
            repeated_state_action_groups=0,
            ambiguous_state_action_groups=0,
            ambiguous_transitions=0,
            newly_zeroed_transitions=0,
            newly_zeroed_response_tokens=0,
            newly_zeroed_initial_transitions=0,
            newly_zeroed_noninitial_transitions=0,
            original_absolute_advantage_mass=original_mass,
            removed_absolute_advantage_mass=0.0,
            fully_zeroed_trajectories=0,
        )

    identities = _decision_identities(episodes)
    update_keys = {(update.trajectory_id, update.turn_index) for update in original}
    if len(update_keys) != len(original):
        raise ValueError("SAAM received duplicate trajectory/turn updates")
    missing = sorted(update_keys - identities.keys())
    if missing:
        raise ValueError(f"SAAM has no audited decision identity for updates: {missing[:5]}")

    grouped: dict[tuple[int, str, str], list[_DecisionIdentity]] = defaultdict(list)
    matchable = 0
    for identity in identities.values():
        if (identity.trajectory_id, identity.turn_index) not in update_keys:
            continue
        if identity.action_signature is None:
            continue
        matchable += 1
        grouped[
            (
                identity.example_index,
                identity.state_signature,
                identity.action_signature,
            )
        ].append(identity)

    repeated_groups = [values for values in grouped.values() if len(values) >= 2]
    ambiguous_groups = [
        values
        for values in repeated_groups
        if {identity.correct for identity in values} == {False, True}
    ]
    ambiguous_keys = {
        (identity.trajectory_id, identity.turn_index)
        for values in ambiguous_groups
        for identity in values
    }

    masked: list[TransitionUpdate] = []
    removed_mass = 0.0
    newly_zeroed = 0
    newly_zeroed_tokens = 0
    newly_zeroed_initial = 0
    for update in original:
        key = (update.trajectory_id, update.turn_index)
        if key in ambiguous_keys and update.advantage != 0.0:
            removed_mass += abs(float(update.advantage))
            newly_zeroed += 1
            newly_zeroed_tokens += len(update.response_ids)
            newly_zeroed_initial += int(update.turn_index == 0)
            masked.append(replace(update, advantage=0.0))
        else:
            masked.append(update)

    original_by_trajectory: dict[str, list[TransitionUpdate]] = defaultdict(list)
    by_trajectory: dict[str, list[TransitionUpdate]] = defaultdict(list)
    for update in original:
        original_by_trajectory[update.trajectory_id].append(update)
    for update in masked:
        by_trajectory[update.trajectory_id].append(update)
    fully_zeroed = sum(
        any(update.advantage != 0.0 for update in original_by_trajectory[trajectory_id])
        and all(update.advantage == 0.0 for update in values)
        for trajectory_id, values in by_trajectory.items()
    )
    return masked, SAAMMaskAudit(
        schema_version=SCHEMA_VERSION,
        credit_assignment=credit_assignment,
        eligible_episodes=sum(
            bool(episode.sample.process_update) for episode in episodes
        ),
        eligible_transitions=len(original),
        matchable_transitions=matchable,
        unmatchable_transitions=len(original) - matchable,
        repeated_state_action_groups=len(repeated_groups),
        ambiguous_state_action_groups=len(ambiguous_groups),
        ambiguous_transitions=len(ambiguous_keys),
        newly_zeroed_transitions=newly_zeroed,
        newly_zeroed_response_tokens=newly_zeroed_tokens,
        newly_zeroed_initial_transitions=newly_zeroed_initial,
        newly_zeroed_noninitial_transitions=newly_zeroed - newly_zeroed_initial,
        original_absolute_advantage_mass=original_mass,
        removed_absolute_advantage_mass=removed_mass,
        fully_zeroed_trajectories=fully_zeroed,
    )


def apply_asymmetric_error_credit(
    episodes: Sequence[PolicyEpisode],
    updates: Sequence[TransitionUpdate],
    *,
    error_penalty: float = 1.0,
) -> tuple[list[TransitionUpdate], SAAMMaskAudit]:
    """Apply asymmetric SAAM plus local deterministic Harness-error penalties.

    Successful shared actions retain positive credit on correct trajectories and
    lose only the negative occurrence on incorrect trajectories.  Every explicit
    model error is overridden by a local negative coefficient, including an error
    inside a trajectory that later reaches a correct terminal answer.  All other
    actions keep the original GRPO coefficient and the caller must retain the
    pre-credit normalization counts.
    """

    if not math.isfinite(error_penalty) or error_penalty <= 0.0:
        raise ValueError("error_penalty must be finite and positive")
    original = list(updates)
    original_mass = sum(abs(float(update.advantage)) for update in original)
    identities = _decision_identities(episodes)
    errors = _identity_errors(episodes)
    update_keys = {(update.trajectory_id, update.turn_index) for update in original}
    if len(update_keys) != len(original):
        raise ValueError("asymmetric SAAM received duplicate trajectory/turn updates")
    missing = sorted(update_keys - identities.keys())
    if missing:
        raise ValueError(
            "asymmetric SAAM has no audited decision identity for updates: "
            f"{missing[:5]}"
        )

    grouped: dict[tuple[int, str, str], list[_DecisionIdentity]] = defaultdict(list)
    matchable = 0
    for identity in identities.values():
        key = (identity.trajectory_id, identity.turn_index)
        if key not in update_keys or identity.action_signature is None:
            continue
        matchable += 1
        grouped[
            (identity.example_index, identity.state_signature, identity.action_signature)
        ].append(identity)
    repeated_groups = [values for values in grouped.values() if len(values) >= 2]
    ambiguous_groups = [
        values
        for values in repeated_groups
        if {identity.correct for identity in values} == {False, True}
    ]
    ambiguous_keys = {
        (identity.trajectory_id, identity.turn_index)
        for values in ambiguous_groups
        for identity in values
    }

    masked: list[TransitionUpdate] = []
    removed_mass = 0.0
    zeroed = 0
    zeroed_tokens = 0
    zeroed_initial = 0
    error_counts: defaultdict[str, int] = defaultdict(int)
    timeout_count = 0
    timeout_penalized = 0
    episode_by_trajectory = {
        str(episode.sample.audit_record["trajectory_id"]): episode
        for episode in episodes
        if episode.sample.process_update
    }
    kept_correct = 0
    suppressed_wrong = 0
    correct_error_flips = 0
    correct_error_flip_mass = 0.0
    for update in original:
        key = (update.trajectory_id, update.turn_index)
        identity = identities[key]
        error_kind = errors.get(key)
        result_reward = (
            episode_by_trajectory[identity.trajectory_id]
            .sample.audit_record.get("result_reward")
            or {}
        )
        # The final four-level result contract always treats a tool timeout as
        # a policy-visible penalty action.  Keep the explicit flag for older
        # records, but do not let a missing sidecar field silently turn a
        # four-level timeout back into the historical zero-credit behavior.
        timeout_penalty_enabled = bool(
            result_reward.get("policy_failure_penalty_enabled", False)
            or result_reward.get("profile") == "four-level"
        )
        if error_kind == "infrastructure_timeout" and not timeout_penalty_enabled:
            effective = 0.0
            decision = "infrastructure_timeout_zero"
            timeout_count += 1
            if update.advantage != 0.0:
                removed_mass += abs(float(update.advantage))
                zeroed += 1
                zeroed_tokens += len(update.response_ids)
                zeroed_initial += int(update.turn_index == 0)
        elif error_kind == "infrastructure_timeout":
            # A four-level result-only run treats a timeout as a policy-visible
            # error.  Keep the historical zeroing for binary controls, but in
            # this explicitly opted-in profile force the timed-out action
            # negative, including a correct trajectory that recovered later.
            effective = -max(abs(float(update.advantage)), error_penalty)
            decision = "deterministic_error_negative:infrastructure_timeout"
            timeout_count += 1
            timeout_penalized += 1
            error_counts["infrastructure_timeout"] += 1
            if identity.correct and update.advantage > 0.0:
                correct_error_flips += 1
                correct_error_flip_mass += float(update.advantage)
        elif error_kind in DETERMINISTIC_ERROR_KINDS:
            effective = -max(abs(float(update.advantage)), error_penalty)
            decision = f"deterministic_error_negative:{error_kind}"
            error_counts[error_kind] += 1
            if identity.correct and update.advantage > 0.0:
                correct_error_flips += 1
                correct_error_flip_mass += float(update.advantage)
        elif key in ambiguous_keys:
            if identity.correct:
                effective = float(update.advantage)
                decision = "shared_success_correct_keep_positive"
                kept_correct += 1
            else:
                effective = 0.0
                decision = "shared_success_wrong_suppress_negative"
                suppressed_wrong += 1
                if update.advantage != 0.0:
                    removed_mass += abs(float(update.advantage))
                    zeroed += 1
                    zeroed_tokens += len(update.response_ids)
                    zeroed_initial += int(update.turn_index == 0)
        else:
            effective = float(update.advantage)
            decision = "unmatched_keep_trajectory"
        # Keep the audit decision available to downstream logging without adding
        # mutable metadata to TransitionUpdate.  The effective coefficient is the
        # only value consumed by the trainer.
        del decision
        masked.append(replace(update, advantage=effective))

    original_by_trajectory: dict[str, list[TransitionUpdate]] = defaultdict(list)
    masked_by_trajectory: dict[str, list[TransitionUpdate]] = defaultdict(list)
    for update in original:
        original_by_trajectory[update.trajectory_id].append(update)
    for update in masked:
        masked_by_trajectory[update.trajectory_id].append(update)
    fully_zeroed = sum(
        any(update.advantage != 0.0 for update in original_by_trajectory[trajectory_id])
        and all(update.advantage == 0.0 for update in values)
        for trajectory_id, values in masked_by_trajectory.items()
    )
    return masked, SAAMMaskAudit(
        schema_version=SCHEMA_VERSION,
        credit_assignment="saam-asymmetric-error",
        eligible_episodes=sum(bool(episode.sample.process_update) for episode in episodes),
        eligible_transitions=len(original),
        matchable_transitions=matchable,
        unmatchable_transitions=len(original) - matchable,
        repeated_state_action_groups=len(repeated_groups),
        ambiguous_state_action_groups=len(ambiguous_groups),
        ambiguous_transitions=len(ambiguous_keys),
        newly_zeroed_transitions=zeroed,
        newly_zeroed_response_tokens=zeroed_tokens,
        newly_zeroed_initial_transitions=zeroed_initial,
        newly_zeroed_noninitial_transitions=zeroed - zeroed_initial,
        original_absolute_advantage_mass=original_mass,
        removed_absolute_advantage_mass=removed_mass,
        fully_zeroed_trajectories=fully_zeroed,
        deterministic_error_transitions=sum(error_counts.values()),
        deterministic_error_transitions_by_kind=dict(sorted(error_counts.items())),
        infrastructure_timeout_transitions=timeout_count,
        timeout_penalized_transitions=timeout_penalized,
        shared_success_correct_kept=kept_correct,
        shared_success_wrong_suppressed=suppressed_wrong,
        correct_error_positive_flips=correct_error_flips,
        correct_error_positive_mass_flipped=correct_error_flip_mass,
    )
