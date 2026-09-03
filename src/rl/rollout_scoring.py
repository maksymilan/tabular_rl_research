#!/usr/bin/env python3
"""Framework-neutral scoring for one completed table-agent rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from terminal_reward import terminal_result_reward

if TYPE_CHECKING:
    from counterfactual_suite import CounterfactualTaskSuite
    from process_credit import ProcessRewardConfig


TurnTokens = tuple[list[int], list[int]]
NONSEMANTIC_GENERATION_FAILURES = frozenset(
    {"generation_oom", "generation_length", "context_overflow"}
)
UNTRAINABLE_RUNTIME_FAILURES = frozenset({"generation_oom", "context_overflow"})
_TIMEOUT_FAILURE_TYPES = frozenset({"timeout_error"})
_TIMEOUT_ERROR_CODES = frozenset({"tool_execution_timeout"})


def _timeout_event(event: Any) -> bool:
    """Recognize only canonical structured timeout markers, never free text."""

    if not isinstance(event, dict):
        return False
    return (
        event.get("failure_type") in _TIMEOUT_FAILURE_TYPES
        or event.get("error_type") in _TIMEOUT_FAILURE_TYPES
        or event.get("execution_error_type") in _TIMEOUT_FAILURE_TYPES
        or event.get("recovered_from_error_type") in _TIMEOUT_FAILURE_TYPES
        or event.get("error_code") in _TIMEOUT_ERROR_CODES
        or event.get("code") in _TIMEOUT_ERROR_CODES
    )


def _has_timeout_evidence(record: dict[str, Any]) -> bool:
    """Detect terminal or recovered tool timeouts in a canonical rollout record."""

    if record.get("failure_type") in _TIMEOUT_FAILURE_TYPES:
        return True
    if any(_timeout_event(event) for event in (record.get("error_events") or [])):
        return True
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if _timeout_event(turn) or _timeout_event(turn.get("error_event")):
            return True
    return False


def _has_structured_error(record: dict[str, Any]) -> bool:
    """Return whether the episode contains any semantic/tool error evidence.

    ``errors`` and ``error_events`` are Harness-owned fields.  Timeout is checked
    separately because a terminal timeout may be represented only by its failure
    type.  ``generation_length`` is also policy-authored evidence: the sampled
    response reached the configured token limit without completing one legal
    action.  OOM and context overflow remain infrastructure/runtime exclusions.
    """

    if record.get("failure_type") == "generation_length":
        return True

    try:
        if int(record.get("errors", 0) or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    if record.get("error_events"):
        return True
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        try:
            if int(turn.get("errors", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        if turn.get("error_event") or turn.get("error_events"):
            return True
    return _has_timeout_evidence(record)


@dataclass
class RolloutSample:
    """One scored episode shared by every optimization backend."""

    reward: float
    correct: bool
    failure_type: str | None
    turns: list[TurnTokens]
    audit_record: dict[str, Any]
    step_rewards: list[float] | None = None
    process_update: bool = True


def episode_example(metadata: dict[str, Any]) -> dict[str, Any]:
    """Preserve dataset-adapter fields when constructing an interactive episode."""
    return {
        "db_id": metadata["db_id"],
        "db_path": metadata.get("db_path"),
        "question": metadata["question"],
        "query": metadata["gold_sql"],
        "gold_sql": metadata["gold_sql"],
        "external_knowledge": metadata.get("external_knowledge"),
    }


def score_completed_rollout(
    env,
    turns: list[TurnTokens],
    metadata: dict[str, Any],
    *,
    sample_index: int,
    reward_mode: str,
    process_config: ProcessRewardConfig | None,
    process_admission_policy: str,
    denotation_comparison: str,
    counterfactual_suite: CounterfactualTaskSuite | None,
    result_reward_profile: str = "binary",
    generation_truncation: dict[str, Any] | None = None,
) -> RolloutSample:
    """Attach terminal/process credit without depending on a trainer implementation."""
    record = env.record()
    record["trajectory_id"] = f"rl_{metadata['example_index']}_sample_{sample_index}"
    if generation_truncation is not None:
        record["generation_truncation"] = dict(generation_truncation)
        # The collector retains the truncated response ids in ``turns`` for a
        # four-level negative update, while the Harness cannot execute that
        # incomplete carrier and therefore does not append a normal turn.  Add
        # one explicit, unmatchable audit event so SAAM's policy-turn and
        # Harness-turn cardinalities remain aligned.  No model-authored text or
        # action arguments are fabricated here.
        audited_turns = record.setdefault("turns", [])
        truncation_index = int(generation_truncation.get("turn_index", len(turns) - 1))
        if len(audited_turns) == len(turns) - 1:
            audited_turns.append(
                {
                    "turn_index": truncation_index,
                    "parsed": None,
                    "generation_truncation": dict(generation_truncation),
                }
            )
    step_rewards = None
    # Preserve the historical exclusion contract for binary controls and process
    # replay.  The four-level result-only experiment deliberately makes two
    # policy-attributable failures trainable: a length-truncated authored response
    # and a Harness-recorded SQL/tool timeout.  The former retains its exact sampled
    # response ids in ``turns``; the latter retains the attempted action and causal
    # prefix.  OOM and context overflow remain untrainable runtime failures.
    penalize_policy_failures = (
        reward_mode == "result-only" and result_reward_profile == "four-level"
    )
    if penalize_policy_failures:
        process_update = record["failure_type"] not in UNTRAINABLE_RUNTIME_FAILURES
    else:
        process_update = (
            record["failure_type"] not in NONSEMANTIC_GENERATION_FAILURES
            and not _has_timeout_evidence(record)
        )
    scalar_reward = terminal_result_reward(
        record["correct"],
        executable=record.get("legal", False),
        profile=result_reward_profile,
        has_errors=_has_structured_error(record),
    )
    if not process_update:
        scalar_reward = 0.0
    if reward_mode == "result-only":
        record["result_reward"] = {
            "profile": result_reward_profile,
            "correct": bool(record["correct"]),
            "executable_terminal": bool(record.get("legal", False)),
            "has_structured_error": _has_structured_error(record),
            "policy_failure_penalty_enabled": penalize_policy_failures,
            "value": scalar_reward,
        }
    if not process_update:
        record["optimization_exclusion"] = "nonsemantic_runtime_failure"

    if reward_mode == "process":
        from external_failure_adapter import normalize_failure_record
        from process_credit import score_rollout_trajectory
        from trajectory_replay import evaluate_counterfactual_suite

        if process_config is None:
            raise ValueError("process reward mode requires a ProcessRewardConfig")
        if not process_update:
            step_rewards = []
            scalar_reward = 0.0
            record["process_reward_exclusion"] = "nonsemantic_runtime_failure"
        else:
            normalized, exclusion = normalize_failure_record(
                record,
                {
                    "example_id": record["trajectory_id"],
                    "dataset": "bird-sql",
                    "split": "train",
                    "db_id": metadata["db_id"],
                    "db_path": metadata.get("db_path"),
                    "question": metadata["question"],
                    "gold_sql": metadata["gold_sql"],
                    "external_knowledge": metadata.get("external_knowledge"),
                    "denotation_comparison": denotation_comparison,
                },
            )
            if normalized is None:
                step_rewards = []
                process_update = False
                scalar_reward = 0.0
                record["process_reward_exclusion"] = exclusion
            else:
                reward = score_rollout_trajectory(
                    normalized,
                    process_config,
                    denotation_comparison=denotation_comparison,
                )
                step_rewards = [step.reward for step in reward.steps]
                if len(step_rewards) != len(turns):
                    raise RuntimeError(
                        "generated turns and replayed process steps do not align: "
                        f"{len(turns)} != {len(step_rewards)}"
                    )
                process_update = reward.process_update
                scalar_reward = reward.total_reward
                record["process_reward"] = reward.to_dict()
                record["process_admission_policy"] = process_admission_policy
                if (
                    reward.correct
                    and process_update
                    and process_admission_policy == "counterfactual-completeness"
                ):
                    if counterfactual_suite is None:
                        raise RuntimeError(
                            "correct process trajectory has no counterfactual task suite"
                        )
                    completeness = evaluate_counterfactual_suite(
                        normalized,
                        counterfactual_suite.database_paths,
                        min_informative_databases=(
                            counterfactual_suite.min_informative_databases
                        ),
                        denotation_comparison=denotation_comparison,
                    )
                    record["counterfactual_completeness"] = completeness.to_dict()
                    if not completeness.passed:
                        process_update = False
                        scalar_reward = 0.0
                        record["process_reward_exclusion"] = (
                            f"counterfactual_completeness:{completeness.reason}"
                        )

    # Persist the exact optimizer-admission decision beside the rollout
    # evidence.  RolloutSample already carries this boolean to transition
    # construction; recording the same final value makes the append-only
    # rollout log independently auditable without changing gradient semantics.
    record["process_update"] = bool(process_update)

    return RolloutSample(
        reward=scalar_reward,
        correct=bool(record["correct"]),
        failure_type=record["failure_type"],
        turns=turns,
        audit_record=record,
        step_rewards=step_rewards,
        process_update=process_update,
    )
