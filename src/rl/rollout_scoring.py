#!/usr/bin/env python3
"""Framework-neutral scoring for one completed table-agent rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from counterfactual_suite import CounterfactualTaskSuite
from external_failure_adapter import normalize_failure_record
from process_credit import ProcessRewardConfig, score_rollout_trajectory
from terminal_reward import terminal_result_reward
from trajectory_replay import evaluate_counterfactual_suite


TurnTokens = tuple[list[int], list[int]]


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
) -> RolloutSample:
    """Attach terminal/process credit without depending on a trainer implementation."""
    record = env.record()
    record["trajectory_id"] = f"rl_{metadata['example_index']}_sample_{sample_index}"
    step_rewards = None
    process_update = record["failure_type"] != "generation_oom"
    scalar_reward = terminal_result_reward(record["correct"])
    if not process_update:
        record["optimization_exclusion"] = "nonsemantic_runtime_failure"

    if reward_mode == "process":
        if process_config is None:
            raise ValueError("process reward mode requires a ProcessRewardConfig")
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

    return RolloutSample(
        reward=scalar_reward,
        correct=bool(record["correct"]),
        failure_type=record["failure_type"],
        turns=turns,
        audit_record=record,
        step_rewards=step_rewards,
        process_update=process_update,
    )
