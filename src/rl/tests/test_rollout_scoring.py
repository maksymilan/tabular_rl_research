from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path


RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from rollout_scoring import score_completed_rollout  # noqa: E402
from frameworks.trl.transition_batch import (  # noqa: E402
    PolicyEpisode,
    PolicyTurn,
    build_transition_updates,
)


class _RecordedEnvironment:
    def __init__(self, record: dict) -> None:
        self._record = deepcopy(record)

    def record(self) -> dict:
        return deepcopy(self._record)


def _score(record: dict, *, generation_truncation: dict | None = None):
    return score_completed_rollout(
        _RecordedEnvironment(record),
        [([1], [2])],
        {
            "task_id": "task-1",
            "example_index": 1,
            "db_id": "db",
            "db_path": "/unused.sqlite",
            "question": "question",
            "gold_sql": "select 1",
            "external_knowledge": None,
        },
        sample_index=0,
        reward_mode="result-only",
        process_config=None,
        process_admission_policy="counterfactual-completeness",
        denotation_comparison="bird-set",
        counterfactual_suite=None,
        result_reward_profile="binary",
        generation_truncation=generation_truncation,
    )


def _record(
    *,
    correct: bool = False,
    legal: bool = False,
    failure_type: str | None = None,
    error_events: list[dict] | None = None,
    turns: list[dict] | None = None,
) -> dict:
    return {
        "example_index": 1,
        "correct": correct,
        "legal": legal,
        "failure_type": failure_type,
        "error_events": error_events or [],
        "turns": turns or [],
    }


def test_context_overflow_is_excluded_from_policy_optimization() -> None:
    sample = _score(_record(failure_type="context_overflow"))

    assert sample.reward == 0.0
    assert not sample.process_update
    assert sample.audit_record["process_update"] is False
    assert sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )
    episode = PolicyEpisode(
        sample=sample,
        policy_turns=[
            PolicyTurn(
                prompt_ids=(1,),
                response_ids=(2,),
                sampling_logprobs=(-0.1,),
            )
        ],
    )
    assert build_transition_updates([episode], reward_mode="result-only") == []


def test_terminal_timeout_is_excluded_without_structured_event() -> None:
    sample = _score(_record(failure_type="timeout_error"))

    assert sample.reward == 0.0
    assert not sample.process_update
    assert sample.audit_record["process_update"] is False
    assert sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )


def test_recovered_correct_timeout_is_excluded_by_structured_evidence() -> None:
    timeout_event = {
        "error_type": "timeout_error",
        "error_code": "tool_execution_timeout",
        "details": {"state_preserved": True},
    }
    sample = _score(
        _record(
            correct=True,
            legal=True,
            failure_type=None,
            error_events=[timeout_event],
        )
    )

    assert sample.correct
    assert sample.audit_record["legal"]
    assert sample.audit_record["error_events"] == [timeout_event]
    assert sample.reward == 0.0
    assert not sample.process_update
    assert sample.audit_record["process_update"] is False
    assert sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )


def test_turn_local_timeout_evidence_is_fail_closed() -> None:
    sample = _score(
        _record(
            turns=[
                {
                    "execution_error_type": "timeout_error",
                    "error_event": {"error_code": "tool_execution_timeout"},
                }
            ]
        )
    )

    assert not sample.process_update
    assert sample.audit_record["process_update"] is False
    assert sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )


def test_ordinary_model_execution_failure_remains_trainable() -> None:
    event = {
        "error_type": "execution_error",
        "error_code": "sqlite_operational_error",
    }
    sample = _score(
        _record(failure_type="execution_error", error_events=[event])
    )

    assert sample.reward == 0.0
    assert sample.process_update
    assert sample.audit_record["process_update"] is True
    assert "optimization_exclusion" not in sample.audit_record


def test_generation_length_persists_false_optimizer_admission() -> None:
    truncation = {"finish_reason": "length", "turn_index": 3}
    sample = _score(
        _record(failure_type="generation_length"),
        generation_truncation=truncation,
    )

    assert sample.reward == 0.0
    assert sample.process_update is False
    assert sample.audit_record["process_update"] is False
    assert sample.audit_record["generation_truncation"] == truncation
    assert sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )
