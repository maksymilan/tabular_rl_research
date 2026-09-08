from __future__ import annotations

import pytest

from rl.frameworks.trl.iql import (
    advantage_weight,
    discounted_terminal_returns,
    expectile_loss_value,
    td_target,
)
from rl.frameworks.trl.mdp_critic import audit_transitions, build_transitions


def _row(example_index: int, sample_index: int, *, correct: bool, tool: str) -> dict:
    return {
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "example_index": example_index,
        "db_id": "db",
        "trajectory_id": f"traj_{example_index}_sample_{sample_index}",
        "correct": correct,
        "legal": True,
        "errors": 0,
        "error_events": [],
        "result_reward": {
            "profile": "four-level",
            "value": 1.5 if correct else -0.5,
        },
        "turns": [
            {
                "model_input": [{"role": "user", "content": "question"}],
                "parsed": {"tool": tool, "arguments": {"table": "T"}},
            },
            {
                "model_input": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "action"},
                ],
                "parsed": {"tool": "answer_from_context", "arguments": {"evidence": {}}},
            },
        ],
    }


def test_iql_scalar_equations() -> None:
    assert expectile_loss_value(0.0, 2.0, 0.7) == pytest.approx(2.8)
    assert expectile_loss_value(2.0, 0.0, 0.7) == pytest.approx(1.2)
    assert td_target(1.0, 10.0, True, 0.99) == 1.0
    assert td_target(0.0, 10.0, False, 0.99) == pytest.approx(9.9)
    assert discounted_terminal_returns([0.0, 1.5], [False, True], gamma=1.0) == [1.5, 1.5]
    assert advantage_weight(
        2.0,
        1.0,
        inverse_temperature=1.0,
        max_weight=20.0,
    ) == pytest.approx(2.7182818)


def test_transition_builder_uses_harness_reward_and_strips_gold() -> None:
    row = _row(7, 0, correct=True, tool="describe_table")
    row["gold_sql"] = "must never be copied"
    transitions, issues = build_transitions([row])
    assert not issues
    assert len(transitions) == 2
    assert transitions[-1].reward == pytest.approx(1.5)
    assert transitions[0].reward == 0.0
    assert transitions[0].next_state_hash == transitions[1].state_hash
    assert "gold_sql" not in transitions[0].to_dict()


def test_audit_requires_repeated_state_action_outcome_variation() -> None:
    rows = [
        _row(7, 0, correct=True, tool="describe_table"),
        _row(7, 1, correct=False, tool="project"),
    ]
    transitions, issues = build_transitions(rows)
    result = audit_transitions(transitions, issues)
    assert result["observed"]["states_with_multiple_actions"] == 1
    assert result["observed"]["states_with_mixed_outcomes"] == 1
    assert result["gate"]["critic_is_not_proven_useful"] is False


def test_forbidden_reference_inside_model_state_fails_closed() -> None:
    row = _row(7, 0, correct=True, tool="describe_table")
    row["turns"][0]["model_input"][0]["gold_sql"] = "secret"
    transitions, issues = build_transitions([row])
    assert transitions == []
    assert "forbidden reference" in issues[0]["reason"]


def test_policy_step_is_part_of_episode_identity_when_raw_id_is_reused() -> None:
    first = _row(7, 0, correct=True, tool="describe_table")
    second = _row(7, 0, correct=False, tool="project")
    first["policy_global_step"] = 1
    first["policy_micro_step"] = 1
    second["policy_global_step"] = 2
    second["policy_micro_step"] = 1
    transitions, issues = build_transitions([first, second])
    assert not issues
    result = audit_transitions(transitions)
    assert result["observed"]["trajectories"] == 2
    assert result["observed"]["terminal_transitions"] == 2
