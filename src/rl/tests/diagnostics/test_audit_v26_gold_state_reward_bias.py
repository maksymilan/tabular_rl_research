#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RL_DIR = HERE.parent
SRC_DIR = RL_DIR.parent.parent
sys.path[:0] = [str(HERE), str(RL_DIR), str(SRC_DIR)]

from rl.scenarios.diagnostics.audit_v26_gold_state_reward_bias import (  # noqa: E402
    allocate_causal_state_delta,
    summarize_variant,
)
from rl.objectives.process_credit import ProcessRewardConfig, StepFeature  # noqa: E402


def feature(index: int, **kwargs) -> StepFeature:
    return StepFeature(
        action_index=index,
        step_id=f"step_{index}",
        tool=kwargs.pop("tool", "condition_filter"),
        legal_success=kwargs.pop("legal_success", True),
        **kwargs,
    )


def test_causal_state_delta_ignores_progress_off_the_success_dependency_slice():
    config = ProcessRewardConfig()
    allocation = allocate_causal_state_delta(
        [
            feature(1, tool="describe_table", target_table_delta=1.0),
            feature(
                2,
                tool="condition_filter",
                target_column_delta=1.0,
                back_slice=1.0,
            ),
            feature(
                3,
                tool="project",
                target_row_delta=1.0,
                back_slice=1.0,
            ),
            feature(4, tool="answer_from_context", is_terminal=True),
        ],
        correct=True,
        config=config,
    )
    assert allocation.raw_progress == (0.25, 0.25, 0.5, 0.0)
    assert allocation.eligible_progress == (0.0, 0.25, 0.5, 0.0)
    assert allocation.positive_allocations == (0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0)
    assert abs(allocation.total_reward - 1.0) < 1e-9


def test_failed_trajectory_stays_negative_despite_complete_target_progress():
    config = ProcessRewardConfig()
    allocation = allocate_causal_state_delta(
        [
            feature(1, target_table_delta=1.0),
            feature(2, target_column_delta=1.0),
            feature(3, target_row_delta=1.0),
            feature(4, tool="answer_from_context", is_terminal=True),
        ],
        correct=False,
        config=config,
    )
    assert abs(sum(allocation.raw_progress) - 1.0) < 1e-9
    assert abs(allocation.total_reward - (0.05 - 0.30)) < 1e-9
    assert allocation.total_reward < 0


def test_state_conditioned_residual_does_not_center_by_tool_identity():
    rows = [
        {
            "trajectory_id": "a",
            "correct": True,
            "tool": "join_tables",
            "bucket": (True, True, False, 3, 1),
            "reward": 0.4,
            "positive": 0.4,
            "target_potential_delta": 0.25,
            "eligible_target_potential_delta": 0.25,
        },
        {
            "trajectory_id": "b",
            "correct": True,
            "tool": "project",
            "bucket": (True, True, False, 3, 1),
            "reward": 0.2,
            "positive": 0.2,
            "target_potential_delta": 0.25,
            "eligible_target_potential_delta": 0.25,
        },
    ] * 20
    summary = summarize_variant(rows, "reward", "positive")
    assert summary["tools"]["join_tables"]["mean_state_conditioned_residual"] > 0
    assert summary["tools"]["project"]["mean_state_conditioned_residual"] < 0
    assert abs(summary["state_conditioned_residual_mean_range"] - 0.2) < 1e-9
