from __future__ import annotations

from src.rl.distillation.repair_plan import (
    RepairCandidate,
    normalized_group_branch_scales,
    parallel_repair_anchor_turns,
    select_verified_repair,
)


def candidate(teacher, anchor, correct, *, errors=0, steps=3.0, alignment=0.0):
    return RepairCandidate(
        teacher=teacher,
        anchor_turn=anchor,
        trials=4,
        correct=correct,
        legal=max(correct, 3),
        total_errors=errors,
        mean_steps=steps,
        alignment=alignment,
    )


def test_four_anchor_states_are_parallel_plan_not_serial_search():
    assert parallel_repair_anchor_turns(8) == (6, 4, 2, 0)
    assert parallel_repair_anchor_turns(3) == (2, 1, 0)
    assert parallel_repair_anchor_turns(1) == (0,)
    assert parallel_repair_anchor_turns(0) == (0,)


def test_latest_stable_repair_wins_before_earlier_higher_pass_count():
    selection = select_verified_repair(
        [candidate("sft2", 6, 2), candidate("exp15", 4, 4)]
    )
    assert selection.supports_dpo
    assert selection.candidate.anchor_turn == 6


def test_one_of_four_is_weak_and_does_not_create_strong_negative():
    selection = select_verified_repair([candidate("exp15", 4, 1)])
    assert selection.strength == "weak"
    assert not selection.supports_dpo


def test_no_verified_continuation_returns_none():
    selection = select_verified_repair([candidate("sft2", 0, 0)])
    assert selection.strength == "none"
    assert selection.candidate is None


def test_repair_scale_is_task_balanced_then_group_balanced():
    scales = normalized_group_branch_scales((2, 1), total_scale=0.9)
    assert scales == ((0.225, 0.225), (0.45,))
    assert sum(sum(group) for group in scales) == 0.9
