from __future__ import annotations

from src.rl.distillation.routing import choose_dense_teacher, route_teachers


def row(sft2_dense, sft2_branch, exp15_dense, exp15_branch):
    return {
        "teachers": {
            "sft2": {
                "dense_eligible": sft2_dense,
                "branch_eligible": sft2_branch,
            },
            "exp15": {
                "dense_eligible": exp15_dense,
                "branch_eligible": exp15_branch,
            },
        }
    }


def test_correct_student_gets_weak_sft2_preservation_not_repair():
    route = route_teachers(row(True, True, True, True), student_correct=True)
    assert route.dense_candidates == ("sft2",)
    assert route.branch_candidates == ()
    assert route.preservation_scale == 0.1


def test_wrong_student_can_use_both_frozen_teachers():
    route = route_teachers(row(True, True, False, True), student_correct=False)
    assert route.dense_candidates == ("sft2",)
    assert route.branch_candidates == ("sft2", "exp15")
    assert route.preservation_scale == 1.0


def test_verified_repair_teacher_overrides_logprob_tie_break():
    chosen = choose_dense_teacher(
        ("sft2", "exp15"),
        {"sft2": -0.5, "exp15": -1.0},
        selected_repair_teacher="exp15",
    )
    assert chosen == "exp15"
