"""Deterministic teacher routing from the frozen eligibility prepass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TeacherRoute:
    dense_candidates: tuple[str, ...]
    branch_candidates: tuple[str, ...]
    preservation_scale: float


def route_teachers(
    eligibility_row: Mapping,
    *,
    student_correct: bool,
    correct_preservation_scale: float = 0.1,
) -> TeacherRoute:
    teachers = eligibility_row["teachers"]
    dense = tuple(
        teacher
        for teacher in ("sft2", "exp15")
        if bool(teachers[teacher]["dense_eligible"])
    )
    branch = tuple(
        teacher
        for teacher in ("sft2", "exp15")
        if bool(teachers[teacher]["branch_eligible"])
    )
    if student_correct:
        # Correct student behavior is retained gently, not overwritten.  Prefer
        # the original SFT2 teacher when both are reliable.
        chosen = ("sft2",) if "sft2" in dense else dense[:1]
        return TeacherRoute(chosen, (), correct_preservation_scale if chosen else 0.0)
    return TeacherRoute(dense, branch, 1.0)


def choose_dense_teacher(
    candidates: tuple[str, ...],
    mean_teacher_logprobs: Mapping[str, float],
    *,
    selected_repair_teacher: str | None = None,
) -> str | None:
    if selected_repair_teacher in candidates:
        return selected_repair_teacher
    if not candidates:
        return None
    return max(candidates, key=lambda name: (mean_teacher_logprobs[name], name == "sft2"))
