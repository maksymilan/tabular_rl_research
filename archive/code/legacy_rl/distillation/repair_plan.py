"""Parallel 3/4, 1/2, 1/4, and start repair planning."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable


def parallel_repair_anchor_turns(nonterminal_turn_count: int) -> tuple[int, ...]:
    """Return unique exact-prefix anchors in latest-to-earliest order.

    An anchor value is the number of already executed nonterminal model turns.
    The four values are planned together and may be generated in one batched
    teacher call.  Short trajectories naturally deduplicate.
    """
    if nonterminal_turn_count < 0:
        raise ValueError("nonterminal_turn_count cannot be negative")
    if nonterminal_turn_count == 0:
        return (0,)
    candidates = (
        floor(3 * nonterminal_turn_count / 4),
        floor(nonterminal_turn_count / 2),
        floor(nonterminal_turn_count / 4),
        0,
    )
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class RepairCandidate:
    teacher: str
    anchor_turn: int
    trials: int
    correct: int
    legal: int
    total_errors: int
    mean_steps: float
    alignment: float = 0.0

    def __post_init__(self) -> None:
        if self.anchor_turn < 0 or self.trials < 1:
            raise ValueError("repair candidate has an invalid anchor or trial count")
        if not (0 <= self.correct <= self.legal <= self.trials):
            raise ValueError("repair candidate correct/legal counts are inconsistent")
        if self.total_errors < 0 or self.mean_steps < 0:
            raise ValueError("repair diagnostics cannot be negative")


@dataclass(frozen=True)
class RepairSelection:
    strength: str
    candidate: RepairCandidate | None

    @property
    def supports_dpo(self) -> bool:
        return self.strength == "strong" and self.candidate is not None


def _candidate_order(candidate: RepairCandidate) -> tuple:
    return (
        -candidate.anchor_turn,
        -candidate.correct / candidate.trials,
        candidate.total_errors,
        candidate.mean_steps,
        -candidate.alignment,
        candidate.teacher,
    )


def select_verified_repair(
    candidates: Iterable[RepairCandidate],
    *,
    minimum_strong_successes: int = 2,
) -> RepairSelection:
    """Choose the latest stable repair; expose one-off repairs as weak only."""
    if minimum_strong_successes < 2:
        raise ValueError("a stable repair must require at least two successes")
    rows = list(candidates)
    strong = [row for row in rows if row.correct >= minimum_strong_successes]
    if strong:
        return RepairSelection("strong", min(strong, key=_candidate_order))
    weak = [row for row in rows if row.correct >= 1]
    if weak:
        return RepairSelection("weak", min(weak, key=_candidate_order))
    return RepairSelection("none", None)


def normalized_group_branch_scales(
    branch_counts: Iterable[int],
    *,
    total_scale: float,
) -> tuple[tuple[float, ...], ...]:
    """Split one task's repair weight equally over groups, then branches.

    A task with many verified repair anchors must not receive a larger total
    branch-DPO weight than a task with one verified anchor.  Within a task,
    every strong teacher-anchor group receives equal mass and divides that
    mass equally over its retained correct continuations.
    """
    counts = tuple(int(value) for value in branch_counts)
    if total_scale < 0.0:
        raise ValueError("repair total scale cannot be negative")
    if any(value < 1 for value in counts):
        raise ValueError("every repair group must retain at least one branch")
    if not counts:
        return ()
    group_scale = total_scale / len(counts)
    return tuple(
        tuple(group_scale / count for _ in range(count))
        for count in counts
    )
