#!/usr/bin/env python3
"""Candidate aggregation strategies that are independent of denotation scoring."""
from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Real


PASS_K_AGGREGATION = "pass-k"
ARCTIC_MAJORITY_AGGREGATION = "arctic-majority"
CANDIDATE_AGGREGATIONS = (PASS_K_AGGREGATION, ARCTIC_MAJORITY_AGGREGATION)
_NUMERIC_NULL = object()


@dataclass(frozen=True)
class QueryResult:
    """Executed SQL result including output names required by Arctic soft similarity."""

    columns: tuple[str, ...]
    rows: tuple[tuple, ...]


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    scores: tuple[float, ...]


def _hashable(value):
    return value if isinstance(value, Hashable) else repr(value)


def _column_counts(result: QueryResult) -> dict[str, Counter]:
    """Approximate the dtype inference applied by pandas.read_sql_query before counting."""
    counts: dict[str, Counter] = {}
    for column_index, column_name in enumerate(result.columns):
        column_counts = counts.setdefault(column_name, Counter())
        values = [row[column_index] for row in result.rows]
        non_null = [value for value in values if value is not None]
        numeric_with_null = (
            len(non_null) != len(values)
            and bool(non_null)
            and all(isinstance(value, Real) for value in non_null)
        )
        for value in values:
            if numeric_with_null:
                value = _NUMERIC_NULL if value is None else float(value)
            column_counts[_hashable(value)] += 1
    return counts


def arctic_soft_result_similarity(
    left: QueryResult | None,
    right: QueryResult | None,
) -> float:
    """Reproduce Arctic's column-wise soft-denotation similarity without pandas."""
    if left is None or right is None or not left.rows or not right.rows:
        return 0.0
    # The pinned upstream pandas implementation returns zero whenever either result has duplicate
    # output labels, despite its attempted stacking workaround. Preserve that observable behavior.
    if len(set(left.columns)) != len(left.columns) or len(set(right.columns)) != len(
        right.columns
    ):
        return 0.0

    left_counts = _column_counts(left)
    right_counts = _column_counts(right)
    total_real_agreement = 0
    total_possible_agreement = 0

    for column_name in left_counts.keys() | right_counts.keys():
        left_column = left_counts.get(column_name)
        right_column = right_counts.get(column_name)
        if not left_column:
            total_possible_agreement += sum(right_column.values())
            continue
        if not right_column:
            total_possible_agreement += sum(left_column.values())
            continue

        values = left_column.keys() | right_column.keys()
        for value in values:
            left_frequency = left_column.get(value, 0)
            right_frequency = right_column.get(value, 0)
            # This mirrors the upstream np.nan branch after pandas promotes NULLs in otherwise
            # numeric columns to floating-point NaN.
            if value is _NUMERIC_NULL:
                left_frequency += right_frequency
                right_frequency = 0
            possible_agreement = max(left_frequency, right_frequency)
            total_possible_agreement += possible_agreement
            total_real_agreement += possible_agreement - abs(
                left_frequency - right_frequency
            )

    if total_possible_agreement == 0:
        return 0.0
    return total_real_agreement / total_possible_agreement


def select_arctic_majority(results: list[QueryResult | None]) -> CandidateSelection:
    """Select Arctic's soft-denotation medoid; ties resolve to the first candidate."""
    if not results:
        raise ValueError("Arctic majority voting requires at least one candidate")

    scores = [1.0 if result is not None else 0.0 for result in results]
    for left_index, left in enumerate(results):
        for right_index in range(left_index + 1, len(results)):
            similarity = arctic_soft_result_similarity(left, results[right_index])
            scores[left_index] += similarity
            scores[right_index] += similarity

    selected_index = max(range(len(scores)), key=scores.__getitem__)
    return CandidateSelection(selected_index=selected_index, scores=tuple(scores))


def add_candidate_aggregation_argument(parser):
    return parser.add_argument(
        "--candidate-aggregation",
        choices=CANDIDATE_AGGREGATIONS,
        default=PASS_K_AGGREGATION,
        help=(
            "top-level multi-candidate score: pass-k preserves existing pass@k behavior; "
            "arctic-majority reproduces Arctic's soft-denotation medoid selection"
        ),
    )
