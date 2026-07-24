#!/usr/bin/env python3
"""Named denotation metrics shared by every SQL and tool-agent evaluator.

Candidate generation/aggregation (greedy, sampling, pass@k) is intentionally owned by the
individual runner and ``passk.py``.  This module owns only the orthogonal question of whether two
executed result sets are equal.  Keeping the registry here makes the metric explicit in CLIs and
artifacts without coupling evaluation semantics to the model-visible SFT protocol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


Comparator = Callable[[object, object], bool]


def _cell(value) -> str:
    """Canonical strict-multiset cell representation used by historical project scoring."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.4f}"


def normalize_rows(rows) -> list[tuple]:
    """Normalize cells and sort rows while preserving duplicate multiplicity."""
    return sorted(tuple(_cell(cell) for cell in row) for row in rows)


def rows_equal(predicted, gold) -> bool:
    """Normalized strict multiset equality; row order is ignored, duplicates are preserved."""
    return normalize_rows(predicted) == normalize_rows(gold)


def bird_rows_equal(predicted, gold) -> bool:
    """Official BIRD EX equality: ignore row order and duplicate-row multiplicity."""
    return {tuple(row) for row in predicted} == {tuple(row) for row in gold}


@dataclass(frozen=True)
class DenotationMetric:
    name: str
    description: str
    compare: Comparator


_METRICS = {
    "strict-multiset": DenotationMetric(
        name="strict-multiset",
        description=(
            "normalized multiset equality; ignores row order and preserves duplicate multiplicity"
        ),
        compare=rows_equal,
    ),
    "bird-set": DenotationMetric(
        name="bird-set",
        description=(
            "BIRD reference EX set equality; ignores row order and duplicate multiplicity"
        ),
        compare=bird_rows_equal,
    ),
}

DENOTATION_COMPARISONS = tuple(_METRICS)
# Current/new episodes use the BIRD reference EX contract everywhere.  The strict metric remains
# registered for immutable historical artifacts and explicit compatibility audits only.
DEFAULT_DENOTATION_COMPARISON = "bird-set"


def get_denotation_metric(name: str) -> DenotationMetric:
    """Resolve a metric by its manifest-safe name."""
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown denotation comparison {name!r}; expected one of {DENOTATION_COMPARISONS}"
        ) from exc


def compare_denotations(
    predicted,
    gold,
    comparison: str = DEFAULT_DENOTATION_COMPARISON,
) -> bool:
    """Compare executed query results under an explicit evaluation contract."""
    return get_denotation_metric(comparison).compare(predicted, gold)


def add_denotation_comparison_argument(parser, *, default: str = DEFAULT_DENOTATION_COMPARISON):
    """Add the current BIRD CLI contract.

    Historical metric implementations stay registered so immutable artifacts can be replayed by
    explicit compatibility code, but active evaluation launchers cannot accidentally start a new
    non-BIRD run.
    """
    get_denotation_metric(default)
    if default != DEFAULT_DENOTATION_COMPARISON:
        raise ValueError("active evaluation entry points must default to bird-set")
    return parser.add_argument(
        "--denotation-comparison",
        choices=(DEFAULT_DENOTATION_COMPARISON,),
        default=default,
        help="executed-result comparison contract; fixed to BIRD reference EX set equality",
    )
