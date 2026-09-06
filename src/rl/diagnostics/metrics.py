"""Small, deterministic statistics used by RL diagnostic reports.

Scenario scripts should keep experiment-specific interpretation and report
formatting.  This module owns the repeated numerical conventions: linear
percentiles, tie-aware ROC AUC, paired exact tests, distributions, and seeded
bootstrap intervals.  Functions accept ordinary Python iterables so they can
be reused by JSONL audits without pulling in a data-frame or scipy dependency.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar


T = TypeVar("T")


def numeric_values(values: Iterable[Any]) -> list[float]:
    """Convert a numeric iterable to floats, rejecting missing observations."""

    result: list[float] = []
    for value in values:
        if value is None:
            continue
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"diagnostic metric received non-finite value: {value!r}")
        result.append(converted)
    return result


def mean(values: Iterable[Any], *, default: float = 0.0) -> float:
    """Return the arithmetic mean, using ``default`` for no observations."""

    observed = numeric_values(values)
    return statistics.fmean(observed) if observed else float(default)


def percentile(values: Iterable[Any], fraction: float, *, default: float = 0.0) -> float:
    """Return a linearly interpolated percentile in the closed unit interval."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(numeric_values(values))
    if not ordered:
        return float(default)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def distribution(values: Iterable[Any]) -> dict[str, float]:
    """Summarize a numeric sequence using the report-wide percentile names."""

    observed = numeric_values(values)
    return {
        "mean": mean(observed),
        "median": statistics.median(observed) if observed else 0.0,
        "p75": percentile(observed, 0.75),
        "p90": percentile(observed, 0.90),
        "max": max(observed, default=0.0),
    }


def auc(positive: Sequence[Any], negative: Sequence[Any]) -> float | None:
    """Compute tie-aware ROC AUC as the positive-vs-negative Mann–Whitney rate."""

    positives = numeric_values(positive)
    negatives = numeric_values(negative)
    if not positives or not negatives:
        return None
    wins = 0.0
    for left in positives:
        for right in negatives:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def exact_sign_p(improved: int, regressed: int) -> float:
    """Two-sided exact sign-test p-value for paired non-tied observations."""

    if improved < 0 or regressed < 0:
        raise ValueError("improved and regressed counts must be non-negative")
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def exact_mcnemar_p(gains: int, regressions: int) -> float:
    """Alias of the paired exact sign test, named for binary outcomes."""

    return exact_sign_p(gains, regressions)


def paired_delta_summary(deltas: Iterable[Any]) -> dict[str, float | int]:
    """Summarize paired deltas and test their non-tied sign balance."""

    observed = numeric_values(deltas)
    improved = sum(value > 0 for value in observed)
    regressed = sum(value < 0 for value in observed)
    return {
        "improved": improved,
        "regressed": regressed,
        "tied": sum(value == 0 for value in observed),
        "delta_sum": sum(observed),
        "mean_delta": mean(observed),
        "exact_two_sided_sign_p": exact_sign_p(improved, regressed),
    }


def js_divergence_bits(left: Mapping[Any, int | float], right: Mapping[Any, int | float]) -> float:
    """Compute Jensen–Shannon divergence in bits for two count mappings."""

    left_total = sum(float(value) for value in left.values())
    right_total = sum(float(value) for value in right.values())
    if left_total < 0 or right_total < 0:
        raise ValueError("distribution counts must be non-negative")
    result = 0.0
    for key in sorted(set(left) | set(right), key=str):
        p = float(left.get(key, 0.0)) / left_total if left_total else 0.0
        q = float(right.get(key, 0.0)) / right_total if right_total else 0.0
        midpoint = (p + q) / 2.0
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def bootstrap_ci(
    positive: Sequence[T],
    negative: Sequence[T],
    statistic: Callable[[list[T], list[T]], float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Return a seeded percentile bootstrap interval for a two-sample statistic."""

    if not positive or not negative:
        raise ValueError("bootstrap samples require non-empty positive and negative groups")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    values = [
        statistic(
            [positive[rng.randrange(len(positive))] for _ in positive],
            [negative[rng.randrange(len(negative))] for _ in negative],
        )
        for _ in range(samples)
    ]
    return {"p2_5": percentile(values, 0.025), "p97_5": percentile(values, 0.975)}


def cohens_d_pooled(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    """Return pooled Cohen's d, or ``None`` when either sample lacks variance."""

    first, second = numeric_values(left), numeric_values(right)
    if len(first) < 2 or len(second) < 2:
        return None
    pooled_variance = (
        (len(first) - 1) * statistics.variance(first)
        + (len(second) - 1) * statistics.variance(second)
    ) / (len(first) + len(second) - 2)
    pooled = math.sqrt(pooled_variance)
    return (mean(first) - mean(second)) / pooled if pooled else None


__all__ = [
    "auc",
    "bootstrap_ci",
    "cohens_d_pooled",
    "distribution",
    "exact_mcnemar_p",
    "exact_sign_p",
    "js_divergence_bits",
    "mean",
    "numeric_values",
    "paired_delta_summary",
    "percentile",
]
