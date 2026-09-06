"""Deterministic statistics shared by offline RL reports."""
from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile for reports whose historical contract uses ceil(q*n)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
