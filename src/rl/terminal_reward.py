#!/usr/bin/env python3
"""Framework-independent terminal result rewards."""
from __future__ import annotations


RESULT_REWARD_PROFILES = {"binary", "execution-ladder"}


def terminal_result_reward(
    correct: object,
    *,
    executable: object = False,
    profile: str = "binary",
) -> float:
    """Score terminal correctness while preserving the historical binary control.

    ``execution-ladder`` is the TRUST-SQL-style result-only reward translated to the
    table-agent environment: a verified correct terminal answer receives 1.0, a legal
    terminal answer whose cited relation executes but has the wrong denotation receives
    0.2, and trajectories without a legal terminal answer receive 0.0.
    """
    if profile not in RESULT_REWARD_PROFILES:
        raise ValueError(f"unsupported result reward profile: {profile}")
    if bool(correct):
        return 1.0
    if profile == "execution-ladder" and bool(executable):
        return 0.2
    return 0.0
