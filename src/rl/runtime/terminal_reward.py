#!/usr/bin/env python3
"""Framework-independent terminal result rewards."""
from __future__ import annotations


RESULT_REWARD_PROFILES = {"binary", "execution-ladder", "four-level"}


def terminal_result_reward(
    correct: object,
    *,
    executable: object = False,
    profile: str = "binary",
    has_errors: object = False,
) -> float:
    """Score terminal correctness while preserving the historical binary control.

    ``execution-ladder`` is the TRUST-SQL-style result-only reward translated to the
    table-agent environment: a verified correct terminal answer receives 1.0, a legal
    terminal answer whose cited relation executes but has the wrong denotation receives
    0.2, and trajectories without a legal terminal answer receive 0.0.

    ``four-level`` preserves the terminal result sign while adding one deterministic
    trajectory-quality bit: correct/error-free = 1.5, correct/with-error = 1.0,
    incorrect/error-free = -0.5, and incorrect/with-error = -1.0.  The error bit is
    supplied by the rollout scorer from structured Harness evidence, including tool
    timeout and an incomplete response that reaches the generation-length limit; it
    is not inferred from model-authored reasoning prose.  In the final SAAM
    asymmetric-error route, a timeout is also a policy-visible penalty action at
    the timed-out transition, even when a later recovery reaches a correct answer.
    """
    if profile not in RESULT_REWARD_PROFILES:
        raise ValueError(f"unsupported result reward profile: {profile}")
    if profile == "four-level":
        return (1.0 if bool(has_errors) else 1.5) if bool(correct) else (
            -1.0 if bool(has_errors) else -0.5
        )
    if bool(correct):
        return 1.0
    if profile == "execution-ladder" and bool(executable):
        return 0.2
    return 0.0
