#!/usr/bin/env python3
"""Framework-independent terminal-{0,1} reward control."""
from __future__ import annotations


def terminal_result_reward(correct: object) -> float:
    """Return one for a correct terminal denotation and zero otherwise."""
    return 1.0 if bool(correct) else 0.0
