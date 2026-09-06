"""Resolve evaluation contracts without tracking JSON in the source tree."""
from __future__ import annotations

import os
from pathlib import Path


def default_contract(name: str) -> Path:
    """Return an explicit contract path supplied by the caller or archive.

    Contracts are run artifacts and are intentionally stored outside
    ``src/rl``.  ``ATOMIC_V26_EVAL_CONTRACT`` can point at a generated contract
    for a new run; the archive fallback is only for historical replay.
    """
    override = os.environ.get("ATOMIC_V26_EVAL_CONTRACT")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[4] / "archive/config/evaluation_contracts/20260906" / name

