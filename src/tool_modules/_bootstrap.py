"""Temporary import bridge while shared legacy layers remain flat modules."""
from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent


def activate_legacy_paths() -> None:
    """Expose shared eval/harness/SFT modules without coupling schemes to each other."""
    for path in (
        SRC_ROOT / "eval",
        SRC_ROOT / "harness",
        SRC_ROOT / "sft",
        SRC_ROOT / "rl",
    ):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.append(rendered)

