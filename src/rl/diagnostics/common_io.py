"""Shared, side-effect-free I/O helpers for RL diagnostics.

Diagnostic entry points used to each carry their own JSONL, digest and reward-config
loader.  Keeping these primitives here makes the entry points about the audit they
perform while preserving identical validation and hashing semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

try:  # direct script execution (PYTHONPATH=src/rl)
    from rl.shared.io import read_jsonl, sha256_file
    from rl.shared.stats import percentile_nearest_rank
except ModuleNotFoundError:  # direct execution with src/rl itself on sys.path
    from shared.io import read_jsonl, sha256_file
    from shared.stats import percentile_nearest_rank

if TYPE_CHECKING:
    from rl.objectives.process_credit import ProcessRewardConfig


percentile = percentile_nearest_rank


def load_process_reward_config(
    path: Path | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> "ProcessRewardConfig":
    """Load and validate a process-reward config, ignoring manifest metadata keys."""
    values: dict[str, Any] = dict(defaults or {})
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        values.update({key: value for key, value in payload.items() if not key.startswith("_")})
    # Keep diagnostics importable in lightweight environments; process-credit
    # dependencies (sqlglot/Harness adapters) are needed only when loading a
    # process-reward config.
    from rl.objectives.process_credit import ProcessRewardConfig

    config = ProcessRewardConfig(**values)
    config.validate()
    return config
