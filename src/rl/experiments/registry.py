"""Named experiment definitions and current-line admission checks.

Scenario launchers should resolve an experiment through this module instead of
embedding a second copy of its identity and optimization settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rl.configuration.experiment_config import RLExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments"

ACTIVE_EXPERIMENT = "qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml"


def config_path(name: str) -> Path:
    path = CONFIG_ROOT / name
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"experiment config must be YAML: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load(name: str = ACTIVE_EXPERIMENT) -> RLExperimentConfig:
    return RLExperimentConfig.load(config_path(name))


def active_defaults() -> dict[str, Any]:
    """Return the current contract's normalized trainer defaults."""
    return load().argparse_defaults(ROOT)


def validate_active_contract() -> dict[str, Any]:
    """Fail closed if the checked-in active config drifts from the final contract."""
    config = load()
    defaults = config.argparse_defaults(ROOT)
    expected = {
        "reward_mode": "result-only",
        "result_reward_profile": "four-level",
        "credit_assignment": "saam-asymmetric-error",
        "policy_reduction": "trajectory_token_mean",
        "group_size": 8,
        "prompts_per_update": 30,
        "optimizer_steps": 200,
        "learning_rate": 4e-7,
        "kl_beta": 0.0,
    }
    mismatches = {
        key: (defaults.get(key), value)
        for key, value in expected.items()
        if defaults.get(key) != value
    }
    if mismatches:
        raise ValueError(f"active experiment contract drift: {mismatches}")
    return defaults
