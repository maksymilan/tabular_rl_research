from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from compare_process_reward_variants import compare, summarize


def _row(reward: float, *, error: bool = False) -> dict:
    return {
        "trajectory_id": "t",
        "correct": True,
        "total_reward": reward,
        "steps": [
            {
                "step_id": "step_1",
                "tool": "join_tables",
                "reward": reward,
                "c_positive": max(reward, 0.0),
                "features": {
                    "error_type": "execution_error" if error else None,
                    "tool_error": float(error),
                    "back_slice": 1.0,
                },
            }
        ],
    }


def test_comparison_counts_removed_positive_credit() -> None:
    result = compare([_row(1.0)], [_row(0.0)])
    assert result["changed_step_count"] == 1
    assert result["positive_to_nonpositive"] == 1
    assert result["nonpositive_to_positive"] == 0


def test_summary_exposes_positive_reward_on_error_steps() -> None:
    summary = summarize([_row(0.5, error=True)])
    assert summary["error_steps"]["count"] == 1
    assert summary["error_steps"]["positive_reward_count"] == 1
    assert summary["error_steps"]["back_slice_count"] == 1
