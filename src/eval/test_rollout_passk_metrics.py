from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_passk import (
    attach_trajectory_generation_stats,
    completion_generation_stats,
)


def test_completion_generation_stats_reduces_logprobs_and_tail_entropy() -> None:
    choice = {
        "logprobs": {
            "content": [
                {
                    "token": "a",
                    "logprob": math.log(0.6),
                    "top_logprobs": [
                        {"token": "a", "logprob": math.log(0.6)},
                        {"token": "b", "logprob": math.log(0.3)},
                    ],
                },
                {
                    "token": "c",
                    "logprob": math.log(0.8),
                    "top_logprobs": [
                        {"token": "c", "logprob": math.log(0.8)},
                    ],
                },
            ]
        }
    }

    stats = completion_generation_stats(choice, requested_top_logprobs=20)

    assert stats["completion_token_count"] == 2
    assert math.isclose(
        stats["action_logprob"],
        math.log(0.6) + math.log(0.8),
    )
    expected_entropy = (
        -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    ) / 2
    assert math.isclose(
        stats["mean_token_entropy_lower_bound"],
        expected_entropy,
    )


def test_trajectory_generation_stats_are_token_weighted() -> None:
    record = {
        "turns": [
            {
                "generation_stats": {
                    "completion_token_count": 2,
                    "action_logprob": -2.0,
                    "mean_token_entropy_lower_bound": 0.5,
                }
            },
            {
                "generation_stats": {
                    "completion_token_count": 1,
                    "action_logprob": -3.0,
                    "mean_token_entropy_lower_bound": 2.0,
                }
            },
        ]
    }

    attach_trajectory_generation_stats(record)

    stats = record["generation_stats"]
    assert stats["turn_count"] == 2
    assert stats["completion_token_count"] == 3
    assert stats["trajectory_action_logprob"] == -5.0
    assert math.isclose(stats["mean_token_surprisal"], 5.0 / 3.0)
    assert math.isclose(stats["mean_token_entropy_lower_bound"], 1.0)
