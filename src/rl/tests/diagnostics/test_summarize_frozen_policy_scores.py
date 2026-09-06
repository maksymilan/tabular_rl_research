from rl.scenarios.diagnostics.summarize_frozen_policy_scores import pearson, percentile, summarize


def test_percentile_and_pearson() -> None:
    assert percentile([0.0, 1.0, 2.0], 0.5) == 1.0
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_reward_direction_summary() -> None:
    rows = [
        {
            "trajectory_id": "a",
            "turn_index": 0,
            "advantage": 1.0,
            "reward_category": "correct_other_clean",
            "scores": {
                "base": {"response_mean_logprob": -1.0, "tool_mean_logprob": -1.0},
                "new": {"response_mean_logprob": -0.5, "tool_mean_logprob": -0.5},
            },
        },
        {
            "trajectory_id": "b",
            "turn_index": 0,
            "advantage": -1.0,
            "reward_category": "incorrect_other_clean",
            "scores": {
                "base": {"response_mean_logprob": -1.0, "tool_mean_logprob": None},
                "new": {"response_mean_logprob": -1.5, "tool_mean_logprob": None},
            },
        },
    ]
    result = summarize(rows, ["new"], "base")["checkpoints"]["new"]
    assert result["reward_direction"]["aligned_fraction"] == 1.0
    assert result["reward_direction"]["net_over_gross_alignment"] == 1.0
    assert result["groups"]["positive_reward"]["expected_direction_fraction"] == 1.0
    assert result["groups"]["negative_reward"]["expected_direction_fraction"] == 1.0
