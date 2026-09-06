from rl.scenarios.diagnostics.score_frozen_transition_policy import enrich_rows, reward_category, summarize


def _transition(correct: bool, advantage: float = 0.5):
    return {
        "trajectory_id": "t",
        "turn_index": 0,
        "trajectory_correct": correct,
        "advantage": advantage,
        "prompt_ids": [1],
        "response_ids": [2],
    }


def _feature(**overrides):
    features = {
        "dense_severe_local_bad_event": False,
        "dense_operator_backslice_bonus": False,
        "dense_observation_support_bonus": False,
        "is_terminal": False,
        "dense_raw_action_credit": 1.0,
    }
    features.update(overrides)
    return {
        "trajectory_id": "t",
        "step_index": 0,
        "tool": "project",
        "features": features,
    }


def test_reward_category_priority() -> None:
    assert reward_category(
        _transition(True),
        _feature(dense_severe_local_bad_event=True, dense_operator_backslice_bonus=True),
    ) == "severe_local_error"
    assert reward_category(
        _transition(True), _feature(dense_operator_backslice_bonus=True)
    ) == "correct_key_backslice"
    assert reward_category(_transition(False), _feature()) == "incorrect_other_clean"


def test_enrich_and_summary_reward_alignment() -> None:
    row = enrich_rows([_transition(True)], [_feature()])[0]
    scored = {
        **row,
        "response_tokens": 1,
        "scores": {
            "base": {
                "response_mean_logprob": -1.0,
                "response_sum_logprob": -1.0,
                "tool_mean_logprob": -1.0,
            },
            "new": {
                "response_mean_logprob": -0.8,
                "response_sum_logprob": -0.8,
                "tool_mean_logprob": -0.7,
            },
        },
    }
    result = summarize([scored], "base", ["new"])["all"]["checkpoints"]["new"]
    assert result["reward_aligned_fraction"] == 1.0
    assert abs(result["reward_alignment_mean"] - 0.1) < 1e-8
