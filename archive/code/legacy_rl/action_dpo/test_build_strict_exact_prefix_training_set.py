from build_strict_exact_prefix_training_set import (
    describe_table_reorder_only,
    stage_band,
    strict_gate_reason,
)


def _row(*, turn=2, positive_successes=4, trials=4, negative_successes=0):
    return {
        "pair_sha256": "new",
        "question_id": "7",
        "positive_action": {
            "tool": "condition_filter",
            "arguments": {"table": "x", "conditions": {"column": "a", "op": "=", "value": 1}},
        },
        "negative_action": {
            "tool": "condition_filter",
            "arguments": {"table": "x", "conditions": {"column": "a", "op": "=", "value": 2}},
        },
        "positive_verified_by": {
            "source": {"turn_index": turn},
            "branching_online_trials": trials,
            "branching_online_successes": positive_successes,
        },
        "negative_observed_from": {
            "online_trials": trials,
            "online_successes": negative_successes,
            "failure_types": {"wrong_answer": trials},
        },
    }


def test_strict_gate_requires_all_positive_and_no_negative_successes() -> None:
    common = {"original_questions": {"7"}, "original_hashes": set(), "min_online_trials": 4, "min_turn_index": 1}
    assert strict_gate_reason(_row(), **common) is None
    assert strict_gate_reason(_row(positive_successes=3), **common) == "positive_not_all_successful"
    assert strict_gate_reason(_row(negative_successes=1), **common) == "negative_not_all_failed"
    assert strict_gate_reason(_row(turn=0), **common) == "before_minimum_stage"


def test_describe_table_reorder_is_rejected() -> None:
    row = _row()
    row["positive_action"] = {"tool": "describe_table", "arguments": {"tables": ["a", "b"]}}
    row["negative_action"] = {"tool": "describe_table", "arguments": {"tables": ["b", "a"]}}
    assert describe_table_reorder_only(row)


def test_stage_bands_are_stable() -> None:
    assert [stage_band(value) for value in (0, 1, 3, 4, 9)] == [
        "t0",
        "t1_3",
        "t1_3",
        "t4_plus",
        "t4_plus",
    ]
