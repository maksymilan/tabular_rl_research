from __future__ import annotations

from rl.scenarios.diagnostics.analyze_all_incorrect_trajectory_patterns import (
    classify_incorrect,
    gold_sql_shape,
    question_shape,
    semantic_features,
    trajectory_features,
)


def _score(*, source: float = 1.0, join: float = 1.0, **tail: float) -> dict:
    categories = {
        "source": {"score": source},
        "join": {"score": join},
        **{name: {"score": value} for name, value in tail.items()},
    }
    return {
        "semantic_eligible": True,
        "scoring_mode": "terminal",
        "score": 0.8,
        "answer_score_validation_only": 0.2,
        "terminal_overlap": {"per_category": categories},
    }


def test_classifies_one_and_two_tail_near_misses() -> None:
    one = semantic_features(_score(predicate=0.0))
    two = semantic_features(_score(value=0.0, output=0.0))
    assert classify_incorrect(failure_type="wrong_answer", legal=True, semantic=one) == "one_tail_near_miss"
    assert classify_incorrect(failure_type="wrong_answer", legal=True, semantic=two) == "two_tail_near_miss"


def test_wrong_route_is_not_near_miss() -> None:
    semantic = semantic_features(_score(source=0.5, predicate=0.0))
    assert classify_incorrect(failure_type="wrong_answer", legal=True, semantic=semantic) == "wrong_source_or_join_route"


def test_shapes_are_deterministic() -> None:
    sql = "SELECT COUNT(*) FROM t WHERE x > 2 LIMIT 3"
    shape = gold_sql_shape(sql)
    assert shape["aggregate"] is True
    assert shape["limit_without_order"] is True
    assert shape["join"] is False
    question = question_shape("How many rows are at least 2 years old?")
    assert question["count"] is True
    assert question["boundary"] is True
    assert question["date_time"] is True


def test_extracts_actions_from_raw_parsed_turns() -> None:
    audit = {
        "steps": 2,
        "turns": [
            {"parsed": {"tool": "describe_table", "arguments": {"tables": ["t"]}}},
            {"parsed": {"tool": "describe_table", "arguments": {"tables": ["t"]}}},
        ],
    }
    features = trajectory_features(audit)
    assert features["tools"] == ["describe_table", "describe_table"]
    assert features["adjacent_exact_action_repeats"] == 1
