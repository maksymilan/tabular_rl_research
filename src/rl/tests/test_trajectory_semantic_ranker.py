from __future__ import annotations

import pytest

from rl.diagnostics.semantic.trajectory_semantic_ranker import (
    GoldSemanticCompiler,
    TrajectorySemanticCompiler,
    semantic_overlap,
    answer_similarity,
    trajectory_quality,
)


SCHEMA = {
    "department": ["id", "name"],
    "employee": ["dept_id", "status", "salary"],
}
GOLD_SQL = """
SELECT d.name, AVG(e.salary)
FROM department AS d
JOIN employee AS e ON d.id = e.dept_id
WHERE e.status = 'active'
GROUP BY d.name
ORDER BY AVG(e.salary) DESC
LIMIT 3
"""


def trajectory_steps(*, aggregate: str = "mean") -> list[dict]:
    return [
        {
            "step_id": "step_1",
            "tool_call": {"tool": "join_tables", "arguments": {}},
            "tool_output": {
                "table": "join_1",
                "columns": [
                    "department.id",
                    "department.name",
                    "employee.dept_id",
                    "employee.status",
                    "employee.salary",
                ],
                "derivation": {
                    "inputs": [
                        {"kind": "table", "role": "base", "ref": "department", "namespace": "department"},
                        {"kind": "table", "role": "joined", "ref": "employee", "namespace": "employee"},
                    ],
                    "semantics": {
                        "edges": [{
                            "join_type": "inner",
                            "namespace": "employee",
                            "input": "employee",
                            "on": [{"left": "department.id", "right": "dept_id"}],
                        }]
                    },
                },
            },
        },
        {
            "step_id": "step_2",
            "tool_call": {"tool": "condition_filter", "arguments": {}},
            "tool_output": {
                "table": "filter_2",
                "columns": [
                    "department.id",
                    "department.name",
                    "employee.dept_id",
                    "employee.status",
                    "employee.salary",
                ],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "join_1"}],
                    "semantics": {
                        "predicate": {"column": "employee.status", "op": "=", "value": "active"}
                    },
                },
            },
        },
        {
            "step_id": "step_3",
            "tool_call": {"tool": "group_aggregate", "arguments": {}},
            "tool_output": {
                "table": "agg_3",
                "columns": ["department.name", "metric"],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "filter_2"}],
                    "semantics": {
                        "row_grain": ["department.name"],
                        "aggregations": [{"output": "metric", "op": aggregate, "source": "employee.salary"}],
                    },
                },
            },
        },
        {
            "step_id": "step_4",
            "tool_call": {"tool": "extreme_value_select", "arguments": {}},
            "tool_output": {
                "table": "top_4",
                "columns": ["department.name", "metric"],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "agg_3"}],
                    "semantics": {"order_by": ["metric DESC"], "top_k": 3},
                },
            },
        },
        {
            "step_id": "step_5",
            "tool_call": {"tool": "answer_from_context", "arguments": {"evidence": {"table": "top_4"}}},
            "tool_output": {},
        },
    ]


def test_join_then_filter_lineage_matches_gold_semantics() -> None:
    gold = GoldSemanticCompiler(SCHEMA).compile(GOLD_SQL)
    candidate, handle, selection = TrajectorySemanticCompiler(SCHEMA).compile(trajectory_steps())

    assert gold.eligible
    assert candidate.eligible
    assert handle == "top_4"
    assert selection == "terminal_evidence"
    overlap = semantic_overlap(gold, candidate)
    assert overlap.score == pytest.approx(1.0)
    assert overlap.per_class["join"]["matched"] == 1
    assert overlap.per_class["predicate"]["matched"] == 1


def test_wrong_aggregate_creates_one_missing_and_one_extra_unit() -> None:
    gold = GoldSemanticCompiler(SCHEMA).compile(GOLD_SQL)
    candidate, _, _ = TrajectorySemanticCompiler(SCHEMA).compile(
        trajectory_steps(aggregate="sum")
    )
    overlap = semantic_overlap(gold, candidate)

    assert overlap.score is not None and overlap.score < 1.0
    aggregate = overlap.per_class["aggregate_compute"]
    assert aggregate["matched"] == 0
    assert len(aggregate["missing"]) == 1
    assert len(aggregate["extra"]) == 1


def test_correct_reward_is_constant_even_when_raw_semantic_quality_is_low() -> None:
    gold = GoldSemanticCompiler(SCHEMA).compile(GOLD_SQL)
    candidate, handle, selection = TrajectorySemanticCompiler(SCHEMA).compile(
        trajectory_steps(aggregate="sum")
    )
    quality = trajectory_quality(
        correct=True,
        gold=gold,
        candidate=candidate,
        predicted_rows=[["sales", 10]],
        gold_rows=[["sales", 10]],
        evidence_handle=handle,
        artifact_selection=selection,
    )

    assert quality.raw_quality < 1.0
    assert quality.training_reward == 1.0


def test_failure_reward_remains_below_every_correct_reward() -> None:
    gold = GoldSemanticCompiler(SCHEMA).compile(GOLD_SQL)
    candidate, handle, selection = TrajectorySemanticCompiler(SCHEMA).compile(trajectory_steps())
    quality = trajectory_quality(
        correct=False,
        gold=gold,
        candidate=candidate,
        predicted_rows=[["sales", 10]],
        gold_rows=[["sales", 10]],
        evidence_handle=handle,
        artifact_selection=selection,
    )

    assert quality.raw_quality == pytest.approx(1.0)
    assert quality.training_reward == pytest.approx(-0.6)


def test_answer_similarity_matches_bird_numeric_cell_equality() -> None:
    score, diagnostics = answer_similarity([[1, "x"]], [[1.0, "x"]])

    assert score == pytest.approx(1.0)
    assert diagnostics["row_jaccard"] == pytest.approx(1.0)
