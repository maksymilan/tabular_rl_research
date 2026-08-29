from __future__ import annotations

import copy

import pytest

from src.rl.tool_state_obligation_quality import score_tool_state_trajectory


SCHEMA = {
    "department": ["id", "name"],
    "employee": ["dept_id", "status", "salary", "tax"],
}


def _main_steps() -> list[dict]:
    return [
        {
            "step_id": "step_1",
            "tool_call": {"tool": "join_tables", "arguments": {}},
            "tool_output": {
                "table": "join_1",
                "columns": [
                    "department.id", "department.name", "employee.dept_id",
                    "employee.status", "employee.salary", "employee.tax",
                ],
                "derivation": {
                    "inputs": [
                        {"kind": "table", "role": "base", "ref": "department"},
                        {"kind": "table", "role": "joined", "ref": "employee"},
                    ],
                    "semantics": {"edges": [{
                        "join_type": "inner", "namespace": "employee", "input": "employee",
                        "on": [{"left": "department.id", "right": "dept_id"}],
                    }]},
                },
            },
        },
        {
            "step_id": "step_2",
            "tool_call": {"tool": "condition_filter", "arguments": {}},
            "tool_output": {
                "table": "filter_2",
                "columns": [
                    "department.id", "department.name", "employee.dept_id",
                    "employee.status", "employee.salary", "employee.tax",
                ],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "join_1"}],
                    "semantics": {"predicate": {
                        "column": "employee.status", "op": "=", "value": "active"
                    }},
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
                        "aggregations": [{"output": "metric", "op": "mean", "source": "employee.salary"}],
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


GOLD = """
SELECT d.name, AVG(e.salary)
FROM department d JOIN employee e ON d.id = e.dept_id
WHERE e.status = 'active'
GROUP BY d.name ORDER BY AVG(e.salary) DESC LIMIT 3
"""


def _score(sql: str, steps: list[dict]):
    return score_tool_state_trajectory(gold_sql=sql, table_columns=SCHEMA, steps=steps)


def test_exact_tool_state_obligations_score_one() -> None:
    result = _score(GOLD, _main_steps())
    assert result.semantic_eligible
    assert result.c_terminal == pytest.approx(1.0)
    assert result.c_max == pytest.approx(1.0)
    assert result.scoring_mode == "terminal"
    assert result.score == pytest.approx(1.0)


def test_filter_before_or_after_join_has_same_quality() -> None:
    after = _main_steps()
    before = copy.deepcopy(after)
    filter_step, join_step = before[1], before[0]
    filter_step["step_id"] = "step_1"
    filter_step["tool_output"]["table"] = "filter_1"
    filter_step["tool_output"]["columns"] = ["dept_id", "status", "salary", "tax"]
    filter_step["tool_output"]["derivation"]["inputs"][0]["ref"] = "employee"
    filter_step["tool_output"]["derivation"]["semantics"]["predicate"]["column"] = "status"
    join_step["step_id"] = "step_2"
    join_step["tool_output"]["table"] = "join_2"
    join_step["tool_output"]["derivation"]["inputs"][1]["ref"] = "filter_1"
    before[:2] = [filter_step, join_step]
    before[2]["tool_output"]["derivation"]["inputs"][0]["ref"] = "join_2"

    left = _score(GOLD, after)
    right = _score(GOLD, before)
    assert left.score == pytest.approx(1.0)
    assert right.score == pytest.approx(left.score)


def _filter_project_steps(split: bool) -> list[dict]:
    predicates = [
        {"column": "status", "op": "=", "value": "active"},
        {"column": "salary", "op": ">", "value": 100},
    ]
    steps: list[dict] = []
    prior = "employee"
    if split:
        for index, predicate in enumerate(predicates, 1):
            handle = f"filter_{index}"
            steps.append({
                "step_id": f"step_{index}",
                "tool_call": {"tool": "condition_filter", "arguments": {}},
                "tool_output": {
                    "table": handle, "columns": ["dept_id", "status", "salary", "tax"],
                    "derivation": {
                        "inputs": [{"kind": "table", "role": "input", "ref": prior}],
                        "semantics": {"predicate": predicate},
                    },
                },
            })
            prior = handle
    else:
        steps.append({
            "step_id": "step_1",
            "tool_call": {"tool": "condition_filter", "arguments": {}},
            "tool_output": {
                "table": "filter_1", "columns": ["dept_id", "status", "salary", "tax"],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "employee"}],
                    "semantics": {"predicate": {"and": predicates}},
                },
            },
        })
        prior = "filter_1"
    project_step = len(steps) + 1
    steps.extend([
        {
            "step_id": f"step_{project_step}",
            "tool_call": {"tool": "project", "arguments": {}},
            "tool_output": {
                "table": "project_final", "columns": ["status", "salary"],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": prior}],
                    "semantics": {"row_operation": "preserve", "column_lineage": [
                        {"output": "status", "sources": ["status"], "kind": "column", "expression": "status"},
                        {"output": "salary", "sources": ["salary"], "kind": "column", "expression": "salary"},
                    ]},
                },
            },
        },
        {
            "step_id": f"step_{project_step + 1}",
            "tool_call": {"tool": "answer_from_context", "arguments": {"evidence": {"table": "project_final"}}},
            "tool_output": {},
        },
    ])
    return steps


def test_combined_and_split_filters_are_equivalent() -> None:
    sql = "SELECT status, salary FROM employee WHERE status = 'active' AND salary > 100"
    combined = _score(sql, _filter_project_steps(False))
    split = _score(sql, _filter_project_steps(True))
    assert combined.score == pytest.approx(1.0)
    assert split.score == pytest.approx(combined.score)


def test_aggregate_then_scalar_compute_is_inlined() -> None:
    steps = [
        {
            "step_id": "step_1",
            "tool_call": {"tool": "group_aggregate", "arguments": {}},
            "tool_output": {
                "table": "agg_1", "columns": ["sum_salary", "sum_tax"],
                "derivation": {
                    "inputs": [{"kind": "table", "role": "input", "ref": "employee"}],
                    "semantics": {"row_grain": [], "aggregations": [
                        {"output": "sum_salary", "op": "sum", "source": "salary"},
                        {"output": "sum_tax", "op": "sum", "source": "tax"},
                    ]},
                },
            },
        },
        {
            "step_id": "step_2",
            "tool_call": {"tool": "scalar_compute", "arguments": {
                "operation": "subtract", "result_name": "net", "operands": [
                    {"value_ref": "step_1", "column": "sum_salary"},
                    {"value_ref": "step_1", "column": "sum_tax"},
                ],
            }},
            "tool_output": {
                "table": "scalar_2", "columns": ["net"],
                "derivation": {
                    "inputs": [
                        {"kind": "value", "role": "operand", "ref": "step_1", "column": "sum_salary"},
                        {"kind": "value", "role": "operand", "ref": "step_1", "column": "sum_tax"},
                    ],
                    "semantics": {"operation": "subtract", "result_column": "net"},
                },
            },
        },
        {
            "step_id": "step_3",
            "tool_call": {"tool": "answer_from_context", "arguments": {"evidence": {"table": "scalar_2"}}},
            "tool_output": {},
        },
    ]
    result = _score("SELECT SUM(salary) - SUM(tax) FROM employee", steps)
    assert result.semantic_eligible
    assert result.score == pytest.approx(1.0)


def test_unused_successful_branch_does_not_change_terminal_quality() -> None:
    clean = _score(GOLD, _main_steps())
    noisy_steps = _main_steps()
    noisy_steps.insert(-1, {
        "step_id": "unused",
        "tool_call": {"tool": "project", "arguments": {}},
        "tool_output": {
            "table": "unused_table", "columns": ["status"],
            "derivation": {
                "inputs": [{"kind": "table", "role": "input", "ref": "employee"}],
                "semantics": {"row_operation": "preserve", "column_lineage": [{
                    "output": "status", "sources": ["status"], "kind": "column", "expression": "status"
                }]},
            },
        },
    })
    noisy = _score(GOLD, noisy_steps)
    assert noisy.c_terminal == clean.c_terminal
    assert noisy.c_max == clean.c_max
    assert noisy.score == pytest.approx(clean.score)


def test_no_terminal_uses_half_max_fallback_without_dependency_credit() -> None:
    incomplete = _main_steps()[:-1]
    result = _score(GOLD, incomplete)
    assert result.terminal_evidence is None
    assert result.c_max == pytest.approx(1.0)
    assert result.scoring_mode == "half_max_fallback"
    assert result.score == pytest.approx(0.5)


def test_missing_predicate_lowers_only_the_relevant_semantic_overlap() -> None:
    steps = _main_steps()
    del steps[1]
    steps[1]["tool_output"]["derivation"]["inputs"][0]["ref"] = "join_1"
    result = _score(GOLD, steps)
    assert result.score is not None and result.score < 1.0
    assert result.terminal_overlap["per_category"]["predicate"]["matched"] == 0
    for category in ("source", "join", "grain", "value", "rank", "output"):
        assert result.terminal_overlap["per_category"][category]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutate", "category"),
    [
        (lambda steps: steps[0]["tool_output"]["derivation"]["semantics"]["edges"][0].update(join_type="left"), "join"),
        (lambda steps: steps[2]["tool_output"]["derivation"]["semantics"]["aggregations"][0].update(op="sum"), "value"),
        (lambda steps: steps[3]["tool_output"]["derivation"]["semantics"].update(top_k=5), "rank"),
    ],
)
def test_single_semantic_fault_monotonically_lowers_quality(mutate, category: str) -> None:
    clean = _score(GOLD, _main_steps())
    corrupted_steps = _main_steps()
    mutate(corrupted_steps)
    corrupted = _score(GOLD, corrupted_steps)
    assert corrupted.score is not None and clean.score is not None
    assert corrupted.score < clean.score
    assert corrupted.terminal_overlap["per_category"][category]["score"] < 1.0


def test_output_alias_does_not_change_quality() -> None:
    renamed = _main_steps()
    aggregate = renamed[2]["tool_output"]
    aggregate["columns"][1] = "average_salary"
    aggregate["derivation"]["semantics"]["aggregations"][0]["output"] = "average_salary"
    ranked = renamed[3]["tool_output"]
    ranked["columns"][1] = "average_salary"
    ranked["derivation"]["semantics"]["order_by"] = ["average_salary DESC"]
    assert _score(GOLD, renamed).score == pytest.approx(_score(GOLD, _main_steps()).score)
