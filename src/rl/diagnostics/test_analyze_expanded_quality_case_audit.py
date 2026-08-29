from __future__ import annotations

from src.rl.diagnostics.analyze_expanded_quality_case_audit import rank_expansion_diagnostic


def _turn(tool, table, inputs, row_count, arguments=None):
    return {
        "tool": tool,
        "arguments": arguments or {},
        "tool_output": {
            "table": table,
            "row_count": row_count,
            "derivation": {"inputs": [
                {"kind": "table", "ref": item} for item in inputs
            ]},
        },
    }


def test_rank_expansion_detects_one_to_many_after_top_k() -> None:
    case = {"environment": {"gold_sql": "SELECT tag FROM album JOIN tags USING(id) LIMIT 1"}, "turns": [
        _turn("extreme_value_select", "top_1", ["base"], 1, {"top_k": 1}),
        _turn("join_tables", "join_2", ["top_1", "tags"], 4, {"base": "top_1"}),
        _turn("project", "project_3", ["join_2"], 4, {"table": "join_2"}),
        {"tool": "answer_from_context", "arguments": {"evidence": {"table": "project_3"}},
         "tool_output": None},
    ]}
    result = rank_expansion_diagnostic(case)
    assert result is not None
    assert result["top_k"] == 1
    assert result["terminal_rows"] == 4
    assert result["ratio"] == 0.25


def test_rank_expansion_ignores_terminal_within_top_k() -> None:
    case = {"environment": {"gold_sql": "SELECT name FROM base LIMIT 1"}, "turns": [
        _turn("extreme_value_select", "top_1", ["base"], 1, {"top_k": 1}),
        _turn("project", "project_2", ["top_1"], 1, {"table": "top_1"}),
        {"tool": "answer_from_context", "arguments": {"evidence": {"table": "project_2"}},
         "tool_output": None},
    ]}
    assert rank_expansion_diagnostic(case) is None


def test_rank_expansion_does_not_use_unrelated_dependency_graph() -> None:
    case = {"environment": {"gold_sql": "SELECT tag FROM album JOIN tags USING(id) LIMIT 1"}, "turns": [
        _turn("extreme_value_select", "top_1", ["base"], 1, {"top_k": 1}),
        _turn("join_tables", "join_2", ["top_1", "tags"], 4, {"base": "other"}),
        _turn("project", "project_3", ["join_2"], 4, {"table": "join_2"}),
        {"tool": "answer_from_context", "arguments": {"evidence": {"table": "project_3"}},
         "tool_output": None},
    ]}
    assert rank_expansion_diagnostic(case) is None
