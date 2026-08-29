from __future__ import annotations

import json

import pytest

from src.rl.diagnostics.analyze_terminal_state_quality_failures import (
    analyze,
    output_content_overlap,
)


def _unit(position: int, expression: list) -> str:
    return json.dumps(["position", position, expression], separators=(",", ":"))


def _row(task_id: str, *, correct: bool, score: float, output: dict) -> dict:
    return {
        "task_id": task_id,
        "db_id": task_id,
        "position": int(task_id.rsplit("_", 1)[-1]),
        "correct": correct,
        "score": score,
        "c_terminal": score,
        "c_max": score,
        "tools": ["project"],
        "terminal_overlap": {"per_category": {
            "source": {"score": 1.0, "gold": 1, "candidate": 1, "matched": 1,
                       "missing": [], "extra": []},
            "output": output,
        }},
    }


def test_output_content_recovers_leading_extra_column_without_calling_it_exact() -> None:
    name = ["column", "country", "name"]
    density = ["div", ["column", "country", "population"], ["column", "country", "area"]]
    industry = ["column", "economy", "industry"]
    row = _row("task_1", correct=False, score=0.8, output={
        "score": 0.0,
        "gold": 2,
        "candidate": 3,
        "matched": 0,
        "missing": [_unit(0, density), _unit(1, industry)],
        "extra": [_unit(0, name), _unit(1, density), _unit(2, industry)],
    })
    assert output_content_overlap(row) == pytest.approx(2 / 3)


def test_output_content_recovers_reordered_exact_columns() -> None:
    left = ["column", "team", "losses"]
    right = ["column", "team", "coach_id"]
    row = _row("task_2", correct=True, score=0.2, output={
        "score": 0.0,
        "gold": 2,
        "candidate": 2,
        "matched": 0,
        "missing": [_unit(0, left), _unit(1, right)],
        "extra": [_unit(0, right), _unit(1, left)],
    })
    assert output_content_overlap(row) == pytest.approx(1.0)


def test_analysis_selects_each_populated_cell_and_marks_position_artifacts() -> None:
    exact = {
        "score": 1.0, "gold": 1, "candidate": 1, "matched": 1, "missing": [], "extra": []
    }
    wrong_position = {
        "score": 0.0, "gold": 2, "candidate": 2, "matched": 0,
        "missing": [_unit(0, ["column", "t", "a"]), _unit(1, ["column", "t", "b"])],
        "extra": [_unit(0, ["column", "t", "b"]), _unit(1, ["column", "t", "a"])],
    }
    rows = [
        _row("task_101", correct=True, score=0.9, output=exact),
        _row("task_102", correct=True, score=0.6, output=exact),
        _row("task_103", correct=True, score=0.2, output=wrong_position),
        _row("task_104", correct=False, score=0.9, output=exact),
        _row("task_105", correct=False, score=0.6, output=exact),
        _row("task_106", correct=False, score=0.2, output=wrong_position),
    ]
    summary, selected = analyze(rows, per_cell=1)
    assert summary["selected_case_count"] == 6
    assert summary["output_position_analysis"]["position_artifact_count"] == 2
    assert {item["cell"] for item in selected} == {
        "correct_high", "correct_mid", "correct_low",
        "incorrect_high", "incorrect_mid", "incorrect_low",
    }
