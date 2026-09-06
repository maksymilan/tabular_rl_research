from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.diagnostics.metrics import (
    auc,
    bootstrap_ci,
    distribution,
    exact_sign_p,
    js_divergence_bits,
    paired_delta_summary,
    percentile,
)
from rl.diagnostics.records import (
    group_by_example_index,
    index_rows,
    only_sample,
    selected_rows,
    trajectory_action_signature,
    trajectory_actions,
)


def test_metrics_share_deterministic_report_conventions() -> None:
    assert auc([1.0], [1.0]) == pytest.approx(0.5)
    assert auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(0.875)
    assert percentile([4, 1, 3, 2], 0.75) == pytest.approx(3.25)
    assert distribution([1, 2, 3, 4]) == {
        "mean": 2.5,
        "median": 2.5,
        "p75": 3.25,
        "p90": 3.7,
        "max": 4.0,
    }
    assert exact_sign_p(2, 2) == 1.0
    assert paired_delta_summary([2, -1, 0]) == {
        "improved": 1,
        "regressed": 1,
        "tied": 1,
        "delta_sum": 1.0,
        "mean_delta": pytest.approx(1 / 3),
        "exact_two_sided_sign_p": 1.0,
    }


def test_bootstrap_is_seeded_and_js_divergence_is_symmetric() -> None:
    statistic = lambda left, right: sum(left) / len(left) - sum(right) / len(right)
    first = bootstrap_ci([1, 2, 3], [0, 1, 2], statistic, samples=50, seed=7)
    second = bootstrap_ci([1, 2, 3], [0, 1, 2], statistic, samples=50, seed=7)
    assert first == second
    assert js_divergence_bits({"a": 2, "b": 1}, {"a": 1, "b": 2}) == pytest.approx(
        js_divergence_bits({"a": 1, "b": 2}, {"a": 2, "b": 1})
    )


def test_records_extract_actions_and_group_rows() -> None:
    row = {
        "example_index": 4,
        "samples": [
            {
                "trajectory_id": "t4",
                "turns": [
                    {"parsed": {"tool": "project", "arguments": {"columns": ["a"]}}},
                    {"parsed": {"tool": "answer_from_context", "arguments": {"value": 1}}},
                    {"text": "reasoning only"},
                ],
            }
        ],
    }
    actions = trajectory_actions(row)
    assert actions[0] == ("project", '{"columns":["a"]}')
    assert len(actions) == 2
    assert trajectory_action_signature(row) == trajectory_action_signature(only_sample(row))
    rows = [row, {"example_index": 4}, {"example_index": 8}]
    assert [len(group) for group in group_by_example_index(rows).values()] == [2, 1]


def test_records_reject_duplicate_index_and_enforce_exact_selection(tmp_path: Path) -> None:
    rows = [{"example_index": 1, "samples": [{}]}, {"example_index": 2, "samples": [{}]}]
    with pytest.raises(ValueError, match="duplicate record key"):
        index_rows(rows + [rows[0]], lambda row: int(row["example_index"]))
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert set(selected_rows(path, [2])) == {2}
    with pytest.raises(ValueError, match="exact selected cohort"):
        selected_rows(path, [3])
    with pytest.raises(ValueError, match="exactly one sample"):
        only_sample({"example_index": 3, "samples": []})
