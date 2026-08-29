from __future__ import annotations

from src.rl.diagnostics.analyze_incorrect_manual_audit_expansion import _pairwise, _spearman


def test_pairwise_ordering_rewards_higher_score_for_lower_severity() -> None:
    rows = [
        {"current_score": 0.9, "severity": 0},
        {"current_score": 0.5, "severity": 1},
        {"current_score": 0.1, "severity": 3},
    ]
    result = _pairwise(rows, "current_score")
    assert result["comparable_pairs"] == 3
    assert result["accuracy"] == 1.0


def test_spearman_is_one_for_matching_order_with_ties() -> None:
    assert _spearman([0.1, 0.2, 0.2, 0.9], [0.0, 0.5, 0.5, 1.0]) == 1.0
