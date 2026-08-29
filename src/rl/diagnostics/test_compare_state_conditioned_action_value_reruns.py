from __future__ import annotations

from src.rl.diagnostics.compare_state_conditioned_action_value_reruns import (
    compare_rows,
)
from src.rl.diagnostics.test_audit_state_conditioned_prefix_structure import _row


def test_pair_preference_stability_counts_same_and_opposite_signs() -> None:
    rows_a = [
        _row(
            index,
            correct=(index < 3 or index == 3),
            first_tables=(["A"] if index < 3 else ["B"]),
            second_tool="project",
            second_prompt=[4, index],
        )
        for index in range(6)
    ]
    rows_b = [
        _row(
            index,
            correct=(index < 2 or index >= 3),
            first_tables=(["A"] if index < 3 else ["B"]),
            second_tool="project",
            second_prompt=[5, index],
        )
        for index in range(6)
    ]
    result = compare_rows(rows_a, rows_b)
    assert result["initial"]["action_pair_comparisons"] == 1
    assert result["initial"]["stability_counts"] == {
        "opposite_nonzero_sign": 1
    }
