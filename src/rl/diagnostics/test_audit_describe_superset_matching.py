from __future__ import annotations

from src.rl.diagnostics.audit_describe_superset_matching import audit_rows
from src.rl.diagnostics.test_audit_state_conditioned_prefix_structure import _row


def test_incorrect_describe_superset_matches_but_subset_does_not() -> None:
    rows = [
        _row(
            0,
            correct=True,
            first_tables=["A"],
            second_tool="project",
            second_prompt=[4, 0],
        ),
        _row(
            1,
            correct=True,
            first_tables=["A"],
            second_tool="project",
            second_prompt=[4, 1],
        ),
        _row(
            2,
            correct=False,
            first_tables=["A", "B"],
            second_tool="project",
            second_prompt=[4, 2],
        ),
        _row(
            3,
            correct=False,
            first_tables=["A", "B"],
            second_tool="project",
            second_prompt=[4, 3],
        ),
    ]
    result = audit_rows(rows)
    audit = result["representations"]["strict_policy_prompt"]
    assert audit["exact_full_action_mask"]["masked_events"] == 0
    assert audit["relaxed_describe_superset_mask"]["masked_events"] == 4
    assert audit["delta"]["added_events"] == 4
    assert audit["describe_relation_counts"][
        "unique_action_pairs_incorrect_strict_superset"
    ] == 1

    reversed_rows = []
    for row in rows:
        copied = _row(
            row["sample"]["audit_record"]["sample_index"],
            correct=row["sample"]["correct"],
            first_tables=(
                ["A", "B"] if row["sample"]["correct"] else ["A"]
            ),
            second_tool="project",
            second_prompt=[5, row["sample"]["audit_record"]["sample_index"]],
        )
        reversed_rows.append(copied)
    reversed_result = audit_rows(reversed_rows)
    reversed_audit = reversed_result["representations"]["strict_policy_prompt"]
    assert reversed_audit["relaxed_describe_superset_mask"]["masked_events"] == 0
    assert reversed_audit["describe_relation_counts"][
        "unique_action_pairs_incorrect_strict_subset_rejected"
    ] == 1
