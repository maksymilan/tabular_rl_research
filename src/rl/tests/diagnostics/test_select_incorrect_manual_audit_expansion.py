from __future__ import annotations

from rl.scenarios.diagnostics.select_incorrect_manual_audit_expansion import BINS, select


def test_selects_equal_incorrect_strata_and_excludes_prior_ids() -> None:
    rows = []
    for bin_index, (_, low, high) in enumerate(BINS):
        for item_index in range(4):
            rows.append({
                "task_id": f"task_{bin_index}_{item_index}",
                "db_id": "db",
                "correct": False,
                "score": low + (high - low) * (item_index + 1) / 5,
            })
    rows.append({"task_id": "correct", "correct": True, "score": 0.5})
    selected, summary = select(
        rows,
        excluded_ids={"task_0_0"},
        per_bin=2,
        seed="test",
    )
    assert len(selected) == 12
    assert {item["cell"] for item in selected} == {item[0] for item in BINS}
    assert "task_0_0" not in {item["task_id"] for item in selected}
    assert summary["availability_after_exclusion"]["wrong_00_0125"] == 3
