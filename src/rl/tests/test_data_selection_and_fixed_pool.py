from __future__ import annotations

from pathlib import Path

import pytest

from rl.data_selection.pilot import legal_rate, pilot_bucket, select_pilot
from rl.fixed_pool.selection import level_of, select


def _record(index: int, *, correct: int = 1, legal: bool = True) -> dict:
    return {
        "example_index": index,
        "samples": [{"correct": i < correct, "legal": legal} for i in range(4)],
    }


def test_pilot_selection_is_deterministic_and_prioritizes_mixed() -> None:
    rows = [_record(2, correct=0), _record(1, correct=2), _record(3, correct=4)]
    assert pilot_bucket(rows[1]) == "mixed_success"
    assert legal_rate(rows[1]) == 1.0
    assert [r["example_index"] for r in select_pilot(rows)] == [1]


def test_fixed_pool_selection_checks_files_and_balances_levels(tmp_path: Path) -> None:
    rows = []
    for level in ("easy", "medium", "hard"):
        for index in range(2):
            db = tmp_path / f"{level}-{index}.db"
            db.touch()
            rows.append({
                "example_id": f"{level}-{index}",
                "db_path": str(db),
                "query": "select 1",
                "metadata": {"difficulty_proxy": level},
            })
    chosen = select(rows, per_level=1, seed=7)
    assert len(chosen) == 3
    assert {r["metadata"]["fixed_pool_difficulty"] for r in chosen} == {
        "simple", "moderate", "challenging"
    }
    assert level_of(chosen[0]) in {"simple", "moderate", "challenging"}


def test_fixed_pool_selection_rejects_missing_level() -> None:
    with pytest.raises(ValueError, match="unsupported difficulty"):
        level_of({"example_id": "x", "metadata": {"difficulty_proxy": "unknown"}})
