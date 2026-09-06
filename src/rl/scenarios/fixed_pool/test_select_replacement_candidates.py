from __future__ import annotations

from rl.scenarios.fixed_pool.select_replacement_candidates import select_reserve_window


def test_selects_next_balanced_window(tmp_path) -> None:
    database = tmp_path / "db.sqlite"
    database.write_bytes(b"sqlite")
    rows = []
    for level in ("easy", "medium", "hard"):
        for index in range(8):
            rows.append(
                {
                    "example_id": f"{level}-{index}",
                    "db_path": str(database),
                    "gold_sql": "select 1",
                    "metadata": {"difficulty_proxy": level},
                }
            )

    first = select_reserve_window(
        rows,
        selected_per_level=2,
        reserve_per_level=3,
        seed=101,
    )
    second = select_reserve_window(
        rows,
        selected_per_level=2,
        reserve_per_level=3,
        seed=101,
    )

    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert len(first) == 9
    assert sorted(row["metadata"]["fixed_pool_difficulty"] for row in first) == [
        "challenging",
        "challenging",
        "challenging",
        "moderate",
        "moderate",
        "moderate",
        "simple",
        "simple",
        "simple",
    ]
    assert all(2 <= row["metadata"]["fixed_pool_reserve_rank"] < 5 for row in first)
