from __future__ import annotations

from select_stratified_bird_train import select


def test_selects_equal_difficulty_counts(tmp_path) -> None:
    database = tmp_path / "db.sqlite"
    database.write_bytes(b"sqlite")
    rows = []
    for level in ("easy", "medium", "hard"):
        for index in range(4):
            rows.append(
                {
                    "example_id": f"{level}-{index}",
                    "db_path": str(database),
                    "gold_sql": "select 1",
                    "metadata": {"difficulty_proxy": level},
                }
            )

    chosen = select(rows, per_level=2, seed=101)

    assert len(chosen) == 6
    assert sorted(row["metadata"]["fixed_pool_difficulty"] for row in chosen) == [
        "challenging",
        "challenging",
        "moderate",
        "moderate",
        "simple",
        "simple",
    ]
