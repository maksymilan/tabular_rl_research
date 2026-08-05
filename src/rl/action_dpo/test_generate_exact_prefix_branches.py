import json

from generate_exact_prefix_branches import (
    anchor_turn_indices,
    build_plan,
    forbidden_visible_field,
    source_rows,
    stable_seed,
)


def _source_row(example_index: int, turns: int) -> dict:
    return {
        "environment": {"example_index": example_index},
        "sample": {
            "audit_record": {
                "sample_index": 1,
                "turns": [{} for _ in range(turns)],
            }
        },
    }


def test_anchor_turn_indices_cover_early_middle_late() -> None:
    assert anchor_turn_indices(10, 3) == [0, 4, 9]
    assert anchor_turn_indices(2, 3) == [0, 1]
    assert anchor_turn_indices(5, 1) == [2]


def test_plan_is_stable_and_keeps_distinct_stages() -> None:
    plan = build_plan(
        [_source_row(7, 10), _source_row(9, 4)],
        pool_sha256="abc",
        anchors_per_trajectory=3,
    )
    assert [item["anchor_id"] for item in plan["anchors"]] == [
        "e00007_s01_t00",
        "e00007_s01_t04",
        "e00007_s01_t09",
        "e00009_s01_t00",
        "e00009_s01_t02",
        "e00009_s01_t03",
    ]
    assert plan["state_normalization"] == "none"


def test_plan_can_exclude_first_turn_anchors() -> None:
    plan = build_plan(
        [_source_row(7, 6)],
        pool_sha256="abc",
        anchors_per_trajectory=6,
        min_turn_index=1,
    )
    assert [item["turn_index"] for item in plan["anchors"]] == [1, 2, 3, 4, 5]
    assert plan["minimum_anchor_turn_index"] == 1


def test_forbidden_visible_field_finds_nested_gold_only() -> None:
    assert forbidden_visible_field([{"role": "user", "content": {"gold_sql": "x"}}]) == "gold_sql"
    assert forbidden_visible_field([{"role": "user", "content": "gold is hidden"}]) is None


def test_stable_seed_depends_on_branch_identity() -> None:
    assert stable_seed(15, "a", 1) == stable_seed(15, "a", 1)
    assert stable_seed(15, "a", 1) != stable_seed(15, "a", 2)


def test_source_rows_can_restrict_clean_trajectories(tmp_path) -> None:
    rows = []
    for example_index in (7, 9):
        rows.append(
            {
                "environment": {
                    "dataset_split": "train",
                    "example_index": example_index,
                },
                "sample": {
                    "correct": True,
                    "audit_record": {
                        "correct": True,
                        "legal": True,
                        "errors": 0,
                        "protocol_version": "version26",
                        "sample_index": 0,
                        "turns": [{"model_output": "x"}],
                    },
                },
            }
        )
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = source_rows(
        pool,
        limit_trajectories=0,
        include_example_indices={9},
    )
    assert [row["environment"]["example_index"] for row in selected] == [9]
