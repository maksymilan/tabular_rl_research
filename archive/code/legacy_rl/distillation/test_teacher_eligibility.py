from __future__ import annotations

import pytest

from src.rl.distillation.teacher_eligibility import (
    build_eligibility_rows,
    interleaved_training_order,
)


def rollout_rows(correct_by_example, *, split="train", k=4):
    rows = []
    for example_index, correct_count in correct_by_example.items():
        for sample_index in range(k):
            correct = sample_index < correct_count
            rows.append(
                {
                    "environment": {
                        "dataset_split": split,
                        "example_index": example_index,
                        "db_id": f"db_{example_index}",
                        "question": f"question {example_index}",
                        # Hidden verifier metadata may exist in source artifacts;
                        # it must not be copied into eligibility rows.
                        "gold_sql": "SELECT forbidden",
                    },
                    "sample": {
                        "correct": correct,
                        "audit_record": {
                            "sample_index": sample_index,
                            "protocol_version": "version26",
                            "correct": correct,
                            "legal": True,
                        },
                    },
                }
            )
    return rows


def test_k4_routing_and_no_gold_leak():
    rows = build_eligibility_rows(
        rollout_rows({0: 3, 1: 0, 2: 2, 3: 1, 4: 0}),
        rollout_rows({0: 0, 1: 3, 2: 4, 3: 0, 4: 0}),
    )
    assert [row["category"] for row in rows] == [
        "sft2_only_dense",
        "exp15_only_dense",
        "both_dense",
        "branch_only",
        "both_wrong",
    ]
    assert all("gold_sql" not in row for row in rows)
    assert rows[3]["teachers"]["sft2"]["branch_eligible"] is True
    assert rows[3]["teachers"]["sft2"]["dense_eligible"] is False


def test_non_train_input_fails_closed():
    with pytest.raises(ValueError, match="train-only"):
        build_eligibility_rows(
            rollout_rows({0: 2}, split="dev"),
            rollout_rows({0: 2}, split="dev"),
        )


def test_incomplete_k4_fails_instead_of_silently_routing():
    with pytest.raises(ValueError, match="sample indices"):
        build_eligibility_rows(
            rollout_rows({0: 2}, k=3),
            rollout_rows({0: 2}, k=3),
            expected_k=4,
        )


def test_training_order_is_frozen_balanced_and_deterministic():
    rows = build_eligibility_rows(
        rollout_rows({0: 3, 1: 3, 2: 0, 3: 0, 4: 2}),
        rollout_rows({0: 0, 1: 0, 2: 3, 3: 3, 4: 2}),
    )
    first = interleaved_training_order(rows, seed=101)
    second = interleaved_training_order(rows, seed=101)
    assert first == second
    assert set(first) == {0, 1, 2, 3, 4}
    categories = {row["example_index"]: row["category"] for row in rows}
    assert len({categories[index] for index in first[:3]}) == 3
