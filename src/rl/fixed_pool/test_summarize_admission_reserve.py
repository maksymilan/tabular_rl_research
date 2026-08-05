from __future__ import annotations

from summarize_admission_reserve import summarize


def test_selects_first_causally_admitted_task_per_level() -> None:
    tasks = []
    trajectories = []
    counterfactual = []
    for level in ("simple", "moderate", "challenging"):
        for rank in range(3):
            task_id = f"{level}-{rank}"
            tasks.append(
                {
                    "example_id": task_id,
                    "metadata": {
                        "fixed_pool_difficulty": level,
                        "fixed_pool_reserve_rank": rank,
                    },
                }
            )
            correct = rank != 0
            trajectories.append(
                {
                    "environment": {"task_id": task_id},
                    "sample": {"correct": correct},
                }
            )
            if correct:
                counterfactual.append({"task_id": task_id, "passed": rank == 1})

    result = summarize(tasks, trajectories, counterfactual, required_per_level=1)

    assert result["status"] == "passed"
    assert result["selected_replacements"] == {
        "simple": ["simple-1"],
        "moderate": ["moderate-1"],
        "challenging": ["challenging-1"],
    }
    assert result["admitted_tasks"] == 6
    assert result["positive_bearing_admitted_tasks"] == 3
