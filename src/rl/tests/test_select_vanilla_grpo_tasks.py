from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl.scenarios.data.select_vanilla_grpo_tasks import freeze_cohort  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def rows() -> list[dict]:
    values = []
    for database_index in range(5):
        db_id = f"db_{database_index}"
        for row_index in range(40):
            task_index = database_index * 40 + row_index
            values.append(
                {
                    "example_id": f"bird_train_{task_index:05d}",
                    "instance_id": f"bird_train_{task_index:05d}",
                    "db_id": db_id,
                    "question": "q" * (20 + (row_index % 10) * 5),
                    "external_knowledge": "hint" if row_index % 2 else None,
                    "db_path": f"/local/train_databases/{db_id}/{db_id}.sqlite",
                    "gold_sql": f"SELECT {task_index}",
                    "query": f"SELECT {task_index}",
                    "gold_exec_results": [[task_index]],
                    "metadata": {"tool_round_trip": "verified"},
                }
            )
    return values


def run_freeze(tmp_path: Path, source_rows: list[dict], suffix: str) -> dict:
    reference = tmp_path / f"reference_{suffix}.jsonl"
    eligible = tmp_path / f"eligible_{suffix}.jsonl"
    evaluation = tmp_path / f"evaluation_{suffix}.jsonl"
    sft_index = tmp_path / f"sft_{suffix}.jsonl"
    output = tmp_path / f"tasks_{suffix}.jsonl"
    manifest = tmp_path / f"manifest_{suffix}.json"
    write_jsonl(reference, source_rows)
    write_jsonl(eligible, source_rows)
    write_jsonl(evaluation, [source_rows[0]])
    write_jsonl(
        sft_index,
        [
            {"source_episode_id": source_rows[1]["example_id"], "source_step_id": "step_1"},
            {"source_episode_id": source_rows[1]["example_id"], "source_step_id": "step_2"},
        ],
    )
    return freeze_cohort(
        reference_path=reference,
        eligible_path=eligible,
        exclusion_paths=[evaluation, sft_index],
        output_path=output,
        manifest_path=manifest,
        count=100,
        seed="test-seed",
        remote_db_root=Path("/remote/bird/train_databases"),
    )


def test_selection_is_disjoint_and_binds_remote_paths(tmp_path: Path) -> None:
    source_rows = rows()
    manifest = run_freeze(tmp_path, source_rows, "base")
    selected = manifest["output"]["task_ids_in_frozen_order"]
    assert source_rows[0]["example_id"] not in selected
    assert source_rows[1]["example_id"] not in selected
    assert manifest["exclusions"][1]["records"] == 2
    assert manifest["exclusions"][1]["unique_task_ids"] == 1
    assert manifest["all_acceptance_gates_passed"] is True
    output_rows = [
        json.loads(line)
        for line in Path(manifest["output"]["path"]).read_text().splitlines()
    ]
    assert all(
        row["db_path"].startswith("/remote/bird/train_databases/")
        for row in output_rows
    )
    assert all(
        row["metadata"]["rl_training_cohort"]
        == "bird-train-vanilla-grpo-cohort-v1"
        for row in output_rows
    )


def test_gold_mutation_cannot_change_selected_ids(tmp_path: Path) -> None:
    source_rows = rows()
    mutated = copy.deepcopy(source_rows)
    for index, row in enumerate(mutated):
        row["gold_sql"] = f"SECRET {index}"
        row["query"] = f"DIFFERENT {index}"
        row["gold_exec_results"] = [["changed", index]]
    first = run_freeze(tmp_path, source_rows, "first")
    second = run_freeze(tmp_path, mutated, "second")
    assert (
        first["output"]["task_ids_in_frozen_order"]
        == second["output"]["task_ids_in_frozen_order"]
    )
