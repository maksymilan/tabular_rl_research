from __future__ import annotations

import json

from assemble_repaired_fixed_pool import assemble


def write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_replaces_failed_groups_in_place_and_preserves_sequences(tmp_path) -> None:
    original = tmp_path / "original"
    reserve = tmp_path / "reserve"
    output = tmp_path / "repaired"
    (original / "groups").mkdir(parents=True)
    (reserve / "groups").mkdir(parents=True)
    tasks = []
    failed_results = []
    failed_by_level = {}
    for level_index, level in enumerate(("simple", "moderate", "challenging")):
        failed_by_level[level] = []
        for local_index in range(20):
            task_id = f"{level}-{local_index}"
            position = level_index * 20 + local_index
            tasks.append(
                {
                    "example_id": task_id,
                    "metadata": {"fixed_pool_difficulty": level},
                }
            )
            group = [
                {
                    "sequence": position * 4 + sample,
                    "sample": {"audit_record": {"sample_index": sample}},
                }
                for sample in range(4)
            ]
            (original / "groups" / f"{task_id}.json").write_text(json.dumps(group))
            if local_index < 2:
                failed_by_level[level].append(task_id)
                failed_results.append({"task_id": task_id, "passed": False})
    write_jsonl(original / "tasks.jsonl", tasks)
    write_jsonl(original / "counterfactual_results.jsonl", failed_results)

    reserve_tasks = []
    selected = {}
    for level in ("simple", "moderate", "challenging"):
        selected[level] = []
        for rank in range(2):
            task_id = f"reserve-{level}-{rank}"
            selected[level].append(task_id)
            reserve_tasks.append(
                {
                    "example_id": task_id,
                    "metadata": {
                        "fixed_pool_difficulty": level,
                        "fixed_pool_reserve_rank": rank,
                    },
                }
            )
            group = [
                {
                    "sequence": rank * 4 + sample,
                    "sample": {"audit_record": {"sample_index": sample}},
                }
                for sample in range(4)
            ]
            (reserve / "groups" / f"{task_id}.json").write_text(json.dumps(group))
    write_jsonl(reserve / "tasks.jsonl", reserve_tasks)
    (reserve / "admission_summary.json").write_text(
        json.dumps({"status": "passed", "selected_replacements": selected})
    )

    manifest = assemble(original_pool=original, reserve_pool=reserve, output_dir=output)

    assert len(manifest["replacements"]) == 6
    repaired = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    assert len(repaired) == 60
    for replacement in manifest["replacements"]:
        task_id = replacement["replacement_task_id"]
        position = replacement["position"]
        group = json.loads((output / "groups" / f"{task_id}.json").read_text())
        assert [row["sequence"] for row in group] == list(
            range(position * 4, position * 4 + 4)
        )
