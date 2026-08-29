from __future__ import annotations

import json

from src.rl.prepare_epoch4_adaptive_rollout_pool import prepare_pool


def _task(index: int, prefix: str, sql: str) -> dict:
    return {
        "example_index": index,
        "example_id": f"{prefix}_train_{index:05d}",
        "db_id": f"db_{index}",
        "question": f"question {index}",
        "gold_sql": sql,
    }


def test_pool_prefers_compiler_medium_hard_and_preserves_source_mix(tmp_path) -> None:
    rows = []
    prefixes = ["bird", "spider", "synsql"]
    hard = "SELECT a, COUNT(*) FROM t JOIN u ON t.id=u.id GROUP BY a"
    easy = "SELECT x FROM t"
    for index in range(30):
        prefix = prefixes[index % 3]
        rows.append(_task(index, prefix, hard if index % 2 else easy))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "out"
    manifest = prepare_pool(tasks_path=tasks, output_dir=output, count=9, seed="test")
    assert manifest["outputs"]["rollout_tasks"]["records"] == 9
    assert manifest["selection"]["simple_fraction"] <= 1 / 3
    assert set(manifest["selection"]["source_counts"]) == {"BIRD", "Spider", "SynSQL"}
    metadata = [json.loads(line) for line in (output / "selection_metadata.jsonl").read_text().splitlines()]
    assert all("gold_sql" not in row for row in metadata)
    assert all("SELECT" not in json.dumps(row) for row in metadata)

