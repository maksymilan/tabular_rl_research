#!/usr/bin/env python3
"""Replace counterfactual-failing task groups with admitted deterministic reserves."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("instance_id"))


def assemble(
    *,
    original_pool: Path,
    reserve_pool: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty repaired pool: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir()

    original_tasks = load_jsonl(original_pool / "tasks.jsonl")
    reserve_tasks = {task_id(row): row for row in load_jsonl(reserve_pool / "tasks.jsonl")}
    admission = json.loads((reserve_pool / "admission_summary.json").read_text())
    if admission.get("status") != "passed":
        raise ValueError("reserve admission summary is not passed")
    failed_results = [
        row
        for row in load_jsonl(original_pool / "counterfactual_results.jsonl")
        if not row["passed"]
    ]
    failed_ids = {str(row["task_id"]) for row in failed_results}
    if not failed_ids:
        raise ValueError("original pool has no counterfactual-failing task groups")

    failed_by_level: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for position, row in enumerate(original_tasks):
        original_id = task_id(row)
        if original_id in failed_ids:
            level = str((row.get("metadata") or {})["fixed_pool_difficulty"])
            failed_by_level[level].append((position, original_id))

    replacement_by_position: dict[int, tuple[str, str]] = {}
    for level in ("simple", "moderate", "challenging"):
        failed = sorted(failed_by_level[level])
        selected = list(admission["selected_replacements"][level])
        if len(failed) != len(selected):
            raise ValueError(
                f"replacement count mismatch for {level}: failed={len(failed)} selected={len(selected)}"
            )
        for (position, original_id), replacement_id in zip(failed, selected, strict=True):
            replacement_by_position[position] = (original_id, replacement_id)

    repaired_tasks: list[dict[str, Any]] = []
    replacements = []
    for position, original_task in enumerate(original_tasks):
        original_id = task_id(original_task)
        if position not in replacement_by_position:
            repaired_tasks.append(original_task)
            shutil.copy2(
                original_pool / "groups" / f"{original_id}.json",
                groups_dir / f"{original_id}.json",
            )
            continue
        failed_id, replacement_id = replacement_by_position[position]
        replacement = json.loads(json.dumps(reserve_tasks[replacement_id]))
        replacement.setdefault("metadata", {})["fixed_pool_replaces_task_id"] = failed_id
        replacement["metadata"]["fixed_pool_original_position"] = position
        repaired_tasks.append(replacement)
        source_group = json.loads(
            (reserve_pool / "groups" / f"{replacement_id}.json").read_text()
        )
        if len(source_group) != 4:
            raise ValueError(f"reserve group is not K4: {replacement_id}")
        for sample_index, row in enumerate(source_group):
            row["sequence"] = position * 4 + sample_index
            audit = row["sample"]["audit_record"]
            if int(audit["sample_index"]) != sample_index:
                raise ValueError(f"unexpected reserve sample index: {replacement_id}")
        (groups_dir / f"{replacement_id}.json").write_text(
            json.dumps(source_group, ensure_ascii=False, separators=(",", ":"))
        )
        replacements.append(
            {
                "position": position,
                "difficulty": replacement["metadata"]["fixed_pool_difficulty"],
                "removed_task_id": failed_id,
                "replacement_task_id": replacement_id,
            }
        )

    tasks_path = output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as target:
        for row in repaired_tasks:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = Counter(
        str((row.get("metadata") or {})["fixed_pool_difficulty"])
        for row in repaired_tasks
    )
    if counts != Counter({"simple": 20, "moderate": 20, "challenging": 20}):
        raise ValueError(f"repaired pool lost difficulty balance: {counts}")
    manifest = {
        "schema_version": "fixed-pool-task-selection-causal-repair-v1",
        "tasks": len(repaired_tasks),
        "difficulty_counts": dict(sorted(counts.items())),
        "task_ids": [task_id(row) for row in repaired_tasks],
        "output": str(tasks_path.resolve()),
        "output_sha256": sha256_file(tasks_path),
        "original_pool": str(original_pool.resolve()),
        "original_tasks_sha256": sha256_file(original_pool / "tasks.jsonl"),
        "reserve_pool": str(reserve_pool.resolve()),
        "reserve_admission_sha256": sha256_file(reserve_pool / "admission_summary.json"),
        "replacements": replacements,
        "replacement_rule": (
            "remove every task group containing a correct trajectory that failed the unchanged "
            "counterfactual-completeness gate; replace in-place with the first causally admitted "
            "reserve task of the same difficulty"
        ),
        "gold_visibility": "hidden harness metadata only; never rendered to the actor",
    }
    (output_dir / "task_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-pool", required=True, type=Path)
    parser.add_argument("--reserve-pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = assemble(
        original_pool=args.original_pool,
        reserve_pool=args.reserve_pool,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
