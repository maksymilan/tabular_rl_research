#!/usr/bin/env python3
"""Build deterministic, failure-stratified indices for environment-feedback gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = ("adjacent_loop", "protocol_error", "tool_error", "clean_control")
STAGE_COUNTS = (12, 30, 60)


def sample_record(record: dict[str, Any]) -> dict[str, Any]:
    samples = record.get("samples") or []
    if len(samples) != 1:
        raise ValueError(
            f"example {record.get('example_index')}: expected exactly one greedy sample"
        )
    return samples[0]


def has_adjacent_identical_actions(sample: dict[str, Any]) -> bool:
    previous: str | None = None
    for turn in sample.get("turns") or []:
        parsed = turn.get("parsed")
        if not isinstance(parsed, dict) or not parsed.get("tool"):
            previous = None
            continue
        current = json.dumps(
            {
                "tool": parsed["tool"],
                "arguments": parsed.get("arguments") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if previous == current:
            return True
        previous = current
    return False


def category(record: dict[str, Any]) -> str | None:
    sample = sample_record(record)
    if sample.get("failure_type") == "max_steps" and has_adjacent_identical_actions(sample):
        return "adjacent_loop"
    if sample.get("failure_type") == "protocol_error":
        return "protocol_error"
    if sample.get("failure_type") in {"argument_validation_error", "execution_error"}:
        return "tool_error"
    if sample.get("correct") and not sample.get("error_events"):
        return "clean_control"
    return None


def diverse_prefix(records: list[dict[str, Any]], size: int) -> list[int]:
    """Round-robin databases so early gates are not dominated by source ordering."""
    by_db: dict[str, list[int]] = defaultdict(list)
    for record in sorted(records, key=lambda item: int(item["example_index"])):
        by_db[str(record["db_id"])].append(int(record["example_index"]))
    selected: list[int] = []
    depth = 0
    databases = sorted(by_db)
    while len(selected) < size:
        added = False
        for db_id in databases:
            if depth < len(by_db[db_id]):
                selected.append(by_db[db_id][depth])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != size:
        raise ValueError(f"requested {size} records but only selected {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-all-jsonl", required=True)
    parser.add_argument(
        "--examples-json",
        default="",
        help="optional full DatasetTask JSON/JSONL used to emit stage-specific task files",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.greedy_all_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    indexed_examples: dict[int, dict[str, Any]] = {}
    if args.examples_json:
        examples_path = Path(args.examples_json)
        if examples_path.suffix == ".jsonl":
            examples = [
                json.loads(line)
                for line in examples_path.read_text().splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(examples_path.read_text())
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        indexed_examples = {
            int(example.get("example_index", local_index)): example
            for local_index, example in enumerate(examples)
        }
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    for record in records:
        name = category(record)
        if name:
            grouped[name].append(record)

    stages: dict[str, dict[str, Any]] = {}
    for stage_number, per_category in enumerate(STAGE_COUNTS, start=1):
        category_indices = {
            name: diverse_prefix(grouped[name], per_category)
            for name in CATEGORIES
        }
        indices = sorted(
            index
            for values in category_indices.values()
            for index in values
        )
        path = output_dir / f"stage{stage_number}_{len(indices)}.json"
        path.write_text(
            json.dumps(indices, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage_manifest = {
            "path": str(path),
            "count": len(indices),
            "per_category": per_category,
            "category_indices": category_indices,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if indexed_examples:
            missing = sorted(set(indices) - set(indexed_examples))
            if missing:
                raise ValueError(f"examples input is missing selected indices: {missing}")
            task_path = output_dir / f"stage{stage_number}_{len(indices)}_tasks.jsonl"
            selected = set(indices)
            task_path.write_text(
                "".join(
                    json.dumps(example, ensure_ascii=False) + "\n"
                    for index, example in indexed_examples.items()
                    if index in selected
                ),
                encoding="utf-8",
            )
            stage_manifest.update({
                "tasks_path": str(task_path),
                "tasks_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            })
        stages[f"stage{stage_number}"] = stage_manifest

    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "examples_json": args.examples_json or None,
        "selection_inputs": [
            "example_index",
            "db_id",
            "sample.correct",
            "sample.failure_type",
            "sample.error_events",
            "sample.turns[].parsed.tool",
            "sample.turns[].parsed.arguments",
        ],
        "gold_used_for_selection": False,
        "available_counts": {name: len(grouped[name]) for name in CATEGORIES},
        "stages": stages,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
