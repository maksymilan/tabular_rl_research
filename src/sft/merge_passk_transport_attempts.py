#!/usr/bin/env python3
"""Merge pass@k shards while excluding transport-incomplete attempts from semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transport_complete(record: dict[str, Any]) -> bool:
    samples = record.get("samples") or []
    expected = int(record.get("n_samples") or 0)
    return bool(samples) and len(samples) == expected and all(
        sample.get("failure_type") != "api_error" for sample in samples
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    task_ids = [int(task["example_index"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task source contains duplicate example_index values")
    expected_ids = set(task_ids)

    complete_by_id: dict[int, tuple[Path, dict[str, Any]]] = {}
    incomplete: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}
    for path in args.input:
        counts = Counter()
        for record in read_jsonl(path):
            index = int(record["example_index"])
            if index not in expected_ids:
                raise ValueError(f"{path}: example_index {index} is outside task source")
            counts["records"] += 1
            if transport_complete(record):
                counts["transport_complete"] += 1
                if index in complete_by_id:
                    previous = complete_by_id[index][0]
                    raise ValueError(f"multiple transport-complete attempts for {index}: {previous}, {path}")
                complete_by_id[index] = (path, record)
            else:
                counts["transport_incomplete"] += 1
                incomplete.append({
                    "example_index": index,
                    "source": str(path),
                    "attempted_samples": int(record.get("attempted_samples") or 0),
                    "api_error_samples": sum(
                        sample.get("failure_type") == "api_error" for sample in record.get("samples") or []
                    ),
                })
        source_counts[str(path)] = dict(counts)

    missing = sorted(expected_ids - set(complete_by_id))
    if missing:
        raise ValueError(f"missing transport-complete attempt for {len(missing)} tasks: {missing[:20]}")
    selected = [complete_by_id[index][1] for index in task_ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
        encoding="utf-8",
    )
    manifest = {
        "tasks": str(args.tasks),
        "task_count": len(tasks),
        "inputs": [str(path) for path in args.input],
        "source_counts": source_counts,
        "selected_transport_complete_records": len(selected),
        "excluded_transport_incomplete_attempts": len(incomplete),
        "excluded_transport_incomplete_audit": incomplete,
        "unique_selected_example_indices": len({int(row["example_index"]) for row in selected}),
        "output": str(args.out),
        "output_sha256": sha256(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "excluded_transport_incomplete_audit"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
