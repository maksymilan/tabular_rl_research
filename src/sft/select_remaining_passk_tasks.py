#!/usr/bin/env python3
"""Remove already-completed task ids from a DatasetTask JSONL rollout bucket."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--completed", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    completed: set[int] = set()
    completed_by_source = {}
    for path in args.completed:
        rows = read_jsonl(path)
        ids = {int(row["example_index"]) for row in rows}
        completed.update(ids)
        completed_by_source[str(path)] = len(ids)
    source = read_jsonl(args.source)
    source_ids = [int(row["example_index"]) for row in source]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"duplicate example_index in {args.source}")
    remaining = [row for row in source if int(row["example_index"]) not in completed]
    removed = [row for row in source if int(row["example_index"]) in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remaining),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.source),
        "source_count": len(source),
        "completed_inputs": completed_by_source,
        "completed_union_count": len(completed),
        "removed_from_source": len(removed),
        "remaining": len(remaining),
        "removed_example_indices": [int(row["example_index"]) for row in removed],
        "output": str(args.out),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

