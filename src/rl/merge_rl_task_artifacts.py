#!/usr/bin/env python3
"""Merge disjoint trainer-ready RL task artifacts with content-hashed provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("examples", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or an examples list")
    return rows


def merge_artifacts(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        rows = read_examples(path)
        source_ids = [int(row["example_index"]) for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{path} contains duplicate example_index values")
        overlap = sorted(set(source_ids) & seen)
        if overlap:
            raise ValueError(f"RL task artifacts overlap on example_index values: {overlap[:20]}")
        if any((row.get("dataset_split") or "train") != "train" for row in rows):
            raise ValueError(f"{path} contains a non-training task")
        seen.update(source_ids)
        merged.extend(rows)
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(rows),
        })
    return merged, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples, sources = merge_artifacts(args.input)
    payload = {
        "dataset_split": "train",
        "selection": {
            "schema_version": "merged-rl-task-artifacts-v1",
            "sources": sources,
            "disjoint_example_indices": True,
        },
        "count": len(examples),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["selection"] | {"count": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
