#!/usr/bin/env python3
"""Split a fixed rollout cohort into deterministic, disjoint round-robin shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def identity(row: dict[str, Any]) -> str:
    value = row.get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("cohort row is missing example_id")
    return value


def shard(
    source_path: Path,
    out_dir: Path,
    *,
    shards: int,
    prefix: str,
) -> dict[str, Any]:
    if shards < 2:
        raise ValueError("shards must be at least 2")
    rows = read_jsonl(source_path)
    ids = [identity(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source cohort contains duplicate example ids")

    buckets = [[] for _ in range(shards)]
    for position, row in enumerate(rows):
        buckets[position % shards].append(row)
    sizes = [len(bucket) for bucket in buckets]
    if max(sizes, default=0) - min(sizes, default=0) > 1:
        raise RuntimeError("round-robin shards are unexpectedly imbalanced")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    all_output_ids: set[str] = set()
    for shard_index, bucket in enumerate(buckets):
        path = out_dir / f"{prefix}.shard-{shard_index:02d}-of-{shards:02d}.jsonl"
        write_jsonl_atomic(path, bucket)
        bucket_ids = {identity(row) for row in bucket}
        overlap = all_output_ids & bucket_ids
        if overlap:
            raise RuntimeError(f"shards overlap: {sorted(overlap)[:5]}")
        all_output_ids.update(bucket_ids)
        outputs.append(
            {
                "shard_index": shard_index,
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(bucket),
                "difficulty": dict(
                    sorted(
                        Counter(
                            (row.get("metadata") or {}).get(
                                "difficulty_proxy",
                                "unknown",
                            )
                            for row in bucket
                        ).items()
                    )
                ),
                "unique_databases": len({row["db_id"] for row in bucket}),
            }
        )
    if all_output_ids != set(ids):
        raise RuntimeError("shards do not reconstruct the source cohort")

    manifest = {
        "method": "deterministic_global_round_robin",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "records": len(rows),
        },
        "shards": outputs,
        "coverage": {
            "source_unique_ids": len(set(ids)),
            "output_unique_ids": len(all_output_ids),
            "missing": 0,
            "overlap": 0,
        },
        "ordering": (
            "source order is preserved within each shard; global position modulo shard count"
        ),
    }
    manifest_path = out_dir / f"{prefix}.shards.manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    manifest = shard(
        args.source.resolve(),
        args.out_dir.resolve(),
        shards=args.shards,
        prefix=args.prefix,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
