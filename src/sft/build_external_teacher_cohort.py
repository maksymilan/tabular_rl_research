#!/usr/bin/env python3
"""Build a fixed-core plus disjoint-complement external-teacher cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
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


def example_id(row: dict[str, Any]) -> str:
    value = row.get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("cohort row is missing example_id")
    return value


def difficulty(row: dict[str, Any]) -> str:
    value = (row.get("metadata") or {}).get("difficulty_proxy")
    if value not in DIFFICULTIES:
        raise ValueError(f"{example_id(row)}: invalid difficulty proxy {value!r}")
    return value


def assert_unique(rows: list[dict[str, Any]], *, source: str) -> set[str]:
    ids = [example_id(row) for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{source}: duplicate example ids: {duplicates[:5]}")
    return set(ids)


def build(
    pool_path: Path,
    fixed_core_path: Path,
    additional_out: Path,
    combined_out: Path,
    bucket_out_dir: Path,
) -> dict[str, Any]:
    pool = read_jsonl(pool_path)
    fixed_core = read_jsonl(fixed_core_path)
    pool_ids = assert_unique(pool, source="pool")
    core_ids = assert_unique(fixed_core, source="fixed core")
    missing = sorted(core_ids - pool_ids)
    if missing:
        raise ValueError(f"fixed core is not a subset of the pool: {missing[:5]}")

    additional = [row for row in pool if example_id(row) not in core_ids]
    combined = [*fixed_core, *additional]
    combined_ids = assert_unique(combined, source="combined cohort")
    if combined_ids != pool_ids:
        raise ValueError("fixed core plus complement does not reconstruct the pool")

    write_jsonl_atomic(additional_out, additional)
    write_jsonl_atomic(combined_out, combined)
    bucket_out_dir.mkdir(parents=True, exist_ok=True)
    bucket_outputs: dict[str, dict[str, Any]] = {}
    for name in DIFFICULTIES:
        path = bucket_out_dir / f"{name}.jsonl"
        rows = [row for row in additional if difficulty(row) == name]
        write_jsonl_atomic(path, rows)
        bucket_outputs[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "records": len(rows),
        }

    manifest = {
        "method": "fixed_core_plus_exact_disjoint_pool_complement",
        "pool": {
            "path": str(pool_path),
            "sha256": sha256(pool_path),
            "records": len(pool),
        },
        "fixed_core": {
            "path": str(fixed_core_path),
            "sha256": sha256(fixed_core_path),
            "records": len(fixed_core),
            "difficulty": dict(sorted(Counter(difficulty(row) for row in fixed_core).items())),
        },
        "additional": {
            "path": str(additional_out),
            "sha256": sha256(additional_out),
            "records": len(additional),
            "difficulty": dict(sorted(Counter(difficulty(row) for row in additional).items())),
            "unique_databases": len({row["db_id"] for row in additional}),
        },
        "combined": {
            "path": str(combined_out),
            "sha256": sha256(combined_out),
            "records": len(combined),
            "difficulty": dict(sorted(Counter(difficulty(row) for row in combined).items())),
            "unique_databases": len({row["db_id"] for row in combined}),
            "ordering": "fixed core in its frozen order, then pool-order complement",
        },
        "overlap_fixed_additional": len(
            core_ids & {example_id(row) for row in additional}
        ),
        "bucket_outputs": bucket_outputs,
        "difficulty_is_teacher_visible": False,
    }
    manifest_path = combined_out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--fixed-core", type=Path, required=True)
    parser.add_argument("--additional-out", type=Path, required=True)
    parser.add_argument("--combined-out", type=Path, required=True)
    parser.add_argument("--bucket-out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        args.pool.resolve(),
        args.fixed_core.resolve(),
        args.additional_out.resolve(),
        args.combined_out.resolve(),
        args.bucket_out_dir.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
