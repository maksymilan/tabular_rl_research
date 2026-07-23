#!/usr/bin/env python3
"""Add reversible aliases for SQL-R1 db_ids normalized in SynSQL databases.zip."""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


def clean_db_id(value: str) -> str:
    return value.replace("\n", "").strip()


def comparison_key(value: str) -> str:
    value = clean_db_id(value).replace("c++", "cplusplus").replace("e.g.", "eg")
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9]", "", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    args = parser.parse_args()

    existing_dirs = [path for path in args.database_root.iterdir() if path.is_dir()]
    by_key: dict[str, list[Path]] = {}
    for path in existing_dirs:
        by_key.setdefault(comparison_key(path.name), []).append(path)

    requested: set[str] = set()
    for parquet in (args.train_parquet, args.test_parquet):
        frame = pd.read_parquet(parquet, columns=["db_id"])
        requested.update(clean_db_id(value) for value in frame["db_id"])

    created: list[tuple[str, str]] = []
    failures: list[str] = []
    for db_id in sorted(requested):
        expected_db = args.database_root / db_id / f"{db_id}.sqlite"
        if expected_db.is_file():
            continue
        candidates = by_key.get(comparison_key(db_id), [])
        if len(candidates) != 1:
            failures.append(f"{db_id!r}: candidates={[p.name for p in candidates]}")
            continue
        target_dir = candidates[0]
        alias_dir = args.database_root / db_id
        if alias_dir.exists() or alias_dir.is_symlink():
            failures.append(f"{db_id!r}: alias path already exists")
            continue
        alias_dir.symlink_to(target_dir.name, target_is_directory=True)
        target_db = target_dir / f"{target_dir.name}.sqlite"
        alias_db = target_dir / f"{db_id}.sqlite"
        if not alias_db.exists():
            alias_db.symlink_to(target_db.name)
        created.append((db_id, target_dir.name))

    for alias, target in created:
        print(f"created {alias!r} -> {target!r}")
    for failure in failures:
        print(f"ERROR {failure}")
    print(f"created={len(created)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
