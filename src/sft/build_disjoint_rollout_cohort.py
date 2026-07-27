#!/usr/bin/env python3
"""Build a deterministic, disjoint BIRD-train cohort for student rollout.

Gold SQL is used only to compute a sampling-time structural difficulty proxy.  It remains in the
task record for harness-owned denotation scoring, but the rollout prompt contract must not expose
it to the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sqlglot
from sqlglot import exp


DIFFICULTIES = ("easy", "medium", "hard")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def example_id(row: dict[str, Any], *, source: str) -> str:
    value = row.get("example_id")
    if not value and isinstance(row.get("source"), dict):
        value = row["source"].get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: row is missing example_id")
    return value


def assert_unique(rows: list[dict[str, Any]], *, source: str) -> set[str]:
    ids = [example_id(row, source=source) for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{source}: duplicate example ids: {duplicates[:5]}")
    return set(ids)


def sql_complexity(sql: str) -> tuple[str, dict[str, int]]:
    """Return the frozen deterministic SQL-structure proxy used by prior BIRD cohorts."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    features = {
        "joins": sum(1 for _ in tree.find_all(exp.Join)),
        "subqueries": sum(1 for _ in tree.find_all(exp.Subquery)),
        "groups": sum(1 for _ in tree.find_all(exp.Group)),
        "havings": sum(1 for _ in tree.find_all(exp.Having)),
        "aggregates": sum(1 for node in tree.walk() if isinstance(node, exp.AggFunc)),
        "windows": sum(1 for _ in tree.find_all(exp.Window)),
        "set_operations": sum(
            1
            for node in tree.walk()
            if isinstance(node, (exp.Union, exp.Intersect, exp.Except))
        ),
        "cases": sum(1 for _ in tree.find_all(exp.Case)),
    }
    score = (
        2 * features["joins"]
        + 3 * features["subqueries"]
        + 2 * features["groups"]
        + 2 * features["havings"]
        + features["aggregates"]
        + 2 * features["windows"]
        + 3 * features["set_operations"]
        + features["cases"]
    )
    features["score"] = score
    if score <= 2:
        return "easy", features
    if score <= 5:
        return "medium", features
    return "hard", features


def copy_with_difficulty(row: dict[str, Any]) -> dict[str, Any]:
    gold_sql = row.get("gold_sql") or row.get("query")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError(f"{row.get('example_id')}: missing gold SQL for sampling proxy")
    difficulty, features = sql_complexity(gold_sql)
    copied = dict(row)
    copied["metadata"] = dict(row.get("metadata") or {})
    copied["metadata"]["difficulty_proxy"] = difficulty
    copied["metadata"]["difficulty_proxy_features"] = features
    copied["metadata"]["rollout_cohort_selection_only"] = True
    return copied


def diverse_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a bucket containing {len(rows)}")
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    random.Random(seed).shuffle(ordered)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_databases: set[str] = set()
    for unique_database_only in (True, False):
        for row in ordered:
            if len(selected) == count:
                break
            row_id = str(row["example_id"])
            if row_id in selected_ids:
                continue
            database = str(row["db_id"])
            if unique_database_only and database in used_databases:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            used_databases.add(database)
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    return selected


def build(
    source_path: Path,
    exclude_paths: list[Path],
    out_path: Path,
    manifest_path: Path,
    *,
    quotas: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    if set(quotas) != set(DIFFICULTIES) or any(value < 0 for value in quotas.values()):
        raise ValueError(f"quotas must contain non-negative values for {DIFFICULTIES}")
    if sum(quotas.values()) <= 0:
        raise ValueError("at least one row must be requested")

    source_rows = read_jsonl(source_path)
    source_ids = assert_unique(source_rows, source=str(source_path))
    excluded_ids: set[str] = set()
    exclusion_sources: list[dict[str, Any]] = []
    for path in exclude_paths:
        rows = read_jsonl(path)
        ids = assert_unique(rows, source=str(path))
        excluded_ids.update(ids)
        exclusion_sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(rows),
                "ids_in_source": len(ids & source_ids),
            }
        )

    candidates: list[dict[str, Any]] = []
    for row in source_rows:
        row_id = example_id(row, source=str(source_path))
        if row_id in excluded_ids:
            continue
        if (row.get("metadata") or {}).get("tool_round_trip") != "verified":
            continue
        candidates.append(copy_with_difficulty(row))

    buckets = {
        difficulty: [
            row
            for row in candidates
            if row["metadata"]["difficulty_proxy"] == difficulty
        ]
        for difficulty in DIFFICULTIES
    }
    selected: list[dict[str, Any]] = []
    for offset, difficulty in enumerate(DIFFICULTIES):
        selected.extend(
            diverse_sample(
                buckets[difficulty],
                quotas[difficulty],
                seed=seed + offset,
            )
        )

    selected_ids = assert_unique(selected, source="selected cohort")
    overlap = selected_ids & excluded_ids
    if overlap:
        raise RuntimeError(f"selected cohort overlaps exclusions: {sorted(overlap)[:5]}")
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("selected cohort does not match requested total")

    write_jsonl_atomic(out_path, selected)
    counts = Counter(row["metadata"]["difficulty_proxy"] for row in selected)
    manifest = {
        "method": "deterministic_disjoint_verified_bird_train_rollout_cohort",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "records": len(source_rows),
            "unique_ids": len(source_ids),
        },
        "exclusions": exclusion_sources,
        "excluded_unique_ids": len(excluded_ids),
        "selection_seed": seed,
        "requested_quotas": quotas,
        "available_after_exclusion": {
            difficulty: len(buckets[difficulty]) for difficulty in DIFFICULTIES
        },
        "counts": dict(sorted(counts.items())),
        "records": len(selected),
        "unique_ids": len(selected_ids),
        "unique_databases": len({row["db_id"] for row in selected}),
        "overlap_with_exclusions": len(overlap),
        "eligibility": "metadata.tool_round_trip == verified",
        "difficulty_mapping": {
            "kind": "deterministic_gold_sql_structure_proxy_for_selection_only",
            "easy": "score <= 2",
            "medium": "3 <= score <= 5",
            "hard": "score >= 6",
            "formula": (
                "2*joins + 3*subqueries + 2*groups + 2*havings + aggregates + "
                "2*windows + 3*set_operations + cases"
            ),
            "model_visible": False,
        },
        "output": {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
        },
        "ordering": "easy, medium, hard; deterministic shuffled order within each bucket",
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--easy", type=int, required=True)
    parser.add_argument("--medium", type=int, required=True)
    parser.add_argument("--hard", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    out_path = args.out.resolve()
    manifest = build(
        args.source.resolve(),
        [path.resolve() for path in args.exclude],
        out_path,
        (args.manifest or out_path.with_suffix(".manifest.json")).resolve(),
        quotas={"easy": args.easy, "medium": args.medium, "hard": args.hard},
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
