#!/usr/bin/env python3
"""Select a deterministic SFT-disjoint, SQL-shape-stratified BIRD-train audit cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_id(row: dict[str, Any]) -> str:
    return str(
        row.get("example_id")
        or row.get("instance_id")
        or f"bird_train_{int(row['example_index']):05d}"
    )


def sql_shape(sql: str) -> tuple[str, dict[str, int]]:
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception:
        return "challenging", {"parse_error": 1}
    joins = sum(1 for _ in tree.find_all(exp.Join))
    aggregates = sum(1 for _ in tree.find_all(exp.AggFunc))
    subqueries = sum(1 for _ in tree.find_all(exp.Subquery))
    predicates = 1 if tree.find(exp.Where) is not None else 0
    groups = 1 if tree.find(exp.Group) is not None else 0
    ranks = 1 if tree.find(exp.Order) is not None or tree.find(exp.Limit) is not None else 0
    sets = int(isinstance(tree, (exp.Union, exp.Intersect, exp.Except)))
    complexity = joins + aggregates + 2 * subqueries + predicates + groups + ranks + 2 * sets
    if subqueries or sets or joins >= 2 or complexity >= 5:
        level = "challenging"
    elif complexity >= 2:
        level = "moderate"
    else:
        level = "simple"
    return level, {
        "joins": joins,
        "aggregates": aggregates,
        "subqueries": subqueries,
        "predicates": predicates,
        "groups": groups,
        "ranks": ranks,
        "sets": sets,
        "complexity": complexity,
    }


def stable_rank(seed: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}:{task_id(row)}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--sft-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-level", type=int, default=32)
    parser.add_argument("--max-per-db", type=int, default=6)
    parser.add_argument("--seed", default="qwen3-cp787-semantic-rank-audit-v1")
    args = parser.parse_args()
    if args.per_level < 1 or args.max_per_db < 1:
        parser.error("per-level and max-per-db must be positive")
    source = read_jsonl(args.tasks)
    if any(row.get("split") != "train" for row in source):
        raise SystemExit("source cohort must contain train records only")
    sft_episode_ids = {
        str(row["source_episode_id"])
        for row in read_jsonl(args.sft_index)
    }
    eligible = [row for row in source if task_id(row) not in sft_episode_ids]
    by_level: dict[str, list[dict[str, Any]]] = {level: [] for level in ("simple", "moderate", "challenging")}
    shapes: dict[str, dict[str, int]] = {}
    for row in eligible:
        level, shape = sql_shape(str(row.get("gold_sql") or row.get("query") or ""))
        by_level[level].append(row)
        shapes[task_id(row)] = shape
    selected: list[dict[str, Any]] = []
    database_counts: Counter[str] = Counter()
    for level in ("simple", "moderate", "challenging"):
        ranked = sorted(by_level[level], key=lambda row: stable_rank(args.seed, row))
        chosen = []
        for row in ranked:
            database = str(row.get("db_id"))
            if database_counts[database] >= args.max_per_db:
                continue
            copy = dict(row)
            metadata = dict(copy.get("metadata") or {})
            metadata.update({
                "semantic_rank_audit_level": level,
                "semantic_rank_audit_sql_shape": shapes[task_id(row)],
                "sft_episode_disjoint": True,
            })
            copy["metadata"] = metadata
            chosen.append(copy)
            database_counts[database] += 1
            if len(chosen) == args.per_level:
                break
        if len(chosen) != args.per_level:
            raise RuntimeError(f"could select only {len(chosen)} {level} tasks")
        selected.extend(chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "qwen3-cp787-semantic-rank-audit-cohort-v1",
        "status": "frozen_diagnostic_cohort",
        "dataset_split": "train",
        "source_tasks": str(args.tasks.resolve()),
        "source_tasks_sha256": sha256_file(args.tasks),
        "sft_index": str(args.sft_index.resolve()),
        "sft_index_sha256": sha256_file(args.sft_index),
        "source_records": len(source),
        "sft_episode_ids": len(sft_episode_ids),
        "eligible_sft_disjoint_records": len(eligible),
        "selected_records": len(selected),
        "per_level": args.per_level,
        "level_counts": dict(Counter((row.get("metadata") or {})["semantic_rank_audit_level"] for row in selected)),
        "database_counts": dict(sorted(database_counts.items())),
        "max_per_db": args.max_per_db,
        "seed": args.seed,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "all_train": all(row.get("split") == "train" for row in selected),
        "all_sft_episode_disjoint": all(task_id(row) not in sft_episode_ids for row in selected),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

