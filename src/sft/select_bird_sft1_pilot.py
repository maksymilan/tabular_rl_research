#!/usr/bin/env python3
"""Deterministically select the fixed BIRD SFT-1 teacher-pilot tasks.

Official BIRD train annotations do not carry a difficulty label.  This script therefore records a
transparent SQL-structure proxy used *only* for sampling; the resulting bucket never enters the
teacher-visible prompt.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from protocol import PROTOCOL_VERSION, protocol_hash  # noqa: E402

DEFAULT_SOURCE = ROOT / "data" / "eval_inputs" / "bird_train_tool_compatible.jsonl"
DEFAULT_OUT = ROOT / "data" / "eval_inputs" / "bird_train_sft1_pilot30.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_complexity(sql: str) -> tuple[str, dict[str, int]]:
    """Return a deterministic structural proxy, not a teacher-visible label."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    features = {
        "joins": sum(1 for _ in tree.find_all(exp.Join)),
        "subqueries": sum(1 for _ in tree.find_all(exp.Subquery)),
        "groups": sum(1 for _ in tree.find_all(exp.Group)),
        "havings": sum(1 for _ in tree.find_all(exp.Having)),
        "aggregates": sum(1 for node in tree.walk() if isinstance(node, exp.AggFunc)),
        "windows": sum(1 for _ in tree.find_all(exp.Window)),
        "set_operations": sum(
            1 for node in tree.walk() if isinstance(node, (exp.Union, exp.Intersect, exp.Except))
        ),
        "cases": sum(1 for _ in tree.find_all(exp.Case)),
    }
    score = (
        features["joins"] * 2
        + features["subqueries"] * 3
        + features["groups"] * 2
        + features["havings"] * 2
        + features["aggregates"]
        + features["windows"] * 2
        + features["set_operations"] * 3
        + features["cases"]
    )
    features["score"] = score
    if score <= 2:
        return "easy", features
    if score <= 5:
        return "medium", features
    return "hard", features


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose(
    records: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_ids = excluded_ids or set()
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    for record in records:
        if record.get("metadata", {}).get("tool_round_trip") != "verified":
            continue
        if record.get("example_id") in excluded_ids:
            continue
        difficulty, features = sql_complexity(record["gold_sql"])
        record = dict(record)
        record["metadata"] = dict(record.get("metadata") or {})
        record["metadata"]["difficulty_proxy"] = difficulty
        record["metadata"]["difficulty_proxy_features"] = features
        buckets[difficulty].append(record)

    rng = random.Random(seed)
    for difficulty in buckets:
        buckets[difficulty].sort(key=lambda record: record["example_id"])
        rng.shuffle(buckets[difficulty])

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_dbs: set[str] = set()
    # First pass maximizes DB diversity globally; second pass is deterministic fallback.
    for difficulty, quota in quotas.items():
        for unique_only in (True, False):
            for record in buckets[difficulty]:
                if sum(item["metadata"]["difficulty_proxy"] == difficulty for item in selected) >= quota:
                    break
                if record["example_id"] in selected_ids:
                    continue
                if unique_only and record["db_id"] in used_dbs:
                    continue
                selected.append(record)
                selected_ids.add(record["example_id"])
                used_dbs.add(record["db_id"])
            if sum(item["metadata"]["difficulty_proxy"] == difficulty for item in selected) >= quota:
                break
        got = sum(item["metadata"]["difficulty_proxy"] == difficulty for item in selected)
        if got != quota:
            raise ValueError(f"could not select {quota} {difficulty} tasks; only found {got}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--exclude-examples-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable JSONL whose example_id values are reserved or already used and excluded"
        ),
    )
    parser.add_argument(
        "--bucket-out-dir",
        type=Path,
        default=None,
        help="optionally also write easy.jsonl, medium.jsonl, and hard.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total",
        type=int,
        default=30,
        help="Number of tasks to select; must be a positive multiple of 10 for the 4:3:3 split.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    manifest_path = (args.manifest or out.with_suffix(".manifest.json")).resolve()
    if args.total <= 0 or args.total % 10:
        parser.error("--total must be a positive multiple of 10 for the exact 4:3:3 split")
    unit = args.total // 10
    quotas = {"easy": 4 * unit, "medium": 3 * unit, "hard": 3 * unit}
    records = read_jsonl(source)
    exclude_paths = [path.resolve() for path in args.exclude_examples_file]
    excluded_ids = set()
    for exclude_path in exclude_paths:
        for record in read_jsonl(exclude_path):
            example_id = record.get("example_id")
            if not example_id and isinstance(record.get("source"), dict):
                example_id = record["source"].get("example_id")
            if example_id:
                excluded_ids.add(str(example_id))
    selected = choose(records, quotas, args.seed, excluded_ids=excluded_ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    bucket_outputs: dict[str, str] = {}
    if args.bucket_out_dir:
        bucket_dir = args.bucket_out_dir.resolve()
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for difficulty in quotas:
            bucket_path = bucket_dir / f"{difficulty}.jsonl"
            with bucket_path.open("w", encoding="utf-8") as f:
                for record in selected:
                    if record["metadata"]["difficulty_proxy"] == difficulty:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            bucket_outputs[difficulty] = str(bucket_path)

    db_counts = collections.Counter(record["db_id"] for record in selected)
    manifest = {
        "dataset": "BIRD train compatible",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "source_difficulty_audit": {
            "field": None,
            "actual_values": [None],
            "finding": "official BIRD train and compatible adapter records do not contain difficulty labels",
        },
        "difficulty_mapping": {
            "kind": "deterministic_gold_sql_structure_proxy_for_selection_only",
            "easy": "complexity score <= 2",
            "medium": "3 <= complexity score <= 5",
            "hard": "complexity score >= 6",
            "formula": "2*joins + 3*subqueries + 2*groups + 2*havings + aggregates + 2*windows + 3*set_operations + cases",
            "teacher_visible": False,
        },
        "selection_seed": args.seed,
        "excluded_examples": {
            "sources": [str(path) for path in exclude_paths],
            "count": len(excluded_ids),
        },
        "quotas": quotas,
        "counts": collections.Counter(record["metadata"]["difficulty_proxy"] for record in selected),
        "database_distribution": dict(sorted(db_counts.items())),
        "unique_databases": len(db_counts),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "tasks": [
            {
                "example_id": record["example_id"],
                "example_index": record["example_index"],
                "db_id": record["db_id"],
                "difficulty": record["metadata"]["difficulty_proxy"],
                "difficulty_features": record["metadata"]["difficulty_proxy_features"],
            }
            for record in selected
        ],
        "output": str(out),
        "bucket_outputs": bucket_outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "manifest": str(manifest_path), "counts": manifest["counts"],
                      "unique_databases": manifest["unique_databases"]}, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
