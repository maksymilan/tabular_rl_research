#!/usr/bin/env python3
"""Select a fresh easy/medium batch plus grounded-incomplete hard rescue tasks."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from select_bird_sft1_pilot import read_jsonl, sql_complexity
from protocol import PROTOCOL_VERSION, protocol_hash


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def difficulty(record: dict[str, Any]) -> str:
    value = (record.get("metadata") or {}).get("difficulty_proxy")
    return value or sql_complexity(record["gold_sql"])[0]


def eligible_ids(paths: list[Path]) -> set[str]:
    eligible: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            diagnostics = row.get("diagnostics") or {}
            if row.get("correct") and diagnostics.get("deterministic_grounding_complete"):
                eligible.add(str(row["trajectory_id"]))
    return eligible


def diverse_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: row["example_id"])
    rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_dbs: set[str] = set()
    for unique_db_only in (True, False):
        for row in rows:
            if len(selected) >= count:
                break
            if row["example_id"] in selected_ids:
                continue
            if unique_db_only and row["db_id"] in used_dbs:
                continue
            selected.append(row)
            selected_ids.add(row["example_id"])
            used_dbs.add(row["db_id"])
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but only selected {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--used-file", type=Path, action="append", default=[])
    parser.add_argument("--eligible-scored", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bucket-out-dir", type=Path, required=True)
    parser.add_argument("--easy", type=int, default=400)
    parser.add_argument("--medium", type=int, default=300)
    parser.add_argument("--hard", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    source = args.source.resolve()
    records = [
        row for row in read_jsonl(source)
        if (row.get("metadata") or {}).get("tool_round_trip") == "verified"
    ]
    by_id = {str(row["example_id"]): row for row in records}
    used_ids = {
        str(row["example_id"])
        for path in args.used_file
        for row in read_jsonl(path.resolve())
        if row.get("example_id")
    }
    grounded_ids = eligible_ids([path.resolve() for path in args.eligible_scored])

    remaining: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        if row["example_id"] not in used_ids:
            remaining[difficulty(row)].append(row)

    fresh_easy = diverse_sample(remaining["easy"], args.easy, args.seed + 1)
    fresh_medium = diverse_sample(remaining["medium"], args.medium, args.seed + 2)
    fresh_hard_count = min(args.hard, len(remaining["hard"]))
    fresh_hard = diverse_sample(remaining["hard"], fresh_hard_count, args.seed + 3)
    rescue_count = args.hard - fresh_hard_count
    rescue_candidates = [
        by_id[example_id]
        for example_id in sorted(used_ids & set(by_id))
        if difficulty(by_id[example_id]) == "hard" and example_id not in grounded_ids
    ]
    rescue_hard = diverse_sample(rescue_candidates, rescue_count, args.seed + 4)

    selected: list[dict[str, Any]] = []
    groups = (
        (fresh_easy, "fresh"),
        (fresh_medium, "fresh"),
        (fresh_hard, "fresh"),
        (rescue_hard, "hard_rescue_no_grounded_success"),
    )
    for rows, origin in groups:
        for row in rows:
            copied = dict(row)
            copied["metadata"] = dict(row.get("metadata") or {})
            bucket, features = sql_complexity(row["gold_sql"])
            copied["metadata"]["difficulty_proxy"] = bucket
            copied["metadata"]["difficulty_proxy_features"] = features
            copied["metadata"]["scale_selection_origin"] = origin
            selected.append(copied)

    out = args.out.resolve()
    write_jsonl(out, selected)
    bucket_dir = args.bucket_out_dir.resolve()
    bucket_outputs: dict[str, str] = {}
    for bucket in ("easy", "medium", "hard"):
        path = bucket_dir / f"{bucket}.jsonl"
        write_jsonl(path, [row for row in selected if difficulty(row) == bucket])
        bucket_outputs[bucket] = str(path)

    counts = collections.Counter(difficulty(row) for row in selected)
    origins = collections.Counter((row.get("metadata") or {}).get("scale_selection_origin") for row in selected)
    manifest = {
        "dataset": "BIRD train compatible",
        "source": str(source),
        "source_sha256": sha256(source),
        "selection_seed": args.seed,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "quotas": {"easy": args.easy, "medium": args.medium, "hard": args.hard},
        "counts": dict(sorted(counts.items())),
        "selection_origins": dict(sorted(origins.items())),
        "used_sources": [str(path.resolve()) for path in args.used_file],
        "used_unique_ids": len(used_ids),
        "grounded_eligible_ids_excluded_from_rescue": len(grounded_ids),
        "remaining_before_selection": {
            key: len(remaining[key]) for key in ("easy", "medium", "hard")
        },
        "unique_databases": len({row["db_id"] for row in selected}),
        "output": str(out),
        "output_sha256": sha256(out),
        "bucket_outputs": bucket_outputs,
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
