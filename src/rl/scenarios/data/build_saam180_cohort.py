#!/usr/bin/env python3
"""Extend the frozen 60-question SAAM cohort to a larger, single-variable cohort.

The 2026-09-14 correctness-only SAAM arm trained on 60 BIRD questions drawn from
the 2-6 band.  This CLI builds a strict superset of that cohort by appending
fresh 2-6 candidates from the versioned screened inventory, so the only training
variable that changes is the amount of data.

Selection rules (deterministic, recorded in the manifest):

* base cohort rows are kept verbatim and first;
* extension rows come from the pool with `correct_count` inside the requested
  band, excluding any task already present in the base cohort (by db_id and
  question digest) and any task flagged as part of the SFT training view;
* per-bucket counts follow largest-remainder allocation over the available
  candidates, and within a bucket the first N by `sha256(seed + NUL + task_id)`
  win;
* `db_path` is rewritten to the training host layout when a prefix is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl.shared.io import atomic_write_text, read_jsonl, sha256_file  # noqa: E402


SELECTION_SCHEMA = "atomic-v26-saam-cohort-extension-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cohort", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extension-records", type=int, required=True)
    parser.add_argument("--correct-count-min", type=int, default=2)
    parser.add_argument("--correct-count-max", type=int, default=6)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--db-path-prefix", default="")
    parser.add_argument("--exclude-known-set", default="checkpoint6380_sft_training_view")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="restrict the extension to these dataset labels (repeatable)",
    )
    return parser.parse_args()


def identity(row: dict) -> tuple[str, str]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"task {row.get('example_id')} has no question text")
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    return (str(row.get("db_id")), digest)


def task_identity(row: dict) -> str:
    value = row.get("example_id") or row.get("instance_id") or row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("pool row is missing example_id/instance_id/task_id")
    return value


def screen_correct_count(row: dict) -> int:
    metadata = row.get("metadata") or {}
    block = metadata.get("screened_rl_candidate_pool")
    if not isinstance(block, dict) or "screen_correct_count" not in block:
        raise ValueError(f"{task_identity(row)}: missing screened_rl_candidate_pool metadata")
    return int(block["screen_correct_count"])


def selection_order(seed: str, task_ids: list[str]) -> list[str]:
    return sorted(
        task_ids,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(),
    )


def allocate(total: int, availability: dict[int, int]) -> dict[int, int]:
    """Largest-remainder allocation of `total` across buckets, capped by supply."""
    buckets = sorted(availability)
    supply = sum(availability[bucket] for bucket in buckets)
    if total > supply:
        raise ValueError(f"requested {total} > available {supply}")
    quota = {bucket: 0 for bucket in buckets}
    if total <= 0:
        return quota
    ideal = {bucket: total * availability[bucket] / supply for bucket in buckets}
    for bucket in buckets:
        quota[bucket] = min(availability[bucket], int(ideal[bucket]))
    remainder = total - sum(quota.values())
    order = sorted(
        buckets,
        key=lambda bucket: (-(ideal[bucket] - int(ideal[bucket])), -availability[bucket], bucket),
    )
    while remainder > 0:
        progressed = False
        for bucket in order:
            if remainder == 0:
                break
            if quota[bucket] < availability[bucket]:
                quota[bucket] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            raise ValueError("allocation failed to place all requested records")
    return quota


def main() -> int:
    args = parse_args()
    base_rows = read_jsonl(args.base_cohort)
    pool_rows = read_jsonl(args.source_pool)
    excluded_by_identity = {identity(row) for row in base_rows}

    buckets: dict[int, list[dict]] = {}
    for row in pool_rows:
        if args.dataset and str(row.get("dataset")) not in set(args.dataset):
            continue
        block = (row.get("metadata") or {}).get("screened_rl_candidate_pool") or {}
        if args.exclude_known_set:
            membership = block.get("known_set_membership") or {}
            if membership.get(args.exclude_known_set):
                continue
        count = screen_correct_count(row)
        if not args.correct_count_min <= count <= args.correct_count_max:
            continue
        if identity(row) in excluded_by_identity:
            continue
        buckets.setdefault(count, []).append(row)

    availability = {bucket: len(rows) for bucket, rows in sorted(buckets.items())}
    quota = allocate(args.extension_records, availability)

    extension_rows: list[dict] = []
    selected_ids: dict[int, list[str]] = {}
    for bucket, take in quota.items():
        if take == 0:
            selected_ids[bucket] = []
            continue
        ordered = selection_order(args.seed, [task_identity(row) for row in buckets[bucket]])
        chosen = set(ordered[:take])
        selected_ids[bucket] = sorted(chosen)
        extension_rows.extend(row for row in buckets[bucket] if task_identity(row) in chosen)

    if len(extension_rows) != args.extension_records:
        raise ValueError(
            f"selected {len(extension_rows)} extension rows != {args.extension_records}"
        )
    extension_rows.sort(key=lambda row: task_identity(row))

    merged: list[dict] = []
    for position, row in enumerate(base_rows + extension_rows):
        item = dict(row)
        if args.db_path_prefix:
            db_path = str(item.get("db_path") or "")
            if not db_path.startswith("/"):
                item["db_path"] = str(Path(args.db_path_prefix) / db_path)
        item["example_index"] = position
        item["index"] = position
        item["split"] = "train"
        item["example_id"] = task_identity(item)
        item["instance_id"] = task_identity(item)
        metadata = dict(item.get("metadata") or {})
        record = {
            "schema_version": SELECTION_SCHEMA,
            "selection_seed": args.seed,
            "role": "original_cohort" if position < len(base_rows) else "extension",
            "correct_count_band": [args.correct_count_min, args.correct_count_max],
            "source_pool": str(args.source_pool),
            "source_pool_sha256": sha256_file(args.source_pool),
        }
        if position < len(base_rows):
            record["base_cohort"] = str(args.base_cohort)
            record["base_cohort_sha256"] = sha256_file(args.base_cohort)
        else:
            record["screen_correct_count"] = screen_correct_count(item)
        metadata["saam_cohort_extension"] = record
        item["metadata"] = metadata
        merged.append(item)

    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in merged
    )
    atomic_write_text(args.output, text)

    histogram = Counter(screen_correct_count(row) for row in extension_rows) if extension_rows else Counter()
    base_histogram = Counter(
        int(((row.get("metadata") or {}).get("saam_cohort_extension") or {}).get("screen_correct_count", -1))
        for row in merged[: len(base_rows)]
    )
    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "base_cohort": {
            "path": str(args.base_cohort),
            "records": len(base_rows),
            "sha256": sha256_file(args.base_cohort),
        },
        "source_pool": {
            "path": str(args.source_pool),
            "records": len(pool_rows),
            "sha256": sha256_file(args.source_pool),
        },
        "filters": {
            "correct_count_band": [args.correct_count_min, args.correct_count_max],
            "excluded_known_set": args.exclude_known_set or None,
            "datasets": sorted(set(args.dataset)) or None,
            "excluded_by_base_identity": len(
                {identity(row) for row in pool_rows} & excluded_by_identity
            ),
        },
        "availability_by_bucket": {str(k): v for k, v in availability.items()},
        "quota_by_bucket": {str(k): v for k, v in quota.items()},
        "selected_task_ids_by_bucket": {str(k): v for k, v in selected_ids.items()},
        "extension_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "base_histogram_present_in_metadata": {
            str(k): v for k, v in sorted(base_histogram.items())
        },
        "output": {
            "path": str(args.output),
            "records": len(merged),
            "sha256": sha256_file(args.output),
        },
        "db_path_prefix": args.db_path_prefix or None,
        "seed": args.seed,
    }
    atomic_write_text(
        args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_task_ids_by_bucket"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
