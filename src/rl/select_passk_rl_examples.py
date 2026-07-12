#!/usr/bin/env python3
"""Build a trainer-ready RL task set from pass@k rollout artifacts.

Result-only group-relative RL only gets a useful update when samples in a group have different
terminal rewards.  Pass@k artifacts are a good source for those tasks: questions that fail pass@1
but succeed by a later sample are neither trivial nor hopeless under the current policy.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
    return records


def pass_keys(record: dict[str, Any]) -> list[int]:
    keys = []
    for key in (record.get("pass_at") or {}).keys():
        try:
            keys.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(keys)


def target_pass_k(records: list[dict[str, Any]], requested: int | None) -> int:
    if requested is not None:
        return requested
    keys = sorted({key for record in records for key in pass_keys(record)})
    if not keys:
        raise ValueError("pass@k artifact does not contain pass_at fields")
    return keys[-1]


def sample_attempt_count(record: dict[str, Any]) -> int:
    samples = record.get("samples") or []
    if samples:
        return len(samples)
    n_samples = record.get("n_samples")
    return int(n_samples) if n_samples else 1


def classify(record: dict[str, Any], *, target_k: int) -> str:
    pass_at = record.get("pass_at") or {}
    pass1 = bool(pass_at.get("1"))
    pass_target = bool(pass_at.get(str(target_k)))
    if pass_target and not pass1:
        return "pass1_fail_passk_success"
    if pass1:
        return "pass1_success"
    return "all_attempted_failed"


def as_training_example(record: dict[str, Any], *, bucket: str, source_path: Path, target_k: int) -> dict[str, Any]:
    query = record.get("query") or record.get("gold_sql")
    if not query:
        raise ValueError(f"record {record.get('example_index')} is missing query/gold_sql")
    return {
        "example_index": int(record["example_index"]),
        "dataset_split": "train",
        "db_id": record["db_id"],
        "question": record["question"],
        "query": query,
        "selection": {
            "bucket": bucket,
            "source": str(source_path),
            "target_pass_k": target_k,
            "pass_at": record.get("pass_at") or {},
            "sample_correct_count": int(record.get("sample_correct_count", 0)),
            "sample_legal_count": int(record.get("sample_legal_count", 0)),
            "sample_attempt_count": sample_attempt_count(record),
            "failure_type": record.get("failure_type"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="pass@k all.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="trainer-ready JSON accepted by group_reinforce.py --examples-json")
    parser.add_argument("--target-pass-k", type=int, default=None,
                        help="success threshold to compare against pass@1; default = largest pass_at key")
    parser.add_argument("--mode", choices=("pass1_fail_passk_success", "all_attempted_failed", "pass1_success", "all"),
                        default="pass1_fail_passk_success")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    target_k = target_pass_k(records, args.target_pass_k)
    buckets = [(record, classify(record, target_k=target_k)) for record in records]
    if args.mode == "all":
        selected = buckets
    else:
        selected = [(record, bucket) for record, bucket in buckets if bucket == args.mode]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(selected)
    if args.limit:
        selected = selected[:args.limit]

    examples = [
        as_training_example(record, bucket=bucket, source_path=args.input, target_k=target_k)
        for record, bucket in selected
    ]
    summary = {
        "source": str(args.input),
        "mode": args.mode,
        "target_pass_k": target_k,
        "total_records": len(records),
        "count": len(examples),
        "bucket_counts": dict(Counter(bucket for _, bucket in buckets)),
        "selected_bucket_counts": dict(Counter(example["selection"]["bucket"] for example in examples)),
        "note": (
            "For pass@k runs with --stop-on-success, pass1_fail_passk_success means the first sample "
            "failed and a later attempted sample succeeded; sample_correct_count is a lower-bound, "
            "not an unbiased estimate of success probability."
        ),
    }
    payload = {
        "dataset_split": "train",
        "selection": summary,
        "count": len(examples),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary | {"output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
