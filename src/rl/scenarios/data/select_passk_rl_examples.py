#!/usr/bin/env python3
"""Build a trainer-ready RL task set from pass@k rollout artifacts.

Result-only group-relative RL only gets a useful update when samples in a group have different
terminal rewards.  Pass@k artifacts are a good source for those tasks: questions that fail pass@1
but succeed by a later sample are neither trivial nor hopeless under the current policy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
try:
    from rl.shared.io import read_jsonl
    from rl.data_selection.passk import (
        classify,
        has_infrastructure_failure,
        has_mixed_attempt_outcomes,
        sample_attempt_count,
        sample_correct_count,
        sample_legal_count,
        select_records,
        target_pass_k,
    )
except ModuleNotFoundError:
    from shared.io import read_jsonl
    from data_selection.passk import (
        classify,
        has_infrastructure_failure,
        has_mixed_attempt_outcomes,
        sample_attempt_count,
        sample_correct_count,
        sample_legal_count,
        select_records,
        target_pass_k,
    )

# Keep the historical helper name for old audit imports while using the shared
# implementation for all new selectors.
load_jsonl = read_jsonl


def as_training_example(record: dict[str, Any], *, bucket: str, source_path: Path, target_k: int,
                        task: dict[str, Any] | None = None) -> dict[str, Any]:
    query = record.get("query") or record.get("gold_sql")
    if not query:
        raise ValueError(f"record {record.get('example_index')} is missing query/gold_sql")
    example = {
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
            "sample_correct_count": sample_correct_count(record),
            "sample_legal_count": int(record.get("sample_legal_count", 0)),
            "sample_attempt_count": sample_attempt_count(record),
            "failure_type": record.get("failure_type"),
        },
    }
    if task is not None:
        if int(task["example_index"]) != int(record["example_index"]):
            raise ValueError("task/rollout example_index mismatch")
        example.update({
            "db_path": task.get("db_path"),
            "gold_sql": task.get("gold_sql") or task.get("query"),
            "external_knowledge": task.get("external_knowledge"),
            "metadata": task.get("metadata"),
        })
    return example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="pass@k all.jsonl")
    parser.add_argument("--tasks", type=Path,
                        help="optional DatasetTask JSON/JSONL supplying db_path and external knowledge")
    parser.add_argument("--output", type=Path, required=True,
                        help="trainer-ready JSON accepted by group_reinforce.py --examples-json")
    parser.add_argument("--target-pass-k", type=int, default=None,
                        help="success threshold to compare against pass@1; default = largest pass_at key")
    parser.add_argument("--mode", choices=(
        "pass1_fail_passk_success",
        "mixed_attempt_outcomes",
        "all_attempted_failed",
        "pass1_success",
        "all",
    ),
                        default="pass1_fail_passk_success")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    task_map: dict[int, dict[str, Any]] = {}
    if args.tasks:
        task_rows = load_jsonl(args.tasks) if args.tasks.suffix == ".jsonl" else json.loads(
            args.tasks.read_text(encoding="utf-8")
        )
        if isinstance(task_rows, dict):
            task_rows = task_rows.get("examples")
        if not isinstance(task_rows, list):
            raise ValueError("--tasks must contain a JSON list/JSONL or an object with examples")
        task_map = {int(task["example_index"]): task for task in task_rows}
        missing = sorted(int(record["example_index"]) for record in records
                         if int(record["example_index"]) not in task_map)
        if missing:
            raise ValueError(f"--tasks is missing {len(missing)} rollout ids: {missing[:20]}")
    target_k = target_pass_k(records, args.target_pass_k)
    buckets = [(record, classify(record, target_k=target_k)) for record in records]
    selected, _ = select_records(
        records, mode=args.mode, target_k=target_k, limit=args.limit,
        seed=args.seed, shuffle=not args.no_shuffle,
    )

    examples = [
        as_training_example(
            record,
            bucket=bucket,
            source_path=args.input,
            target_k=target_k,
            task=task_map.get(int(record["example_index"])),
        )
        for record, bucket in selected
    ]
    summary = {
        "source": str(args.input),
        "mode": args.mode,
        "target_pass_k": target_k,
        "total_records": len(records),
        "count": len(examples),
        "bucket_counts": dict(Counter(bucket for _, bucket in buckets)),
        "mixed_attempt_outcome_count": sum(
            has_mixed_attempt_outcomes(record) for record in records
        ),
        "infrastructure_contaminated_count": sum(
            has_infrastructure_failure(record) for record in records
        ),
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
