#!/usr/bin/env python3
"""Select a small RL pilot task set from rollout/pass@k artifacts.

The first RL batch should avoid pure formatting/API failures and emphasize tasks where the
model can already execute legal tool trajectories but still needs better feedback use.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def legal_rate(record: dict) -> float:
    samples = record.get("samples") or []
    if not samples:
        return 1.0 if record.get("legal") else 0.0
    return sum(bool(sample.get("legal")) for sample in samples) / len(samples)


def correct_count(record: dict) -> int:
    samples = record.get("samples") or []
    if not samples:
        return int(bool(record.get("correct")))
    return int(record.get("sample_correct_count", sum(bool(sample.get("correct")) for sample in samples)))


def bucket(record: dict) -> str:
    samples = record.get("samples") or []
    n_samples = len(samples) if samples else 1
    cc = correct_count(record)
    lr = legal_rate(record)
    if lr < 0.8:
        return "format_or_execution_unstable"
    if cc == 0:
        return "legal_but_all_wrong"
    if 0 < cc < n_samples:
        return "mixed_success"
    return "already_easy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rollout/pass@k all.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--include-hard", action="store_true",
                        help="include legal-but-all-wrong tasks after mixed-success tasks")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    enriched = [{**record, "rl_bucket": bucket(record), "rl_legal_rate": legal_rate(record)}
                for record in records]
    order = {"mixed_success": 0, "legal_but_all_wrong": 1, "already_easy": 2,
             "format_or_execution_unstable": 3}
    allowed = {"mixed_success"}
    if args.include_hard:
        allowed.add("legal_but_all_wrong")
    selected = [
        record for record in sorted(enriched, key=lambda item: (
            order[item["rl_bucket"]],
            item.get("example_index", 10**9),
        ))
        if record["rl_bucket"] in allowed
    ][:args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for record in selected:
            sink.write(json.dumps({
                "example_index": record.get("example_index"),
                "db_id": record.get("db_id"),
                "question": record.get("question"),
                "gold_sql": record.get("gold_sql"),
                "bucket": record["rl_bucket"],
                "legal_rate": record["rl_legal_rate"],
                "sample_correct_count": correct_count(record),
                "n_samples": record.get("n_samples", len(record.get("samples") or []) or 1),
            }, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": str(output),
        "selected": len(selected),
        "limit": args.limit,
        "bucket_counts": dict(Counter(record["rl_bucket"] for record in enriched)),
        "selected_bucket_counts": dict(Counter(record["rl_bucket"] for record in selected)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
