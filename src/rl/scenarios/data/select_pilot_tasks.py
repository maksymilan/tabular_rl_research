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
try:
    from rl.data_selection.passk import sample_correct_count
    from rl.data_selection.pilot import legal_rate, pilot_bucket, select_pilot
    from rl.shared.io import read_jsonl
except ModuleNotFoundError:
    from data_selection.passk import sample_correct_count
    from data_selection.pilot import legal_rate, pilot_bucket, select_pilot
    from shared.io import read_jsonl


def load_records(path: Path) -> list[dict]:
    """Compatibility name for historical callers; implementation is shared."""
    return read_jsonl(path)


def correct_count(record: dict) -> int:
    return sample_correct_count(record)


bucket = pilot_bucket


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rollout/pass@k all.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--include-hard", action="store_true",
                        help="include legal-but-all-wrong tasks after mixed-success tasks")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    selected = select_pilot(records, limit=args.limit, include_hard=args.include_hard)

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
