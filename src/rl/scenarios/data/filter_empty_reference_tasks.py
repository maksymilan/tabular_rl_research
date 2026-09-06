#!/usr/bin/env python3
"""Create an RL task artifact with empty/zero reference-result tasks removed."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from rl.runtime.reference_result_filter import (
    audit_task_environment,
    example_environment,
    read_examples_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--example-index",
        type=int,
        action="append",
        help="optionally retain only these example indices before reference-result filtering",
    )
    args = parser.parse_args()

    if args.offset < 0 or args.limit < 0:
        raise SystemExit("--offset and --limit must be non-negative")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")

    payload, examples = read_examples_payload(args.input)
    selected = examples[args.offset:]
    if args.limit:
        selected = selected[:args.limit]
    if args.example_index:
        requested = set(args.example_index)
        selected = [
            example for example in selected
            if int(example["example_index"]) in requested
        ]
        found = {int(example["example_index"]) for example in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested example indices are missing: {missing}")
    selected = selected[args.shard_index::args.num_shards]

    retained: list[dict[str, Any]] = []
    audits = []
    for example in selected:
        audit = audit_task_environment(example_environment(example))
        audits.append(audit)
        if not audit.excluded:
            retained.append(example)

    kind_counts = collections.Counter(audit.kind for audit in audits)
    summary = {
        "schema_version": "rl-empty-reference-task-filter-v1",
        "source": str(args.input),
        "offset": args.offset,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "input_count": len(selected),
        "retained_count": len(retained),
        "excluded_count": len(selected) - len(retained),
        "kind_counts": dict(sorted(kind_counts.items())),
        "policy": {
            "empty_rows": "exclude",
            "null_scalar": "exclude",
            "zero_scalar": "exclude",
            "nonempty": "retain",
        },
    }
    output_payload: dict[str, Any]
    if payload is None:
        output_payload = {
            "dataset_split": "train",
            "count": len(retained),
            "selection": {"reference_result_filter": summary},
            "examples": retained,
        }
    else:
        output_payload = {
            **{key: value for key, value in payload.items() if key not in {"count", "examples"}},
            "count": len(retained),
            "examples": retained,
        }
        selection = dict(output_payload.get("selection") or {})
        selection["reference_result_filter"] = summary
        output_payload["selection"] = selection

    audit_payload = {
        **summary,
        "tasks": [audit.to_dict() for audit in audits],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.audit_output.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
