#!/usr/bin/env python3
"""Validate and merge disjoint Atomic v26 pass@1 evaluation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from rl.evaluation.runners.aggregation import merge_pass1_shards
except ModuleNotFoundError:
    from evaluation.runners.aggregation import merge_pass1_shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    result = merge_pass1_shards(
        examples=args.examples,
        shards=args.shard,
        output=args.output,
        adapter=args.adapter,
        expected_count=args.expected,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
