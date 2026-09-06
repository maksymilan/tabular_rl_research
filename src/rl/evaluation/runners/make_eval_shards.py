#!/usr/bin/env python3
"""Create deterministic, disjoint JSONL shards for data-parallel eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both ``python -m`` and the existing direct-path launcher.
    from .eval_plan import partition_rows, read_jsonl
except ImportError:  # pragma: no cover - exercised by direct launcher
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from rl.evaluation.runners.eval_plan import partition_rows, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--first", type=Path, help="legacy first JSONL shard")
    parser.add_argument("--second", type=Path, help="legacy second JSONL shard")
    parser.add_argument("--output-dir", type=Path, help="directory for shard_XX.jsonl files")
    parser.add_argument("--shard-count", type=int, help="number of round-robin shards")
    parser.add_argument("--expected", type=int)
    parser.add_argument("--partition", choices=("round_robin", "contiguous"), default="round_robin")
    args = parser.parse_args()
    rows = read_jsonl(args.examples)
    indices = [int(row.get("example_index", i)) for i, row in enumerate(rows)]
    expected_count = args.expected if args.expected is not None else len(rows)
    expected = list(range(expected_count))
    if len(indices) != expected_count or sorted(indices) != expected:
        raise ValueError(
            f"evaluation input must contain exactly indices 0..{expected_count - 1}"
        )
    shard_count = args.shard_count or (2 if args.first and args.second else None)
    if shard_count is None or shard_count <= 0:
        raise ValueError("provide --shard-count or both legacy --first/--second")
    if args.first or args.second:
        if not (args.first and args.second) or shard_count != 2:
            raise ValueError("legacy --first/--second requires exactly two shards")
        paths = [args.first, args.second]
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for generic sharding")
        paths = [args.output_dir / f"shard_{index:02d}.jsonl" for index in range(shard_count)]
    indexed_rows = {int(row.get("example_index", i)): row for i, row in enumerate(rows)}
    row_shards = partition_rows([indexed_rows[index] for index in indices], shard_count, policy=args.partition)
    for path, shard_rows in zip(paths, row_shards):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows),
            encoding="utf-8",
        )
    print(json.dumps({"records": len(indices), "shard_sizes": [len(x) for x in row_shards]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
