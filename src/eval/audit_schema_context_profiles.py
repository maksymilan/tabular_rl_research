#!/usr/bin/env python3
"""Render-only audit for BIRD tool-agent initial-context profiles (no model/API calls)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from executor import Harness  # noqa: E402
from schema_context_ablation import (  # noqa: E402
    INITIAL_CONTEXT_PROFILES,
    build_initial_context,
    context_contract_sha256,
)


def load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--schema-metadata-json", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--schema-value-count", type=int, default=2)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=INITIAL_CONTEXT_PROFILES,
        default=list(INITIAL_CONTEXT_PROFILES),
    )
    args = parser.parse_args()

    examples = load_jsonl(Path(args.examples_file), args.limit)
    started = time.time()
    summaries = {}
    for profile in args.profiles:
        sizes = []
        for example in examples:
            harness = Harness(example["db_path"])
            try:
                context = build_initial_context(
                    harness,
                    example,
                    profile=profile,
                    schema_metadata_json=args.schema_metadata_json,
                    value_count=args.schema_value_count,
                )
            finally:
                harness.conn.close()
            sizes.append(
                len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
            )
        summaries[profile] = {
            "count": len(sizes),
            "min_chars": min(sizes),
            "mean_chars": round(statistics.mean(sizes), 1),
            "max_chars": max(sizes),
            "context_contract_sha256": context_contract_sha256(profile),
        }
    print(
        json.dumps(
            {
                "examples_file": args.examples_file,
                "schema_metadata_json": args.schema_metadata_json,
                "elapsed_seconds": round(time.time() - started, 3),
                "profiles": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
