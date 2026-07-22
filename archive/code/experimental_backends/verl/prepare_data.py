#!/usr/bin/env python3
"""Serialize framework-neutral result-only records into Verl Parquet.

The examples contain only the initial model prompt plus hidden environment metadata.  Gold SQL is
never rendered to the model; the custom agent loop uses it only to score the terminal denotation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from task_data import load_result_only_task_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--selection", type=Path,
                        help="JSONL from select_pilot_tasks.py; required for the RL train set")
    parser.add_argument("--examples-json", type=Path,
                        help="verified training candidate JSON from build_result_only_candidates.py")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        records = load_result_only_task_records(
            ROOT,
            split=args.split,
            selection=args.selection,
            examples_json=args.examples_json,
            limit=args.limit,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rows = [
        {
            "data_source": record["data_source"],
            "agent_name": "table_tool_agent",
            "prompt": record["prompt"],
            "reward_model": {"style": "rule", "ground_truth": "hidden_execution_result"},
            "extra_info": record["environment"],
        }
        for record in records
    ]

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - executed in the verl environment
        raise SystemExit("datasets is required; run this script inside the verl environment") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    print(json.dumps({
        "split": args.split,
        "source": f"spider_{args.split}",
        "selection": str(args.selection) if args.selection else None,
        "examples_json": str(args.examples_json) if args.examples_json else None,
        "records": len(rows),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
