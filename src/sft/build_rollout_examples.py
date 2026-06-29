#!/usr/bin/env python3
"""Build explicit train examples for rollout/evaluation scripts.

Recovery data must be generated from training-set rollouts, not Spider dev. This helper converts a
trajectory-id subset such as data/trajectories/subset_v10_clean_1600.ids.json into a JSON examples
file backed by data/spider_data/train_spider.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "spider_data" / "train_spider.json"


def load_ids(path: Path, key: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        values = payload.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{path}: missing list key {key!r}")
        return [str(item) for item in values]
    raise ValueError(f"{path}: expected JSON list or object")


def train_index_from_id(trajectory_id: str) -> int:
    prefix = "spider_train_"
    if not trajectory_id.startswith(prefix):
        raise ValueError(f"not a Spider train trajectory id: {trajectory_id}")
    return int(trajectory_id[len(prefix):])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="ids JSON from select_subset.py")
    parser.add_argument("--key", default="subset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ids = load_ids(Path(args.ids), args.key)
    if args.limit:
        ids = ids[:args.limit]
    train = json.loads(TRAIN.read_text(encoding="utf-8"))

    examples = []
    for trajectory_id in ids:
        index = train_index_from_id(trajectory_id)
        if index < 0 or index >= len(train):
            raise ValueError(f"{trajectory_id}: train index out of range")
        ex = dict(train[index])
        ex["example_index"] = index
        ex["trajectory_id"] = trajectory_id
        ex["dataset_split"] = "train"
        examples.append(ex)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source": str(TRAIN.relative_to(ROOT)),
        "ids": str(Path(args.ids)),
        "key": args.key,
        "count": len(examples),
        "examples": examples,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "count": len(examples)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
