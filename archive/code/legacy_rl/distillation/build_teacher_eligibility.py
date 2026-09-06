#!/usr/bin/env python3
"""CLI for freezing K4 teacher routing before a streaming student run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))

from src.rl.distillation.teacher_eligibility import (
    build_eligibility_rows,
    interleaved_training_order,
    load_jsonl,
    sha256_file,
)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft2-rollouts", required=True, type=Path)
    parser.add_argument("--exp15-rollouts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument("--minimum-dense-correct", type=int, default=2)
    parser.add_argument("--protocol-version", default="version26")
    parser.add_argument("--order-seed", type=int, default=101)
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite frozen teacher eligibility artifacts")
    rows = build_eligibility_rows(
        load_jsonl(args.sft2_rollouts),
        load_jsonl(args.exp15_rollouts),
        expected_k=args.expected_k,
        minimum_dense_correct=args.minimum_dense_correct,
        expected_protocol_version=args.protocol_version,
    )
    order = interleaved_training_order(rows, seed=args.order_seed)
    order_rank = {example_index: rank for rank, example_index in enumerate(order)}
    for row in rows:
        row["training_order"] = order_rank.get(int(row["example_index"]))
    _write_jsonl_atomic(args.output, rows)

    manifest = {
        "schema_version": "teacher-eligibility-manifest-v1",
        "status": "frozen",
        "dataset_split": "train",
        "sft2_rollouts": str(args.sft2_rollouts.resolve()),
        "sft2_rollouts_sha256": sha256_file(args.sft2_rollouts),
        "exp15_rollouts": str(args.exp15_rollouts.resolve()),
        "exp15_rollouts_sha256": sha256_file(args.exp15_rollouts),
        "eligibility": str(args.output.resolve()),
        "eligibility_sha256": sha256_file(args.output),
        "expected_k": args.expected_k,
        "minimum_dense_correct": args.minimum_dense_correct,
        "protocol_version": args.protocol_version,
        "order_seed": args.order_seed,
        "training_order": order,
        "training_order_sha256": _sha256_json(order),
        "tasks": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(f"{args.manifest.name}.next.{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.manifest)


if __name__ == "__main__":
    main()
