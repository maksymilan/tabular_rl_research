#!/usr/bin/env python3
"""Combine disjoint frozen teacher K-rollout artifacts with fresh sequencing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))

from src.rl.distillation.teacher_eligibility import (
    collect_teacher_rollouts,
    load_jsonl,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument("--protocol-version", default="version26")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite combined teacher rollout artifacts")

    rows = [row for path in args.input for row in load_jsonl(path)]
    groups = collect_teacher_rollouts(
        rows,
        expected_k=args.expected_k,
        expected_protocol_version=args.protocol_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.next.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for sequence, row in enumerate(rows):
            retained = dict(row)
            retained["sequence"] = sequence
            target.write(json.dumps(retained, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    manifest = {
        "schema_version": "combined-teacher-rollouts-v1",
        "status": "frozen",
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.input
        ],
        "protocol_version": args.protocol_version,
        "expected_k": args.expected_k,
        "tasks": len(groups),
        "trajectories": len(rows),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
