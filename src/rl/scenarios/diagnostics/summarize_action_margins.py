#!/usr/bin/env python3
"""Summarize and compare fixed-prefix action margins across checkpoints."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from rl.diagnostics.records import load_jsonl as _load_jsonl


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["margin"]) for row in rows]
    return {
        "pairs": len(rows),
        "fixed_prefix_top1_accuracy": (
            sum(value > 0 for value in margins) / len(margins) if margins else None
        ),
        "ties": sum(value == 0 for value in margins),
        "mean_margin": fmean(margins) if margins else None,
        "median_margin": median(margins) if margins else None,
        "positive_action_logprob": (
            fmean(float(row["positive_action_logprob"]) for row in rows) if rows else None
        ),
        "negative_action_logprob": (
            fmean(float(row["negative_action_logprob"]) for row in rows) if rows else None
        ),
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the mandated parquet artifact") from exc
    parquet_rows = [
        {
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(parquet_rows), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reference", default="sft2")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    combined = []
    for value in args.scores:
        name, separator, path_text = value.partition("=")
        if not separator:
            raise SystemExit(f"invalid --scores value: {value}")
        rows = load_jsonl(Path(path_text))
        if any(row.get("checkpoint") != name for row in rows):
            raise SystemExit(f"checkpoint label mismatch for {name}")
        by_checkpoint[name] = rows
        combined.extend(rows)
    if args.reference not in by_checkpoint:
        raise SystemExit(f"missing reference checkpoint: {args.reference}")

    pair_sets = {
        name: {row["pair_sha256"] for row in rows} for name, rows in by_checkpoint.items()
    }
    common_pairs = set.intersection(*pair_sets.values()) if pair_sets else set()
    if any(pairs != common_pairs for pairs in pair_sets.values()):
        raise SystemExit("checkpoint score files do not contain exactly the same pair ids")

    category_payload: dict[str, Any] = {}
    checkpoint_payload: dict[str, Any] = {}
    reference_by_pair = {
        row["pair_sha256"]: float(row["margin"]) for row in by_checkpoint[args.reference]
    }
    for name, rows in by_checkpoint.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["category"]].append(row)
        category_payload[name] = {
            category: summary(group) for category, group in sorted(grouped.items())
        }
        overall = summary(rows)
        margin_deltas = [
            float(row["margin"]) - reference_by_pair[row["pair_sha256"]] for row in rows
        ]
        overall["mean_margin_delta_vs_reference"] = (
            fmean(margin_deltas) if margin_deltas else None
        )
        overall["pairs_improved_vs_reference"] = sum(value > 0 for value in margin_deltas)
        overall["pairs_regressed_vs_reference"] = sum(value < 0 for value in margin_deltas)
        checkpoint_payload[name] = overall

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "margin_by_category.json").write_text(
        json.dumps(category_payload, ensure_ascii=False, indent=2) + "\n"
    )
    comparison = {
        "schema_version": "fixed-prefix-checkpoint-comparison-v1",
        "reference": args.reference,
        "pairs": len(common_pairs),
        "checkpoints": checkpoint_payload,
    }
    (args.output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )
    write_parquet(args.output_dir / "fixed_prefix_scores.parquet", combined)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
