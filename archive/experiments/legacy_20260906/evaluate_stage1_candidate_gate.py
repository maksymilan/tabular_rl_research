#!/usr/bin/env python3
"""Evaluate predeclared checkpoint-selection requirements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def greedy_metrics(path: Path, expected_questions: int) -> dict[str, float | int]:
    rows = load_jsonl(path)
    if len(rows) != expected_questions:
        raise ValueError(
            f"greedy evaluation requires exact {expected_questions} rows: {path}"
        )
    samples = [row["samples"][0] for row in rows]
    if any(len(row.get("samples") or []) != 1 for row in rows):
        raise ValueError("greedy gate received non-greedy samples")
    return {
        "questions": len(rows),
        "correct": sum(bool(sample.get("correct")) for sample in samples),
        "greedy_at_1": sum(bool(sample.get("correct")) for sample in samples) / len(samples),
        "valid_rate": sum(bool(sample.get("legal")) for sample in samples) / len(samples),
    }


def prefix_metrics(path: Path) -> dict[str, float | int]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"empty fixed-prefix scores: {path}")
    margins = [float(row["margin"]) for row in rows]
    return {
        "pairs": len(rows),
        "top1_accuracy": sum(value > 0 for value in margins) / len(margins),
        "mean_margin": fmean(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--sft2-greedy", required=True, type=Path)
    parser.add_argument("--candidate-greedy", required=True, type=Path)
    parser.add_argument("--sft2-prefix", required=True, type=Path)
    parser.add_argument("--candidate-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-questions", type=int, default=300)
    args = parser.parse_args()

    sft2_greedy = greedy_metrics(args.sft2_greedy, args.expected_questions)
    candidate_greedy = greedy_metrics(args.candidate_greedy, args.expected_questions)
    sft2_prefix = prefix_metrics(args.sft2_prefix)
    candidate_prefix = prefix_metrics(args.candidate_prefix)
    if candidate_prefix["pairs"] != sft2_prefix["pairs"]:
        raise ValueError("candidate and SFT2 fixed-prefix coverage differs")
    checks = {
        "greedy_not_below_sft2_by_more_than_1pp": (
            candidate_greedy["greedy_at_1"] >= sft2_greedy["greedy_at_1"] - 0.01
        ),
        "fixed_prefix_mean_margin_above_sft2": (
            candidate_prefix["mean_margin"] > sft2_prefix["mean_margin"]
        ),
        "valid_not_below_sft2_by_more_than_1pp": (
            candidate_greedy["valid_rate"] >= sft2_greedy["valid_rate"] - 0.01
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "stage1-checkpoint-selection-requirements-v2",
        "candidate": args.candidate_name,
        "status": "passed" if all(checks.values()) else "failed",
        "eligible_for_finalist_k4": all(checks.values()),
        "checks": checks,
        "sft2": {"greedy": sft2_greedy, "fixed_prefix": sft2_prefix},
        "candidate_metrics": {
            "greedy": candidate_greedy,
            "fixed_prefix": candidate_prefix,
        },
        "deltas": {
            "greedy_at_1_pp": 100
            * (candidate_greedy["greedy_at_1"] - sft2_greedy["greedy_at_1"]),
            "valid_rate_pp": 100
            * (candidate_greedy["valid_rate"] - sft2_greedy["valid_rate"]),
            "fixed_prefix_mean_margin": candidate_prefix["mean_margin"]
            - sft2_prefix["mean_margin"],
        },
        "note": "checkpoint-selection requirements, not a statistical significance claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
