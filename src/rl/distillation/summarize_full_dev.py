#!/usr/bin/env python3
"""Full BIRD-dev greedy summary for a streaming teacher-union candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def exact_mcnemar_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = min(gains, regressions)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def sample_of(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError("full-dev greedy row must contain exactly one sample")
    return samples[0]


def metrics(rows: list[dict[str, Any]], difficulties: dict[int, str]) -> dict[str, Any]:
    if len(rows) != 1534 or len({int(row["example_index"]) for row in rows}) != 1534:
        raise ValueError("full-dev summary requires 1,534 unique examples")
    samples = [sample_of(row) for row in rows]
    correct = sum(bool(sample["correct"]) for sample in samples)
    legal = sum(bool(sample["legal"]) for sample in samples)
    by_difficulty = {}
    for level in ("simple", "moderate", "challenging"):
        indices = [
            index
            for index, row in enumerate(rows)
            if difficulties[int(row["example_index"])] == level
        ]
        by_difficulty[level] = {
            "correct": sum(bool(samples[index]["correct"]) for index in indices),
            "total": len(indices),
        }
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "valid": legal,
        "valid_rate": legal / len(rows),
        "avg_steps": fmean(float(sample["steps"]) for sample in samples),
        "difficulty": by_difficulty,
    }


def paired(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_by_id = {int(row["example_index"]): sample_of(row) for row in candidate}
    baseline_by_id = {int(row["example_index"]): sample_of(row) for row in baseline}
    if set(candidate_by_id) != set(baseline_by_id) or len(candidate_by_id) != 1534:
        raise ValueError("paired full-dev example ids differ")
    gains = []
    regressions = []
    for example_index in sorted(candidate_by_id):
        candidate_correct = bool(candidate_by_id[example_index]["correct"])
        baseline_correct = bool(baseline_by_id[example_index]["correct"])
        if candidate_correct and not baseline_correct:
            gains.append(example_index)
        elif baseline_correct and not candidate_correct:
            regressions.append(example_index)
    return {
        "gains": len(gains),
        "regressions": len(regressions),
        "net": len(gains) - len(regressions),
        "exact_mcnemar_p": exact_mcnemar_p(len(gains), len(regressions)),
        "gain_example_indices": gains,
        "regression_example_indices": regressions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", action="append", default=[], help="LABEL:all.jsonl")
    parser.add_argument("--eval-inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite summary: {args.output}")
    eval_rows = load_jsonl(args.eval_inputs)
    difficulties = {
        int(row["example_index"]): str((row.get("metadata") or {})["difficulty"])
        for row in eval_rows
    }
    candidate = load_jsonl(args.candidate)
    output = {
        "schema_version": "streaming-teacher-union-full-dev-summary-v1",
        "candidate": metrics(candidate, difficulties),
        "baselines": {},
        "paired": {},
        "interpretation": "single-run exploratory result; no K4 finalist evaluation",
    }
    for item in args.baseline:
        label, separator, path = item.partition(":")
        if not separator or not label or not path:
            raise ValueError("baseline must use LABEL:all.jsonl")
        rows = load_jsonl(Path(path))
        output["baselines"][label] = metrics(rows, difficulties)
        output["paired"][label] = paired(candidate, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
