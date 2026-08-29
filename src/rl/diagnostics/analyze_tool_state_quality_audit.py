#!/usr/bin/env python3
"""Second-stage statistics for a completed tool-state quality audit JSONL."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


PRODUCER_TOOLS = (
    "condition_filter", "project", "join_tables", "group_aggregate",
    "scalar_compute", "extreme_value_select", "set_op",
)
CATEGORIES = ("source", "join", "predicate", "grain", "value", "set", "rank", "output")


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        rank_sum += ((index + 1 + stop) / 2) * sum(label for _, label in ordered[index:stop])
        index = stop
    return (rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]], list[dict[str, Any]]], float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        pos = [positive[rng.randrange(len(positive))] for _ in positive]
        neg = [negative[rng.randrange(len(negative))] for _ in negative]
        values.append(statistic(pos, neg))
    return {"p2_5": percentile(values, 0.025), "p97_5": percentile(values, 0.975)}


def analyze(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    scored = [row for row in rows if row.get("score") is not None]
    correct = [row for row in scored if row["correct"]]
    incorrect = [row for row in scored if not row["correct"]]

    def values(group: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in group if row.get(key) is not None]

    correct_scores, incorrect_scores = values(correct, "score"), values(incorrect, "score")
    gap = statistics.fmean(correct_scores) - statistics.fmean(incorrect_scores)
    pooled = math.sqrt(
        ((len(correct_scores) - 1) * statistics.variance(correct_scores)
         + (len(incorrect_scores) - 1) * statistics.variance(incorrect_scores))
        / (len(correct_scores) + len(incorrect_scores) - 2)
    )

    formulations = {
        "q_rank": lambda row: float(row["score"]),
        "c_terminal": lambda row: float(row["c_terminal"]),
        "c_max": lambda row: float(row["c_max"]),
        "terminal_else_max": lambda row: (
            float(row["c_terminal"])
            if row.get("terminal_evidence") is not None
            else float(row["c_max"])
        ),
    }
    ablations = {}
    for name, function in formulations.items():
        ablations[name] = {
            "correct_mean": statistics.fmean(function(row) for row in correct),
            "incorrect_mean": statistics.fmean(function(row) for row in incorrect),
            "roc_auc": auc([function(row) for row in correct], [function(row) for row in incorrect]),
        }

    category_stats = {}
    for category in CATEGORIES:
        ok = [
            float(row["terminal_overlap"]["per_category"][category]["score"])
            for row in correct if category in (row.get("terminal_overlap") or {}).get("per_category", {})
        ]
        bad = [
            float(row["terminal_overlap"]["per_category"][category]["score"])
            for row in incorrect if category in (row.get("terminal_overlap") or {}).get("per_category", {})
        ]
        category_stats[category] = {
            "correct_count": len(ok),
            "correct_mean": statistics.fmean(ok) if ok else None,
            "incorrect_count": len(bad),
            "incorrect_mean": statistics.fmean(bad) if bad else None,
        }

    tool_stats = {}
    for tool in PRODUCER_TOOLS:
        ok = [float(row["score"]) for row in correct if tool in row["tools"]]
        bad = [float(row["score"]) for row in incorrect if tool in row["tools"]]
        tool_stats[tool] = {
            "correct_count": len(ok),
            "correct_mean": statistics.fmean(ok) if ok else None,
            "incorrect_count": len(bad),
            "incorrect_mean": statistics.fmean(bad) if bad else None,
        }
    correct_tool_means = [item["correct_mean"] for item in tool_stats.values() if item["correct_mean"] is not None]

    return {
        "schema_version": "tool-state-quality-validation-analysis-v1",
        "eligible": len(scored),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "separation": {
            "mean_gap": gap,
            "cohens_d_pooled": gap / pooled if pooled else None,
            "roc_auc": auc(correct_scores, incorrect_scores),
            "mean_gap_bootstrap_95": bootstrap(
                correct, incorrect,
                lambda ok, bad: statistics.fmean(float(row["score"]) for row in ok)
                - statistics.fmean(float(row["score"]) for row in bad),
            ),
            "roc_auc_bootstrap_95": bootstrap(
                correct, incorrect,
                lambda ok, bad: float(auc(
                    [float(row["score"]) for row in ok],
                    [float(row["score"]) for row in bad],
                )),
            ),
            "correct_at_or_below_incorrect_median": sum(
                value <= statistics.median(incorrect_scores) for value in correct_scores
            ),
            "incorrect_at_or_above_correct_median": sum(
                value >= statistics.median(correct_scores) for value in incorrect_scores
            ),
            "correct_score_one": sum(math.isclose(value, 1.0) for value in correct_scores),
            "incorrect_score_one": sum(math.isclose(value, 1.0) for value in incorrect_scores),
        },
        "component_ablations": ablations,
        "terminal_availability": {
            "correct_without_terminal": sum(row.get("terminal_evidence") is None for row in correct),
            "incorrect_without_terminal": sum(row.get("terminal_evidence") is None for row in incorrect),
        },
        "category_overlap": category_stats,
        "tool_presence": tool_stats,
        "correct_tool_mean_range": max(correct_tool_means) - min(correct_tool_means),
        "warnings": [
            "tool-presence means are observational and confounded by task semantics",
            "the pool has one trajectory per task, so no empirical within-task ranking test is possible",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
