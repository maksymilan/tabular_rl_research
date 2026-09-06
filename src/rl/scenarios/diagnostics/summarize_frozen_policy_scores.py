#!/usr/bin/env python3
"""Summarize exact-prefix policy movement and reward-direction agreement."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("correlation vectors do not align")
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left2 = sum((value - left_mean) ** 2 for value in left)
    right2 = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left2 * right2)
    return numerator / denominator if denominator else None


def summarize_checkpoint(
    rows: list[dict[str, Any]], checkpoint: str, baseline: str
) -> dict[str, Any]:
    deltas = [
        float(row["scores"][checkpoint]["response_mean_logprob"])
        - float(row["scores"][baseline]["response_mean_logprob"])
        for row in rows
    ]
    advantages = [
        float(row.get("checkpoint_advantages", {}).get(checkpoint, row["advantage"]))
        for row in rows
    ]
    alignments = [
        advantage * delta
        for advantage, delta in zip(advantages, deltas)
    ]
    absolute = [abs(value) for value in deltas]
    positive_alignment = sum(max(0.0, value) for value in alignments)
    negative_alignment = sum(min(0.0, value) for value in alignments)
    gross_alignment = positive_alignment - negative_alignment

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["reward_category"]].append(index)
        groups["positive_reward" if advantages[index] > 0 else "negative_reward"].append(index)
    group_results = {}
    for name, indices in sorted(groups.items()):
        group_deltas = [deltas[index] for index in indices]
        group_alignments = [alignments[index] for index in indices]
        expected_positive = name == "positive_reward"
        expected_negative = name == "negative_reward"
        group_results[name] = {
            "transitions": len(indices),
            "mean_logprob_delta": mean(group_deltas),
            "mean_abs_logprob_delta": mean([abs(value) for value in group_deltas]),
            "mean_reward_alignment": mean(group_alignments),
            "reward_aligned_fraction": mean(
                [float(value > 0.0) for value in group_alignments]
            ),
            "expected_direction_fraction": (
                mean([float(value > 0.0) for value in group_deltas])
                if expected_positive
                else mean([float(value < 0.0) for value in group_deltas])
                if expected_negative
                else None
            ),
        }
    tool_deltas = []
    for row in rows:
        current = row["scores"][checkpoint]["tool_mean_logprob"]
        reference = row["scores"][baseline]["tool_mean_logprob"]
        if current is not None and reference is not None:
            tool_deltas.append(float(current) - float(reference))
    return {
        "transitions": len(rows),
        "response_logprob_delta": {
            "mean": mean(deltas),
            "mean_abs": mean(absolute),
            "median_abs": percentile(absolute, 0.5),
            "p90_abs": percentile(absolute, 0.9),
            "p99_abs": percentile(absolute, 0.99),
            "max_abs": max(absolute),
            "fraction_abs_gt_0_001": mean([float(value > 0.001) for value in absolute]),
            "fraction_abs_gt_0_01": mean([float(value > 0.01) for value in absolute]),
            "fraction_abs_gt_0_05": mean([float(value > 0.05) for value in absolute]),
        },
        "tool_logprob_delta": {
            "transitions": len(tool_deltas),
            "mean": mean(tool_deltas),
            "mean_abs": mean([abs(value) for value in tool_deltas]),
        },
        "reward_direction": {
            "advantage_delta_pearson": pearson(advantages, deltas),
            "mean_alignment": mean(alignments),
            "aligned_fraction": mean([float(value > 0.0) for value in alignments]),
            "positive_alignment_mass": positive_alignment,
            "negative_alignment_mass": negative_alignment,
            "net_alignment_mass": sum(alignments),
            "net_over_gross_alignment": (
                sum(alignments) / gross_alignment if gross_alignment else None
            ),
        },
        "groups": group_results,
    }


def summarize(
    rows: list[dict[str, Any]], checkpoints: list[str], baseline: str
) -> dict[str, Any]:
    keys = [(str(row["trajectory_id"]), int(row["turn_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("score rows contain duplicate transition keys")
    expected = set(checkpoints) | {baseline}
    for row in rows:
        if set(row["scores"]) != expected:
            raise ValueError("score row checkpoint labels do not match")
    return {
        "schema_version": "frozen-policy-movement-analysis-v1",
        "baseline": baseline,
        "rows": len(rows),
        "checkpoints": {
            checkpoint: summarize_checkpoint(rows, checkpoint, baseline)
            for checkpoint in checkpoints
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(load_jsonl(args.scores), args.checkpoint, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": result["rows"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
