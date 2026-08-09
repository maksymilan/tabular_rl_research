#!/usr/bin/env python3
"""Measure paired closed-loop policy behavior shift on a frozen cohort."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["indices"] if isinstance(payload, dict) else payload
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("indices contain duplicates")
    return indices


def only_sample(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError(f"expected one sample for {row.get('example_index')}")
    return samples[0]


def actions(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sequence = []
    for turn in only_sample(row).get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if tool is None or arguments is None:
            continue
        sequence.append(
            (
                str(tool),
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    return tuple(sequence)


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_value in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_value in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def common_prefix_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values, default=0.0),
    }


def group_shift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "sequence_changed": sum(row["exact_action_sequence_changed"] for row in rows),
        "first_action_changed": sum(row["first_exact_action_changed"] for row in rows),
        "mean_exact_edit_distance": statistics.fmean(
            float(row["exact_action_edit_distance"]) for row in rows
        ),
        "mean_normalized_exact_edit_distance": statistics.fmean(
            float(row["normalized_exact_action_edit_distance"]) for row in rows
        ),
        "mean_step_delta": statistics.fmean(int(row["step_delta"]) for row in rows),
    }


def js_divergence_bits(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    result = 0.0
    for key in keys:
        p = left[key] / left_total if left_total else 0.0
        q = right[key] / right_total if right_total else 0.0
        midpoint = (p + q) / 2.0
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def selected(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["example_index"]): row
        for row in load_jsonl(path)
        if int(row["example_index"]) in wanted
    }
    if set(rows) != wanted:
        raise ValueError(f"{path} does not contain the exact selected cohort")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indices = load_indices(args.indices)
    wanted = set(indices)
    baseline = selected(args.baseline, wanted)
    candidate = selected(args.candidate, wanted)

    per_example = []
    baseline_tools: Counter[str] = Counter()
    candidate_tools: Counter[str] = Counter()
    for index in indices:
        base_sample = only_sample(baseline[index])
        candidate_sample = only_sample(candidate[index])
        base_actions = actions(baseline[index])
        candidate_actions = actions(candidate[index])
        base_tool_sequence = tuple(tool for tool, _ in base_actions)
        candidate_tool_sequence = tuple(tool for tool, _ in candidate_actions)
        baseline_tools.update(base_tool_sequence)
        candidate_tools.update(candidate_tool_sequence)
        exact_edit = edit_distance(base_actions, candidate_actions)
        tool_edit = edit_distance(base_tool_sequence, candidate_tool_sequence)
        denominator = max(len(base_actions), len(candidate_actions), 1)
        prefix = common_prefix_length(base_actions, candidate_actions)
        per_example.append(
            {
                "example_index": index,
                "baseline_correct": bool(base_sample["correct"]),
                "candidate_correct": bool(candidate_sample["correct"]),
                "baseline_legal": bool(base_sample["legal"]),
                "candidate_legal": bool(candidate_sample["legal"]),
                "baseline_steps": int(base_sample["steps"]),
                "candidate_steps": int(candidate_sample["steps"]),
                "step_delta": int(candidate_sample["steps"]) - int(base_sample["steps"]),
                "baseline_action_count": len(base_actions),
                "candidate_action_count": len(candidate_actions),
                "action_count_delta": len(candidate_actions) - len(base_actions),
                "exact_action_edit_distance": exact_edit,
                "normalized_exact_action_edit_distance": exact_edit / denominator,
                "tool_edit_distance": tool_edit,
                "normalized_tool_edit_distance": tool_edit / denominator,
                "exact_common_prefix_length": prefix,
                "first_exact_action_changed": base_actions[:1] != candidate_actions[:1],
                "exact_action_sequence_changed": base_actions != candidate_actions,
                "tool_sequence_changed": base_tool_sequence != candidate_tool_sequence,
            }
        )

    changed = [row for row in per_example if row["exact_action_sequence_changed"]]
    exact_edits = [float(row["exact_action_edit_distance"]) for row in per_example]
    normalized_exact_edits = [
        float(row["normalized_exact_action_edit_distance"]) for row in per_example
    ]
    tool_edits = [float(row["tool_edit_distance"]) for row in per_example]
    normalized_tool_edits = [
        float(row["normalized_tool_edit_distance"]) for row in per_example
    ]
    step_deltas = [int(row["step_delta"]) for row in per_example]
    absolute_step_deltas = [abs(value) for value in step_deltas]
    tool_deltas = {
        tool: candidate_tools[tool] - baseline_tools[tool]
        for tool in sorted(set(baseline_tools) | set(candidate_tools))
    }

    outcome_conditions = {
        "gain": lambda row: row["candidate_correct"] and not row["baseline_correct"],
        "regression": lambda row: row["baseline_correct"] and not row["candidate_correct"],
        "both_correct": lambda row: row["baseline_correct"] and row["candidate_correct"],
        "both_wrong": lambda row: not row["baseline_correct"] and not row["candidate_correct"],
    }
    outcome_group_shift = {
        name: group_shift([row for row in per_example if condition(row)])
        for name, condition in outcome_conditions.items()
    }

    result = {
        "schema_version": "paired-policy-behavior-shift-v1",
        "total": len(indices),
        "sequence_change": {
            "exact_action_sequence_changed": len(changed),
            "exact_action_sequence_unchanged": len(indices) - len(changed),
            "tool_sequence_changed": sum(row["tool_sequence_changed"] for row in per_example),
            "arguments_only_changed": sum(
                row["exact_action_sequence_changed"] and not row["tool_sequence_changed"]
                for row in per_example
            ),
            "first_exact_action_changed": sum(
                row["first_exact_action_changed"] for row in per_example
            ),
            "exact_edit_distance": distribution(exact_edits),
            "exact_edit_distance_changed_only": distribution(
                [float(row["exact_action_edit_distance"]) for row in changed]
            ),
            "normalized_exact_edit_distance": distribution(normalized_exact_edits),
            "tool_edit_distance": distribution(tool_edits),
            "normalized_tool_edit_distance": distribution(normalized_tool_edits),
        },
        "trajectory_length_change": {
            "mean_step_delta": statistics.fmean(step_deltas),
            "median_step_delta": statistics.median(step_deltas),
            "mean_absolute_step_delta": statistics.fmean(absolute_step_deltas),
            "shorter": sum(value < 0 for value in step_deltas),
            "same": sum(value == 0 for value in step_deltas),
            "longer": sum(value > 0 for value in step_deltas),
            "total_action_count_delta": sum(
                int(row["action_count_delta"]) for row in per_example
            ),
        },
        "marginal_tools": {
            "baseline": dict(sorted(baseline_tools.items())),
            "candidate": dict(sorted(candidate_tools.items())),
            "candidate_minus_baseline": tool_deltas,
            "jensen_shannon_divergence_bits": js_divergence_bits(
                baseline_tools, candidate_tools
            ),
        },
        "outcome_groups": {
            "gain": sum(
                row["candidate_correct"] and not row["baseline_correct"]
                for row in per_example
            ),
            "regression": sum(
                row["baseline_correct"] and not row["candidate_correct"]
                for row in per_example
            ),
            "both_correct": sum(
                row["baseline_correct"] and row["candidate_correct"]
                for row in per_example
            ),
            "both_wrong": sum(
                not row["baseline_correct"] and not row["candidate_correct"]
                for row in per_example
            ),
        },
        "outcome_group_shift": outcome_group_shift,
        "per_example": per_example,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
