#!/usr/bin/env python3
"""Compatibility frontend for historical routed-coupled behavior summaries.

New experiments should use ``analyze_evaluation_results.py`` directly.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from src.rl.diagnostics.analyze_evaluation_results import analyze_results
except ModuleNotFoundError:  # Direct execution from a runtime snapshot.
    from analyze_evaluation_results import analyze_results


def _legacy_pair(values: dict[str, Any]) -> dict[str, Any]:
    accuracy = values["accuracy"]
    legal = values["legal"]
    sequence = values["sequence_change"]
    return {
        "gains": accuracy["gain_example_indices"],
        "regressions": accuracy["regression_example_indices"],
        "net": accuracy["net"],
        "exact_p": accuracy["exact_mcnemar_p"],
        "legal_gains": legal["gain_example_indices"],
        "legal_regressions": legal["regression_example_indices"],
        "legal_net": legal["net"],
        "legal_exact_p": legal["exact_mcnemar_p"],
        "action_sequence_changed": sequence[
            "exact_action_sequence_changed_indices"
        ],
        "action_sequence_changed_count": sequence[
            "exact_action_sequence_changed"
        ],
        "first_action_changed": sequence["first_exact_action_changed_indices"],
        "first_action_changed_count": sequence["first_exact_action_changed"],
    }


def _legacy_arm(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": values["correct"],
        "total": values["total"],
        "accuracy": values["accuracy"],
        "legal": values["legal"],
        "legal_rate": values["legal_rate"],
        "mean_steps": values["mean_steps"],
        "difficulty": {
            level: {
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
            for level, metrics in values["difficulty"].items()
        },
        "adjacent_exact_repeat_calls": values["adjacent_exact_repeat_calls"],
        "questions_with_adjacent_exact_repeat": values[
            "questions_with_adjacent_exact_repeat"
        ],
        "action_count": values["action_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {name: Path(path) for name, path in args.candidate}
    if len(candidates) != len(args.candidate):
        raise SystemExit("candidate names must be unique")
    comparisons = [(name, "sft2") for name in candidates]
    comparisons.extend(
        (right, left) for left, right in combinations(candidates, 2)
    )
    unified = analyze_results(
        examples_path=args.examples,
        indices_path=args.indices,
        arm_paths={"sft2": args.baseline, **candidates},
        comparisons=comparisons,
        protocol_version="version36",
        temperature=0.0,
        top_p=1.0,
        denotation_comparison="bird-set",
    )
    result = {
        "schema_version": "routed-coupled-behavior-smoke-summary-v1",
        "indices": unified["cohort"]["indices"],
        "difficulty_counts": unified["cohort"]["difficulty_counts"],
        "baseline": _legacy_arm(unified["arms"]["sft2"]),
        "candidates": {
            name: {
                **_legacy_arm(unified["arms"][name]),
                "vs_sft2": _legacy_pair(
                    unified["comparisons"][f"{name}_vs_sft2"]
                ),
            }
            for name in candidates
        },
        "candidate_pairwise": {
            f"{right}_vs_{left}": _legacy_pair(
                unified["comparisons"][f"{right}_vs_{left}"]
            )
            for left, right in combinations(candidates, 2)
        },
    }
    if set(candidates) == {"lr1e-6", "lr4e-6"}:
        result["lr4e-6_vs_lr1e-6"] = _legacy_pair(
            unified["comparisons"]["lr4e-6_vs_lr1e-6"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
