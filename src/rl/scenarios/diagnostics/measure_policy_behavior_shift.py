#!/usr/bin/env python3
"""Compatibility frontend for the historical two-arm policy-shift schema.

New experiments should use ``analyze_evaluation_results.py`` directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from rl.scenarios.diagnostics.analyze_evaluation_results import analyze_results
except ModuleNotFoundError:  # Direct execution from a runtime snapshot.
    from rl.scenarios.diagnostics.analyze_evaluation_results import analyze_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_results(
        examples_path=args.examples,
        indices_path=args.indices,
        arm_paths={"baseline": args.baseline, "candidate": args.candidate},
        comparisons=[("candidate", "baseline")],
        include_per_example=True,
    )
    comparison = result["comparisons"]["candidate_vs_baseline"]
    output = {
        "schema_version": "paired-policy-behavior-shift-v1",
        "total": result["cohort"]["total"],
        "sequence_change": comparison["sequence_change"],
        "trajectory_length_change": comparison["trajectory_length_change"],
        "marginal_tools": comparison["marginal_tools"],
        "outcome_groups": comparison["outcome_groups"],
        "outcome_group_shift": comparison["outcome_group_shift"],
        "per_example": comparison["per_example"],
    }
    # Remove index-list extensions that did not exist in the historical schema.
    output["sequence_change"].pop("exact_action_sequence_changed_indices", None)
    output["sequence_change"].pop("first_exact_action_changed_indices", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
