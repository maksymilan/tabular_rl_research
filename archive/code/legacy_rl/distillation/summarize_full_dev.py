#!/usr/bin/env python3
"""Compatibility frontend for historical full-dev summary JSON.

New experiments should call ``src/rl/diagnostics/analyze_evaluation_results.py``.
This entrypoint retains the old CLI and output schema for frozen queues.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from src.rl.diagnostics.analyze_evaluation_results import (
        analyze_results,
        exact_mcnemar_p,
    )
except ModuleNotFoundError:  # Direct execution from a runtime snapshot.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
    from analyze_evaluation_results import analyze_results, exact_mcnemar_p


def _legacy_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": values["correct"],
        "accuracy": values["accuracy"],
        "valid": values["legal"],
        "valid_rate": values["legal_rate"],
        "avg_steps": values["mean_steps"],
        "difficulty": {
            level: {
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
            for level, metrics in values["difficulty"].items()
        },
    }


def _legacy_pair(values: dict[str, Any]) -> dict[str, Any]:
    accuracy = values["accuracy"]
    return {
        "gains": accuracy["gains"],
        "regressions": accuracy["regressions"],
        "net": accuracy["net"],
        "exact_mcnemar_p": accuracy["exact_mcnemar_p"],
        "gain_example_indices": accuracy["gain_example_indices"],
        "regression_example_indices": accuracy["regression_example_indices"],
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
    arm_paths = {"candidate": args.candidate}
    comparisons = []
    for item in args.baseline:
        label, separator, path = item.partition(":")
        if not separator or not label or not path:
            raise ValueError("baseline must use LABEL:all.jsonl")
        if label in arm_paths:
            raise ValueError(f"duplicate baseline label: {label}")
        arm_paths[label] = Path(path)
        comparisons.append(("candidate", label))
    unified = analyze_results(
        examples_path=args.eval_inputs,
        arm_paths=arm_paths,
        comparisons=comparisons,
        expected_count=1534,
    )
    output = {
        "schema_version": "streaming-teacher-union-full-dev-summary-v1",
        "candidate": _legacy_metrics(unified["arms"]["candidate"]),
        "baselines": {
            label: _legacy_metrics(unified["arms"][label])
            for label in arm_paths
            if label != "candidate"
        },
        "paired": {
            label: _legacy_pair(
                unified["comparisons"][f"candidate_vs_{label}"]
            )
            for label in arm_paths
            if label != "candidate"
        },
        "interpretation": "single-run exploratory result; no K4 finalist evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
