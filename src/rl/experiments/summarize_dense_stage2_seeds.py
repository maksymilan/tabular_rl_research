#!/usr/bin/env python3
"""Summarize the three independent full-dev scale120 replications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement", action="append", required=True)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "selected":
        raise ValueError("Stage2 summary requires a selected Stage1 method")
    rows: list[dict[str, Any]] = []
    for value in args.requirement:
        seed_text, separator, path_text = value.partition(":")
        if not separator:
            raise ValueError("--requirement must be SEED:PATH")
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        candidate = payload["candidate_metrics"]["greedy"]
        baseline = payload["sft2"]["greedy"]
        if candidate["questions"] != 1534 or baseline["correct"] != 762:
            raise ValueError(f"incomplete or wrong-baseline seed result: {value}")
        rows.append(
            {
                "seed": int(seed_text),
                "candidate": payload["candidate"],
                "requirements_status": payload["status"],
                "correct": int(candidate["correct"]),
                "accuracy": float(candidate["greedy_at_1"]),
                "valid_rate": float(candidate["valid_rate"]),
                "delta_correct": int(candidate["correct"]) - 762,
                "delta_accuracy_pp": 100 * (float(candidate["greedy_at_1"]) - 762 / 1534),
            }
        )
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != [101, 202, 303]:
        raise ValueError("Stage2 requires exactly independent train seeds 101/202/303")
    accuracies = [row["accuracy"] for row in rows]
    valid_rates = [row["valid_rate"] for row in rows]
    all_positive = all(row["correct"] > 762 for row in rows)
    all_requirements = all(row["requirements_status"] == "passed" for row in rows)
    valid_floor = fmean(valid_rates) >= 1194 / 1534 - 0.01
    passed = all_positive and all_requirements and valid_floor
    payload = {
        "schema_version": "dense-stage2-scale120-seed-summary-v1",
        "status": "passed" if passed else "stopped",
        "selected_stage1_method": selection["selected"],
        "training_contract": {
            "tasks": 120,
            "trajectories": 480,
            "difficulty_counts": {"simple": 40, "moderate": 40, "challenging": 40},
            "independent_sft2_initialization": True,
            "train_seeds": [101, 202, 303],
        },
        "seeds": rows,
        "aggregate": {
            "accuracy_mean": fmean(accuracies),
            "accuracy_population_std": pstdev(accuracies),
            "valid_rate_mean": fmean(valid_rates),
            "valid_rate_population_std": pstdev(valid_rates),
            "positive_directions": sum(row["correct"] > 762 for row in rows),
        },
        "continuation_checks": {
            "all_three_full_dev_directions_positive": all_positive,
            "all_three_test_requirements_passed": all_requirements,
            "mean_valid_within_1pp_of_sft2": valid_floor,
        },
        "next_step": (
            "eligible for one final-method K4 evaluation"
            if passed
            else "stop this RL direction; do not add reward variants"
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"refusing to overwrite a different summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
