#!/usr/bin/env python3
"""Rescore immutable replay-derived step features under reward ablations."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from rl.objectives.process_credit import (
    ProcessRewardConfig,
    StepFeature,
    allocate_process_rewards,
)


def load_config(path: Path) -> ProcessRewardConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = ProcessRewardConfig(
        **{
            key: value
            for key, value in payload.items()
            if not key.startswith("_")
        }
    )
    config.validate()
    return config


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return [
        row["process_reward"] if "process_reward" in row else row
        for row in rows
    ]


def score_rows(
    rows: list[dict[str, Any]],
    config: ProcessRewardConfig,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        features = [
            StepFeature(**step["features"])
            for step in row["steps"]
        ]
        result = allocate_process_rewards(
            str(row["trajectory_id"]),
            features,
            correct=bool(row["correct"]),
            config=config,
            diagnostics={"feature_source": "immutable_replay_derived"},
        )
        scored.append(result.to_dict())
    return scored


def normalized_entropy(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probabilities = [value / total for value in positive]
    return -sum(p * math.log(p) for p in probabilities) / math.log(len(probabilities))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for row in rows for step in row["steps"]]
    correct = [row for row in rows if row["correct"]]
    failed = [row for row in rows if not row["correct"]]
    error_steps = [
        step
        for step in steps
        if step["features"].get("error_type")
        or float(step["features"].get("tool_error") or 0) > 0
    ]
    back_slice_steps = [
        step
        for step in steps
        if float(step["features"].get("back_slice") or 0) > 0
    ]
    tool_reward: dict[str, float] = defaultdict(float)
    for step in steps:
        tool_reward[str(step.get("tool") or "unknown")] += float(step["reward"])
    signs = Counter(
        "positive" if step["reward"] > 0 else "negative" if step["reward"] < 0 else "zero"
        for step in steps
    )
    return {
        "episodes": len(rows),
        "correct_episodes": len(correct),
        "failed_episodes": len(failed),
        "steps": len(steps),
        "step_reward_signs": dict(signs),
        "correct_total_reward_mean": (
            fmean(row["total_reward"] for row in correct) if correct else None
        ),
        "failed_total_reward_mean": (
            fmean(row["total_reward"] for row in failed) if failed else None
        ),
        "correct_positive_credit_entropy_mean": (
            fmean(
                normalized_entropy([step["c_positive"] for step in row["steps"]])
                for row in correct
            )
            if correct
            else None
        ),
        "error_steps": {
            "count": len(error_steps),
            "positive_reward_count": sum(step["reward"] > 0 for step in error_steps),
            "negative_reward_count": sum(step["reward"] < 0 for step in error_steps),
            "zero_reward_count": sum(step["reward"] == 0 for step in error_steps),
            "positive_reward_mass": sum(
                max(0.0, float(step["reward"])) for step in error_steps
            ),
            "negative_reward_mass": sum(
                min(0.0, float(step["reward"])) for step in error_steps
            ),
            "back_slice_count": sum(
                float(step["features"].get("back_slice") or 0) > 0
                for step in error_steps
            ),
        },
        "back_slice_steps": {
            "count": len(back_slice_steps),
            "positive_reward_count": sum(
                step["reward"] > 0 for step in back_slice_steps
            ),
            "reward_mass": sum(
                float(step["reward"]) for step in back_slice_steps
            ),
        },
        "tool_reward_sum": dict(sorted(tool_reward.items())),
    }


def compare(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_steps = {
        (row["trajectory_id"], step["step_id"]): float(step["reward"])
        for row in baseline
        for step in row["steps"]
    }
    candidate_steps = {
        (row["trajectory_id"], step["step_id"]): float(step["reward"])
        for row in candidate
        for step in row["steps"]
    }
    if baseline_steps.keys() != candidate_steps.keys():
        raise ValueError("variant step identities do not align")
    differences = [
        candidate_steps[key] - baseline_steps[key]
        for key in baseline_steps
    ]
    return {
        "changed_step_count": sum(
            not math.isclose(value, 0.0, abs_tol=1e-10)
            for value in differences
        ),
        "step_count": len(differences),
        "mean_absolute_step_delta": (
            fmean(abs(value) for value in differences)
            if differences
            else 0.0
        ),
        "positive_to_nonpositive": sum(
            baseline_steps[key] > 0 and candidate_steps[key] <= 0
            for key in baseline_steps
        ),
        "nonpositive_to_positive": sum(
            baseline_steps[key] <= 0 and candidate_steps[key] > 0
            for key in baseline_steps
        ),
    }


def parse_variant(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("variant must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-trajectories", type=Path, required=True)
    parser.add_argument(
        "--variant",
        type=parse_variant,
        action="append",
        required=True,
        help="NAME=CONFIG_JSON; the first variant is the comparison baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = load_rows(args.scored_trajectories)
    variants = {}
    scored_by_name = {}
    for name, path in args.variant:
        if name in variants:
            raise SystemExit(f"duplicate variant name: {name}")
        config = load_config(path)
        scored = score_rows(source_rows, config)
        variants[name] = {
            "config_path": str(path),
            "config": config.__dict__,
            "summary": summarize(scored),
        }
        scored_by_name[name] = scored

    baseline_name = args.variant[0][0]
    comparisons = {
        name: compare(scored_by_name[baseline_name], scored)
        for name, scored in scored_by_name.items()
        if name != baseline_name
    }
    output = {
        "schema_version": "process-reward-variant-comparison-v1",
        "feature_source": str(args.scored_trajectories),
        "feature_source_note": (
            "all variants reuse identical replay-derived features; no model output "
            "or database execution is regenerated"
        ),
        "baseline": baseline_name,
        "variants": variants,
        "comparisons_to_baseline": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
