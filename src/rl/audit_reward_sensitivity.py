#!/usr/bin/env python3
"""Offline sensitivity audit for the atomic process-reward coefficients.

The audit reuses harness-authored step features from a verified successful SFT cohort and replays
semantic failure trajectories.  It never uses model-authored reasoning as evidence.  Each
one-factor scenario is evaluated against the same episodes, so changes are attributable to the
named coefficient only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from process_credit import (  # noqa: E402
    EpisodeReward,
    ProcessRewardConfig,
    StepFeature,
    allocate_process_rewards,
    replay_step_features,
)


POSITIVE_WEIGHTS = (
    "w_back_slice",
    "w_new_evidence",
    "w_search_reduction",
    "w_feedback_response",
    "w_target_potential",
)
LOCAL_PENALTIES = (
    "lambda_tool_error",
    "lambda_repeat_without_feedback",
    "lambda_legal_no_state_change",
    "lambda_ignored_feedback",
    "lambda_unsupported_guess",
    "lambda_empty_result",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def normalized_entropy(shares: list[float]) -> float:
    positive = [value for value in shares if value > 0]
    if len(positive) <= 1:
        return 0.0
    return -sum(value * math.log(value) for value in positive) / math.log(len(positive))


def load_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def _step_feature(payload: dict[str, Any]) -> StepFeature:
    field_names = {item.name for item in dataclasses.fields(StepFeature)}
    return StepFeature(**{
        key: value
        for key, value in payload.items()
        if key in field_names
    })


def load_success_features(
    actions_path: Path,
    diagnostics_path: Path,
) -> list[tuple[str, bool, list[StepFeature], dict[str, Any]]]:
    grouped: dict[str, list[StepFeature]] = defaultdict(list)
    for row in read_jsonl(actions_path):
        grouped[str(row["trajectory_id"])].append(_step_feature(row["features"]))
    diagnostics = {
        str(row["trajectory_id"]): dict(row.get("diagnostics") or {})
        for row in read_jsonl(diagnostics_path)
    }
    if set(grouped) != set(diagnostics):
        missing_actions = sorted(set(diagnostics) - set(grouped))
        missing_diagnostics = sorted(set(grouped) - set(diagnostics))
        raise ValueError(
            "successful action/diagnostic episode mismatch: "
            f"missing_actions={missing_actions[:5]}, "
            f"missing_diagnostics={missing_diagnostics[:5]}"
        )
    episodes = []
    for trajectory_id in sorted(grouped):
        features = sorted(grouped[trajectory_id], key=lambda item: item.action_index)
        correct = bool(diagnostics[trajectory_id].get("replay_correct"))
        if not correct:
            raise ValueError(f"successful cohort contains failed replay {trajectory_id}")
        episodes.append((trajectory_id, True, features, diagnostics[trajectory_id]))
    return episodes


def load_failure_features(
    paths: list[Path],
) -> list[tuple[str, bool, list[StepFeature], dict[str, Any]]]:
    episodes = []
    seen: set[str] = set()
    for path in paths:
        for trajectory in read_jsonl(path):
            trajectory_id = str(trajectory.get("trajectory_id"))
            if not trajectory_id or trajectory_id in seen:
                raise ValueError(f"missing or duplicate failure trajectory id {trajectory_id!r}")
            seen.add(trajectory_id)
            features, diagnostics = replay_step_features(
                trajectory,
                denotation_comparison="bird-set",
            )
            if diagnostics.get("replay_correct"):
                raise ValueError(f"failure cohort unexpectedly replays correct: {trajectory_id}")
            episodes.append((trajectory_id, False, features, diagnostics))
    return episodes


def clone_features(features: list[StepFeature]) -> list[StepFeature]:
    return [dataclasses.replace(feature, references=list(feature.references)) for feature in features]


def score_episodes(
    episodes: list[tuple[str, bool, list[StepFeature], dict[str, Any]]],
    config: ProcessRewardConfig,
) -> list[EpisodeReward]:
    return [
        allocate_process_rewards(
            trajectory_id,
            clone_features(features),
            correct=correct,
            config=config,
            diagnostics=diagnostics,
        )
        for trajectory_id, correct, features, diagnostics in episodes
    ]


def build_scenarios(config: ProcessRewardConfig) -> list[tuple[str, str, float, ProcessRewardConfig]]:
    scenarios = [("baseline", "baseline", 1.0, config)]
    for field in POSITIVE_WEIGHTS:
        base = getattr(config, field)
        for multiplier in (0.25, 0.5, 2.0, 4.0):
            candidate = dataclasses.replace(config, **{field: base * multiplier})
            candidate.validate()
            scenarios.append((
                f"{field}@{multiplier:g}x",
                field,
                multiplier,
                candidate,
            ))
    for value in (0.0, 0.025, 0.1, 0.2):
        candidate = dataclasses.replace(config, eta_failure_progress=value)
        candidate.validate()
        scenarios.append((
            f"eta_failure_progress@{value:g}",
            "eta_failure_progress",
            value,
            candidate,
        ))
    for value in (0.0, 0.01, 0.04, 0.08):
        candidate = dataclasses.replace(config, lambda_answer_format=value)
        candidate.validate()
        scenarios.append((
            f"lambda_answer_format@{value:g}",
            "lambda_answer_format",
            value,
            candidate,
        ))
    for value in (0.15, 0.2, 0.45, 0.6):
        candidate = dataclasses.replace(config, lambda_terminal_failure=value)
        candidate.validate()
        scenarios.append((
            f"lambda_terminal_failure@{value:g}",
            "lambda_terminal_failure",
            value,
            candidate,
        ))
    for multiplier in (0.5, 1.5, 2.0):
        candidate = dataclasses.replace(
            config,
            **{
                field: getattr(config, field) * multiplier
                for field in LOCAL_PENALTIES
            },
        )
        candidate.validate()
        scenarios.append((
            f"local_penalties@{multiplier:g}x",
            "local_penalties",
            multiplier,
            candidate,
        ))
    for value in (0.4, 0.6, 0.95):
        candidate = dataclasses.replace(config, penalty_cap=value)
        candidate.validate()
        scenarios.append((
            f"penalty_cap@{value:g}",
            "penalty_cap",
            value,
            candidate,
        ))
    return scenarios


def reward_map(results: list[EpisodeReward]) -> dict[tuple[str, int], float]:
    return {
        (episode.trajectory_id, step.action_index): step.reward
        for episode in results
        for step in episode.steps
    }


def summarize_results(
    results: list[EpisodeReward],
    *,
    baseline_rewards: dict[tuple[str, int], float],
) -> dict[str, Any]:
    successful = [episode for episode in results if episode.correct]
    failed = [episode for episode in results if not episode.correct]
    all_steps = [
        (episode, step)
        for episode in results
        for step in episode.steps
    ]
    current_rewards = reward_map(results)
    common_keys = set(current_rewards) & set(baseline_rewards)
    reward_l1 = sum(
        abs(current_rewards[key] - baseline_rewards[key])
        for key in common_keys
    )
    sign_changes = sum(
        (current_rewards[key] > 0) != (baseline_rewards[key] > 0)
        or (current_rewards[key] < 0) != (baseline_rewards[key] < 0)
        for key in common_keys
    )
    max_shares = []
    entropies = []
    for episode in successful:
        shares = [step.c_positive for step in episode.steps]
        max_shares.append(max(shares, default=0.0))
        entropies.append(normalized_entropy(shares))
    tool_abs_mass: Counter[str] = Counter()
    for _, step in all_steps:
        tool_abs_mass[str(step.tool or "unknown")] += abs(step.reward)
    total_abs_mass = sum(tool_abs_mass.values())
    dominant_tool, dominant_mass = (
        tool_abs_mass.most_common(1)[0] if tool_abs_mass else ("none", 0.0)
    )
    failed_offsets = []
    for episode in failed:
        offset = (
            float(episode.diagnostics.get("failure_progress_mass") or 0.0)
            + float(episode.diagnostics.get("answer_format_mass") or 0.0)
        )
        failed_offsets.append(
            offset / episode.capped_penalty
            if episode.capped_penalty > 0
            else math.inf
        )
    success_totals = [episode.total_reward for episode in successful]
    failure_totals = [episode.total_reward for episode in failed]
    rewards = [step.reward for _, step in all_steps]
    return {
        "episodes": len(results),
        "successful_episodes": len(successful),
        "failed_episodes": len(failed),
        "steps": len(all_steps),
        "reward_l1_vs_baseline": round(reward_l1, 8),
        "step_sign_changes_vs_baseline": sign_changes,
        "success_total_reward": {
            "min": round(min(success_totals), 8) if success_totals else None,
            "mean": round(statistics.mean(success_totals), 8) if success_totals else None,
            "max": round(max(success_totals), 8) if success_totals else None,
            "nonpositive": sum(value <= 0 for value in success_totals),
        },
        "failure_total_reward": {
            "min": round(min(failure_totals), 8) if failure_totals else None,
            "mean": round(statistics.mean(failure_totals), 8) if failure_totals else None,
            "max": round(max(failure_totals), 8) if failure_totals else None,
            "nonnegative": sum(value >= 0 for value in failure_totals),
        },
        "step_reward": {
            "min": round(min(rewards), 8) if rewards else None,
            "mean": round(statistics.mean(rewards), 8) if rewards else None,
            "p90": round(percentile(rewards, 0.9), 8),
            "max": round(max(rewards), 8) if rewards else None,
            "positive": sum(value > 0 for value in rewards),
            "negative": sum(value < 0 for value in rewards),
            "zero": sum(value == 0 for value in rewards),
            "mean_absolute": (
                round(statistics.mean(abs(value) for value in rewards), 8)
                if rewards else None
            ),
        },
        "positive_credit_concentration": {
            "max_share_mean": round(statistics.mean(max_shares), 8) if max_shares else None,
            "max_share_p90": round(percentile(max_shares, 0.9), 8),
            "normalized_entropy_mean": (
                round(statistics.mean(entropies), 8) if entropies else None
            ),
        },
        "penalty_cap": {
            "successful_capped": sum(
                math.isclose(episode.capped_penalty, episode.raw_penalty_mass)
                is False
                for episode in successful
            ),
            "failed_capped": sum(
                math.isclose(episode.capped_penalty, episode.raw_penalty_mass)
                is False
                for episode in failed
            ),
        },
        "failure_positive_offset_ratio": {
            "mean": (
                round(statistics.mean(failed_offsets), 8)
                if failed_offsets else None
            ),
            "max": round(max(failed_offsets), 8) if failed_offsets else None,
        },
        "dominant_tool_absolute_reward": {
            "tool": dominant_tool,
            "share": round(dominant_mass / total_abs_mass, 8) if total_abs_mass else 0.0,
            "absolute_mass": round(dominant_mass, 8),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--successful-actions", type=Path, required=True)
    parser.add_argument("--successful-diagnostics", type=Path, required=True)
    parser.add_argument("--failure-trajectory", type=Path, action="append", default=[])
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config_json)
    episodes = load_success_features(
        args.successful_actions,
        args.successful_diagnostics,
    )
    episodes.extend(load_failure_features(args.failure_trajectory))
    baseline_results = score_episodes(episodes, config)
    baseline_rewards = reward_map(baseline_results)

    scenarios = []
    for name, axis, value, candidate in build_scenarios(config):
        results = baseline_results if name == "baseline" else score_episodes(episodes, candidate)
        scenarios.append({
            "name": name,
            "axis": axis,
            "value": value,
            "config": dataclasses.asdict(candidate),
            "metrics": summarize_results(
                results,
                baseline_rewards=baseline_rewards,
            ),
        })

    output = {
        "schema_version": "atomic-process-reward-sensitivity-v1",
        "denotation_comparison": "bird-set",
        "successful_actions": str(args.successful_actions.resolve()),
        "successful_diagnostics": str(args.successful_diagnostics.resolve()),
        "failure_trajectories": [
            str(path.resolve()) for path in args.failure_trajectory
        ],
        "baseline_config": dataclasses.asdict(config),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "invariants": {
            "all_scenarios_keep_success_positive": all(
                row["metrics"]["success_total_reward"]["nonpositive"] == 0
                for row in scenarios
            ),
            "all_scenarios_keep_failures_negative": all(
                row["metrics"]["failure_total_reward"]["nonnegative"] == 0
                for row in scenarios
            ),
            "all_scenarios_use_same_steps": len({
                row["metrics"]["steps"] for row in scenarios
            }) == 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scenario_count": output["scenario_count"],
        "episodes": scenarios[0]["metrics"]["episodes"],
        "steps": scenarios[0]["metrics"]["steps"],
        "invariants": output["invariants"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if all(output["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
