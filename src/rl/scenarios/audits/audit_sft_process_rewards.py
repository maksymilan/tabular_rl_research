#!/usr/bin/env python3
"""Measure atomic process-reward coverage on the exact turns retained by an SFT dataset.

The SFT index identifies the causal source episode and legal source step for every supervised
target.  This audit replays each complete source episode once under the canonical harness, assigns
reward without consulting model-authored reasoning, and then projects those rewards onto the turns
that survived SFT filtering.  Ablations reuse the same replay-derived features, so only the named
reward term changes.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "sft")]

from rl.objectives.process_credit import (  # noqa: E402
    ProcessRewardConfig,
    allocate_process_rewards,
    replay_step_features,
)
from rl.diagnostics.common_io import (  # noqa: E402
    load_process_reward_config,
    percentile,
    read_jsonl,
    sha256_file,
)
from protocol import TOOLS  # noqa: E402


TOOL_BUCKETS = frozenset(set(TOOLS) | {"unknown"})
FEATURES = (
    "back_slice",
    "new_used_evidence",
    "search_reduction",
    "feedback_response",
    "target_table_delta",
    "target_column_delta",
    "target_row_delta",
    "target_potential_delta",
    "answer_format",
    "tool_error",
    "repeat_without_feedback",
    "legal_no_state_change",
    "ignored_feedback",
    "unsupported_guess",
    "empty_result_penalty",
)


sha256 = sha256_file


def entropy(shares: list[float]) -> float:
    positive = [value for value in shares if value > 0]
    if len(positive) <= 1:
        return 0.0
    raw = -sum(value * math.log(value) for value in positive)
    return raw / math.log(len(positive))


load_config = load_process_reward_config


def ablation_configs(config: ProcessRewardConfig) -> dict[str, ProcessRewardConfig]:
    """Return one-factor ablations with every unrelated variable held fixed."""
    return {
        "full": config,
        "without_back_slice": dataclasses.replace(config, w_back_slice=0.0),
        "without_new_evidence": dataclasses.replace(config, w_new_evidence=0.0),
        "without_search_reduction": dataclasses.replace(config, w_search_reduction=0.0),
        "without_feedback_response": dataclasses.replace(config, w_feedback_response=0.0),
        "without_target_potential": dataclasses.replace(config, w_target_potential=0.0),
        "without_answer_format": dataclasses.replace(config, lambda_answer_format=0.0),
        "without_failure_progress": dataclasses.replace(config, eta_failure_progress=0.0),
        "without_tool_error_penalty": dataclasses.replace(config, lambda_tool_error=0.0),
        "without_repeat_penalty": dataclasses.replace(
            config, lambda_repeat_without_feedback=0.0
        ),
        "without_noop_penalty": dataclasses.replace(
            config, lambda_legal_no_state_change=0.0
        ),
        "without_ignored_feedback_penalty": dataclasses.replace(
            config, lambda_ignored_feedback=0.0
        ),
        "without_unsupported_guess_penalty": dataclasses.replace(
            config, lambda_unsupported_guess=0.0
        ),
        "without_empty_result_penalty": dataclasses.replace(
            config, lambda_empty_result=0.0
        ),
        "without_local_penalties": dataclasses.replace(
            config,
            lambda_tool_error=0.0,
            lambda_repeat_without_feedback=0.0,
            lambda_legal_no_state_change=0.0,
            lambda_ignored_feedback=0.0,
            lambda_unsupported_guess=0.0,
            lambda_empty_result=0.0,
        ),
    }


def empty_tool_stats() -> dict[str, Any]:
    return {
        "source_actions": 0,
        "source_legal_steps": 0,
        "source_positive_steps": 0,
        "source_negative_steps": 0,
        "source_zero_steps": 0,
        "source_reward_sum": 0.0,
        "source_feature_hits": collections.Counter(),
        "source_feature_mass": collections.Counter(),
        "retained_sft_steps": 0,
        "positive_steps": 0,
        "negative_steps": 0,
        "zero_steps": 0,
        "reward_sum": 0.0,
        "positive_credit_sum": 0.0,
        "negative_credit_sum": 0.0,
        "feature_hits": collections.Counter(),
        "feature_mass": collections.Counter(),
    }


def serialize_tool_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for tool in sorted(TOOL_BUCKETS):
        item = stats[tool]
        output[tool] = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in item.items()
            if key not in {
                "feature_hits",
                "feature_mass",
                "source_feature_hits",
                "source_feature_mass",
            }
        }
        output[tool]["feature_hits"] = {
            name: int(item["feature_hits"][name]) for name in FEATURES
        }
        output[tool]["feature_mass"] = {
            name: round(item["feature_mass"][name], 8) for name in FEATURES
        }
        output[tool]["source_feature_hits"] = {
            name: int(item["source_feature_hits"][name]) for name in FEATURES
        }
        output[tool]["source_feature_mass"] = {
            name: round(item["source_feature_mass"][name], 8) for name in FEATURES
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-index", type=Path, required=True)
    parser.add_argument("--trajectory-input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--denotation-comparison", choices=("bird-set",), default="bird-set")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_json)
    configs = ablation_configs(config)
    for value in configs.values():
        value.validate()

    index_rows = read_jsonl(args.sft_index)
    retained_keys = {
        (str(row["source_episode_id"]), str(row["source_step_id"])) for row in index_rows
    }
    if len(retained_keys) != len(index_rows):
        raise ValueError("SFT index contains duplicate source episode/step targets")
    retained_episode_ids = {episode_id for episode_id, _ in retained_keys}

    trajectories: dict[str, dict[str, Any]] = {}
    trajectory_sources: dict[str, str] = {}
    for input_path in args.trajectory_input:
        for trajectory in read_jsonl(input_path):
            trajectory_id = str(trajectory["trajectory_id"])
            if trajectory_id in trajectories:
                raise ValueError(f"duplicate trajectory_id across source inputs: {trajectory_id}")
            trajectories[trajectory_id] = trajectory
            trajectory_sources[trajectory_id] = str(input_path.resolve())
    missing_episodes = sorted(retained_episode_ids - trajectories.keys())
    if missing_episodes:
        raise ValueError(f"{len(missing_episodes)} SFT episodes have no source trajectory")

    selected_ids = sorted(retained_episode_ids)
    if args.limit_episodes > 0:
        selected_ids = selected_ids[: args.limit_episodes]
        retained_keys = {key for key in retained_keys if key[0] in set(selected_ids)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "scored_sft_steps.jsonl"
    source_details_path = args.output_dir / "scored_source_actions.jsonl"
    diagnostics_path = args.output_dir / "episode_diagnostics.jsonl"
    errors_path = args.output_dir / "replay_errors.jsonl"
    episode_results = []
    replay_errors = []
    all_source_step_keys: set[tuple[str, str]] = set()

    for index, trajectory_id in enumerate(selected_ids, 1):
        trajectory = trajectories[trajectory_id]
        try:
            features, diagnostics = replay_step_features(
                trajectory,
                denotation_comparison=args.denotation_comparison,
            )
            correct = bool(
                diagnostics["replay_correct"] and trajectory.get("label_status") == "verified"
            )
            results = {
                name: allocate_process_rewards(
                    trajectory_id,
                    features,
                    correct=correct,
                    config=ablation,
                    diagnostics=diagnostics,
                )
                for name, ablation in configs.items()
            }
            episode_results.append(
                {
                    "trajectory_id": trajectory_id,
                    "source": trajectory_sources[trajectory_id],
                    "features": features,
                    "diagnostics": diagnostics,
                    "ablations": results,
                }
            )
            all_source_step_keys.update(
                (trajectory_id, step.step_id)
                for step in results["full"].steps
                if step.features["legal_success"]
            )
        except Exception as exc:  # preserve a complete audit of problematic source trajectories
            replay_errors.append(
                {
                    "trajectory_id": trajectory_id,
                    "source": trajectory_sources[trajectory_id],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if not args.quiet and (index == 1 or index % 25 == 0 or index == len(selected_ids)):
            print(
                f"[{index}/{len(selected_ids)}] scored={len(episode_results)} "
                f"errors={len(replay_errors)}"
            )

    with errors_path.open("w", encoding="utf-8") as sink:
        for row in replay_errors:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing_steps = sorted(retained_keys - all_source_step_keys)
    full_rewards: dict[tuple[str, str], float] = {}
    full_all_rewards: dict[tuple[str, str], float] = {}
    tool_stats = {tool: empty_tool_stats() for tool in TOOL_BUCKETS}
    with (
        details_path.open("w", encoding="utf-8") as sink,
        source_details_path.open("w", encoding="utf-8") as source_sink,
        diagnostics_path.open("w", encoding="utf-8") as diagnostics_sink,
    ):
        for item in episode_results:
            trajectory_id = item["trajectory_id"]
            full = item["ablations"]["full"]
            diagnostics_sink.write(
                json.dumps(
                    {
                        "trajectory_id": trajectory_id,
                        "difficulty": trajectories[trajectory_id].get("difficulty"),
                        "diagnostics": item["diagnostics"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for step in full.steps:
                tool = step.tool if step.tool in tool_stats else "unknown"
                full_all_rewards[(trajectory_id, step.step_id)] = step.reward
                tool_stats[tool]["source_actions"] += 1
                retained = (trajectory_id, step.step_id) in retained_keys
                if step.features["legal_success"]:
                    tool_stats[tool]["source_legal_steps"] += 1
                tool_stats[tool]["source_positive_steps"] += step.reward > 0
                tool_stats[tool]["source_negative_steps"] += step.reward < 0
                tool_stats[tool]["source_zero_steps"] += step.reward == 0
                tool_stats[tool]["source_reward_sum"] += step.reward
                for name in FEATURES:
                    value = float(step.features.get(name, 0))
                    tool_stats[tool]["source_feature_hits"][name] += value > 0
                    tool_stats[tool]["source_feature_mass"][name] += value
                source_sink.write(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "step_id": step.step_id,
                            "tool": step.tool,
                            "retained_sft_step": retained,
                            "reward": step.reward,
                            "c_positive": step.c_positive,
                            "c_negative": step.c_negative,
                            "features": step.features,
                            "difficulty": trajectories[trajectory_id].get("difficulty"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if not retained:
                    continue
                tool_stats[tool]["retained_sft_steps"] += 1
                tool_stats[tool]["positive_steps"] += step.reward > 0
                tool_stats[tool]["negative_steps"] += step.reward < 0
                tool_stats[tool]["zero_steps"] += step.reward == 0
                tool_stats[tool]["reward_sum"] += step.reward
                tool_stats[tool]["positive_credit_sum"] += step.c_positive
                tool_stats[tool]["negative_credit_sum"] += step.c_negative
                for name in FEATURES:
                    value = float(step.features.get(name, 0))
                    tool_stats[tool]["feature_hits"][name] += value > 0
                    tool_stats[tool]["feature_mass"][name] += value
                full_rewards[(trajectory_id, step.step_id)] = step.reward
                sink.write(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "step_id": step.step_id,
                            "tool": step.tool,
                            "reward": step.reward,
                            "c_positive": step.c_positive,
                            "c_negative": step.c_negative,
                            "features": step.features,
                            "difficulty": trajectories[trajectory_id].get("difficulty"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Summarize ablations after all full rewards have an unambiguous episode/step key.
    ablations: dict[str, Any] = {}
    for name in configs:
        retained_steps = []
        process_updates = 0
        max_shares = []
        entropies = []
        full_episode_reward_mass = 0.0
        retained_reward_mass = 0.0
        changed = lost_positive = gained_positive = 0
        l1 = 0.0
        tool_delta = collections.Counter()
        source_changed = source_lost_positive = source_gained_positive = 0
        source_l1 = 0.0
        source_tool_delta = collections.Counter()
        for item in episode_results:
            result = item["ablations"][name]
            process_updates += result.process_update
            full_episode_reward_mass += result.total_reward
            max_shares.append(max((step.c_positive for step in result.steps), default=0.0))
            entropies.append(entropy([step.c_positive for step in result.steps]))
            for step in result.steps:
                key = (result.trajectory_id, step.step_id)
                if name != "full":
                    source_before = full_all_rewards[key]
                    source_delta = step.reward - source_before
                    source_changed += abs(source_delta) > 1e-10
                    source_lost_positive += source_before > 0 and step.reward <= 0
                    source_gained_positive += source_before <= 0 and step.reward > 0
                    source_l1 += abs(source_delta)
                    source_tool_delta[step.tool or "unknown"] += source_delta
                if key not in retained_keys:
                    continue
                retained_steps.append(step)
                retained_reward_mass += step.reward
                if name != "full":
                    before = full_rewards[key]
                    delta = step.reward - before
                    changed += abs(delta) > 1e-10
                    lost_positive += before > 0 and step.reward <= 0
                    gained_positive += before <= 0 and step.reward > 0
                    l1 += abs(delta)
                    tool_delta[step.tool or "unknown"] += delta
        reward_values = [step.reward for step in retained_steps]
        ablations[name] = {
            "episodes_with_process_update": process_updates,
            "full_episode_reward_mass": round(full_episode_reward_mass, 8),
            "retained_sft_reward_mass": round(retained_reward_mass, 8),
            "retained_step_signs": {
                "positive": sum(value > 0 for value in reward_values),
                "negative": sum(value < 0 for value in reward_values),
                "zero": sum(value == 0 for value in reward_values),
            },
            "retained_reward": {
                "mean": (
                    round(statistics.mean(reward_values), 8) if reward_values else 0.0
                ),
                "p50": (
                    round(statistics.median(reward_values), 8) if reward_values else 0.0
                ),
                "p90": round(percentile(reward_values, 0.9), 8),
            },
            "positive_credit_concentration": {
                "max_share_mean": (
                    round(statistics.mean(max_shares), 8) if max_shares else 0.0
                ),
                "max_share_p90": round(percentile(max_shares, 0.9), 8),
                "normalized_entropy_mean": (
                    round(statistics.mean(entropies), 8) if entropies else 0.0
                ),
            },
            "effect_vs_full_on_retained_steps": {
                "changed_steps": changed,
                "lost_positive_steps": lost_positive,
                "gained_positive_steps": gained_positive,
                "reward_l1": round(l1, 8),
                "reward_delta_by_tool": {
                    key: round(value, 8) for key, value in sorted(tool_delta.items())
                },
            },
            "effect_vs_full_on_all_source_actions": {
                "changed_steps": source_changed,
                "lost_positive_steps": source_lost_positive,
                "gained_positive_steps": source_gained_positive,
                "reward_l1": round(source_l1, 8),
                "reward_delta_by_tool": {
                    key: round(value, 8)
                    for key, value in sorted(source_tool_delta.items())
                },
            },
        }

    diagnostics = [item["diagnostics"] for item in episode_results]
    summary = {
        "schema_version": "atomic-process-reward-sft-audit-v1",
        "denotation_comparison": args.denotation_comparison,
        "sft_index": {
            "path": str(args.sft_index.resolve()),
            "sha256": sha256(args.sft_index),
            "records": len(index_rows),
            "unique_retained_steps": len(retained_keys),
            "retained_source_episodes": len(retained_episode_ids),
        },
        "trajectory_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.trajectory_input
        ],
        "available_source_trajectories": len(trajectories),
        "source_trajectories_without_retained_steps": sorted(
            trajectories.keys() - retained_episode_ids
        ),
        "selected_source_episodes": len(selected_ids),
        "scored_source_episodes": len(episode_results),
        "replay_errors": len(replay_errors),
        "missing_retained_source_steps": len(missing_steps),
        "replay_correct": sum(row["replay_correct"] for row in diagnostics),
        "process_update": sum(
            item["ablations"]["full"].process_update for item in episode_results
        ),
        "deterministic_grounding_complete": sum(
            row["deterministic_grounding_complete"] for row in diagnostics
        ),
        "structured_grounding_available": sum(
            row["structured_grounding_available"] for row in diagnostics
        ),
        "action_grounding_clean": sum(
            row["action_grounding_clean"] for row in diagnostics
        ),
        "target_support_complete": {
            "sql_parse": sum(row["target_support"]["sql_parse_complete"] for row in diagnostics),
            "rows": sum(row["target_support"]["rows_complete"] for row in diagnostics),
        },
        "config": dataclasses.asdict(config),
        "tool_reward_coverage_on_retained_sft_steps": serialize_tool_stats(tool_stats),
        "ablations": ablations,
        "identifiability": {
            "successful_episodes": sum(
                item["ablations"]["full"].correct for item in episode_results
            ),
            "failed_episodes": sum(
                not item["ablations"]["full"].correct for item in episode_results
            ),
            "terminal_failure_and_failure_progress": (
                "not identifiable from a positive-only SFT dataset when failed_episodes is zero"
            ),
            "interpretation": (
                "offline reward coverage diagnostic only; it does not estimate policy improvement"
            ),
        },
        "invariants": {
            "bird_set_only": args.denotation_comparison == "bird-set",
            "all_selected_episodes_scored": len(episode_results) == len(selected_ids),
            "all_retained_steps_resolved": not missing_steps,
            "all_replays_correct": all(row["replay_correct"] for row in diagnostics),
            "all_reward_identities_hold": all(
                abs(
                    result.total_reward
                    - float(result.diagnostics["expected_total_reward"])
                )
                < 1e-8
                for item in episode_results
                for result in item["ablations"].values()
            ),
        },
        "artifacts": {
            "scored_sft_steps": str(details_path.resolve()),
            "scored_source_actions": str(source_details_path.resolve()),
            "episode_diagnostics": str(diagnostics_path.resolve()),
            "replay_errors": str(errors_path.resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(summary["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
