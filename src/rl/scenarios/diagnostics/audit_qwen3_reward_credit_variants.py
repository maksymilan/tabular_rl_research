#!/usr/bin/env python3
"""Compare Qwen3 rollout reward credit variants without generating new data.

The diagnostic is deliberately limited to fields already present in a frozen
policy-episode pool.  It compares binary correctness, the current four-level
scalar reward, and a decomposed correctness-primary reward.  It does not read
gold SQL or execute a database.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.fixed_rollout_pool import deserialize_episode  # noqa: E402
from rl.frameworks.trl.transition_batch import (  # noqa: E402
    PolicyEpisode,
    build_transition_updates,
    class_conditional_routing_advantages,
    standardized_group_advantages,
)


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _has_harness_error(episode: PolicyEpisode) -> bool:
    audit_record = episode.sample.audit_record
    if audit_record.get("errors"):
        return True
    if audit_record.get("error_events"):
        return True
    for turn in episode.sample.audit_record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if isinstance(turn.get("error_event"), dict):
            return True
        if turn.get("execution_error_type") or turn.get("execution_error"):
            return True
    return False


def _terminal_overlap(episode: PolicyEpisode) -> dict[str, float] | None:
    """Measure result closeness for offline diagnosis only.

    These samples are already stored in the rollout audit.  The overlap is
    never used by a training reward; it only distinguishes an executable but
    wrong answer from a protocol/tool failure.
    """
    turns = episode.sample.audit_record.get("turns") or []
    if not turns:
        return None
    terminal = turns[-1]
    predicted = terminal.get("pred_sample")
    gold = terminal.get("gold_sample")
    if not isinstance(predicted, list) or not isinstance(gold, list):
        return None

    def row_key(row: Any) -> tuple[Any, ...]:
        if isinstance(row, list):
            return tuple(row)
        return (row,)

    predicted_set = {row_key(row) for row in predicted}
    gold_set = {row_key(row) for row in gold}
    intersection = len(predicted_set & gold_set)
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(gold_set) if gold_set else (1.0 if not predicted_set else 0.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "row_precision": precision,
        "row_recall": recall,
        "row_f1": f1,
        "exact": float(predicted_set == gold_set),
        "predicted_rows": float(len(predicted_set)),
        "gold_rows": float(len(gold_set)),
        "overlap_rows": float(intersection),
    }


def _four_level(episode: PolicyEpisode) -> float:
    correct = bool(episode.sample.correct)
    error = _has_harness_error(episode)
    if correct and not error:
        return 1.5
    if correct:
        return 1.0
    if not error:
        return -0.5
    return -1.0


def _decomposed_advantages(
    episodes: list[PolicyEpisode],
    *,
    clean_weight: float,
) -> list[float]:
    """Correctness-primary advantage with clean/error as a within-class tie-break."""

    correctness = [1.0 if bool(ep.sample.correct) else 0.0 for ep in episodes]
    primary = standardized_group_advantages(
        correctness,
        [bool(ep.sample.process_update) for ep in episodes],
    )
    clean = [1.0 if not _has_harness_error(ep) else 0.0 for ep in episodes]
    secondary = [0.0] * len(episodes)
    for is_correct in (False, True):
        indices = [
            i for i, ep in enumerate(episodes)
            if bool(ep.sample.correct) == is_correct and bool(ep.sample.process_update)
        ]
        values = [clean[i] for i in indices]
        local = standardized_group_advantages(values, [True] * len(values))
        for i, value in zip(indices, local, strict=True):
            secondary[i] = value
    return [a + clean_weight * b for a, b in zip(primary, secondary, strict=True)]


def _sign_preserving_advantages(
    episodes: list[PolicyEpisode],
    *,
    clean_weight: float,
) -> list[float]:
    """Keep denotation correctness primary while using clean/error as magnitude."""

    if not 0.0 <= clean_weight < 1.0:
        raise ValueError("sign-preserving clean weight must be in [0, 1)")
    primary = standardized_group_advantages(
        [1.0 if bool(ep.sample.correct) else 0.0 for ep in episodes],
        [bool(ep.sample.process_update) for ep in episodes],
    )
    result = []
    for ep, advantage in zip(episodes, primary, strict=True):
        if advantage == 0.0:
            result.append(0.0)
            continue
        clean = not _has_harness_error(ep)
        if bool(ep.sample.correct):
            multiplier = 1.0 + clean_weight if clean else 1.0 - clean_weight
        else:
            multiplier = 1.0 - clean_weight if clean else 1.0 + clean_weight
        result.append(float(advantage) * multiplier)
    return result


def _transition_coefficients(
    episodes: list[PolicyEpisode],
    trajectory_advantages: list[float],
    *,
    reduction: str,
) -> tuple[list[float], list[bool], list[bool]]:
    updates = build_transition_updates(
        episodes,
        reward_mode="result-only",
        train_turns="all",
    )
    by_id = {
        str(ep.sample.audit_record["trajectory_id"]): i
        for i, ep in enumerate(episodes)
    }
    original_count = len(updates)
    original_trajectories = len({u.trajectory_id for u in updates})
    coefficients = []
    correct = []
    nonzero = []
    for update in updates:
        index = by_id[update.trajectory_id]
        # Replace the update's native advantage with the candidate trajectory
        # advantage while retaining the exact token weight.
        mean_turns = original_count / original_trajectories
        if reduction == "transition_mean":
            coefficient = trajectory_advantages[index]
        else:
            field = (
                update.trajectory_turn_weight
                if reduction == "trajectory_mean"
                else update.trajectory_token_weight
            )
            coefficient = trajectory_advantages[index] * field * mean_turns
        coefficients.append(float(coefficient))
        correct.append(bool(update.trajectory_correct))
        nonzero.append(coefficient != 0.0)
    return coefficients, correct, nonzero


def _summarize_group(
    episodes: list[PolicyEpisode],
    rewards: list[float],
    *,
    reduction: str,
) -> dict[str, Any]:
    eligible = [bool(ep.sample.process_update) for ep in episodes]
    advantages = standardized_group_advantages(rewards, eligible)
    coefficients, correct, nonzero = _transition_coefficients(
        episodes, advantages, reduction=reduction
    )
    eligible_coefficients = [c for c in coefficients]
    return {
        "reward_values": sorted(set(rewards)),
        "trajectory_nonzero": sum(a != 0.0 for a in advantages),
        "trajectory_count": sum(eligible),
        "transition_count": len(coefficients),
        "nonzero_transition_fraction": (
            sum(nonzero) / len(nonzero) if nonzero else 0.0
        ),
        "absolute_coefficient_mass": sum(abs(c) for c in eligible_coefficients),
        "positive_on_correct_fraction": (
            sum(c > 0.0 and ok for c, ok in zip(coefficients, correct, strict=True))
            / max(1, sum(c > 0.0 for c in coefficients))
        ),
        "negative_on_wrong_fraction": (
            sum(c < 0.0 and not ok for c, ok in zip(coefficients, correct, strict=True))
            / max(1, sum(c < 0.0 for c in coefficients))
        ),
    }


def audit(path: Path, *, clean_weight: float, reduction: str) -> dict[str, Any]:
    grouped: dict[int, list[PolicyEpisode]] = defaultdict(list)
    for row in _rows(path):
        episode = deserialize_episode(row)
        grouped[int(episode.sample.audit_record["example_index"])].append(episode)
    if not grouped:
        raise ValueError("empty rollout pool")

    variants = {
        "binary": lambda group: [1.0 if ep.sample.correct else 0.0 for ep in group],
        "four_level": lambda group: [_four_level(ep) for ep in group],
        "decomposed": lambda group: _decomposed_advantages(
            group, clean_weight=clean_weight
        ),
        "sign_preserving": lambda group: _sign_preserving_advantages(
            group, clean_weight=clean_weight
        ),
        "class_conditional_routing": lambda group: class_conditional_routing_advantages(
            group, clean_weight=clean_weight
        ),
    }
    output: dict[str, Any] = {
        "schema_version": "qwen3-reward-credit-variant-audit-v1",
        "input": str(path),
        "tasks": len(grouped),
        "group_sizes": sorted({len(group) for group in grouped.values()}),
        "policy_reduction": reduction,
        "decomposed_clean_weight": clean_weight,
        "variants": {},
    }
    outcome_buckets: dict[str, list[PolicyEpisode]] = defaultdict(list)
    for group in grouped.values():
        for episode in group:
            outcome_buckets[
                (
                    "correct" if episode.sample.correct else "wrong"
                )
                + ("_error" if _has_harness_error(episode) else "_clean")
            ].append(episode)
    semantic_summary: dict[str, Any] = {}
    for bucket, episodes in sorted(outcome_buckets.items()):
        overlaps = [_terminal_overlap(ep) for ep in episodes]
        overlaps = [value for value in overlaps if value is not None]
        semantic_summary[bucket] = {
            "trajectories": len(episodes),
            "legal": sum(bool(ep.sample.audit_record.get("legal")) for ep in episodes),
            "executable_terminal": sum(
                bool((ep.sample.audit_record.get("result_reward") or {}).get("executable_terminal"))
                for ep in episodes
            ),
            "mean_steps": (
                sum(float(ep.sample.audit_record.get("steps") or 0) for ep in episodes)
                / len(episodes)
                if episodes
                else 0.0
            ),
            "mean_elapsed_seconds": (
                sum(float(ep.sample.audit_record.get("elapsed_seconds") or 0) for ep in episodes)
                / len(episodes)
                if episodes
                else 0.0
            ),
            "terminal_overlap_available": len(overlaps),
            "mean_row_precision": (
                sum(value["row_precision"] for value in overlaps) / len(overlaps)
                if overlaps
                else None
            ),
            "mean_row_recall": (
                sum(value["row_recall"] for value in overlaps) / len(overlaps)
                if overlaps
                else None
            ),
            "mean_row_f1": (
                sum(value["row_f1"] for value in overlaps) / len(overlaps)
                if overlaps
                else None
            ),
            "exact_terminal_matches": sum(value["exact"] for value in overlaps),
        }
    output["outcome_buckets"] = semantic_summary
    for name, reward_fn in variants.items():
        task_summaries = []
        homogeneous = 0
        wrong_positive_trajectories = 0
        correct_negative_trajectories = 0
        wrong_positive_transitions = 0
        correct_negative_transitions = 0
        total_transitions = 0
        for example_index, group in sorted(grouped.items()):
            if name in {
                "decomposed",
                "sign_preserving",
                "class_conditional_routing",
            }:
                # These functions return final advantages, so summarize them
                # directly instead of applying a second group normalization.
                advantages = reward_fn(group)
                coefficients, correct, nonzero = _transition_coefficients(
                    group, advantages, reduction=reduction
                )
                summary = {
                    "reward_values": sorted(set(advantages)),
                    "trajectory_nonzero": sum(a != 0.0 for a in advantages),
                    "trajectory_count": sum(bool(ep.sample.process_update) for ep in group),
                    "transition_count": len(coefficients),
                    "nonzero_transition_fraction": sum(nonzero) / max(1, len(nonzero)),
                    "absolute_coefficient_mass": sum(abs(c) for c in coefficients),
                    "positive_on_correct_fraction": sum(
                        c > 0.0 and ok for c, ok in zip(coefficients, correct, strict=True)
                    ) / max(1, sum(c > 0.0 for c in coefficients)),
                    "negative_on_wrong_fraction": sum(
                        c < 0.0 and not ok for c, ok in zip(coefficients, correct, strict=True)
                    ) / max(1, sum(c < 0.0 for c in coefficients)),
                }
            else:
                summary = _summarize_group(group, reward_fn(group), reduction=reduction)
            if name in {
                "decomposed",
                "sign_preserving",
                "class_conditional_routing",
            }:
                final_advantages = reward_fn(group)
            else:
                final_advantages = standardized_group_advantages(
                    reward_fn(group),
                    [bool(ep.sample.process_update) for ep in group],
                )
            wrong_positive_trajectories += sum(
                advantage > 0.0 and not ep.sample.correct
                for ep, advantage in zip(group, final_advantages, strict=True)
            )
            correct_negative_trajectories += sum(
                advantage < 0.0 and ep.sample.correct
                for ep, advantage in zip(group, final_advantages, strict=True)
            )
            transition_count = summary["transition_count"]
            total_transitions += transition_count
            wrong_positive_transitions += sum(
                transition_count_for_episode * (advantage > 0.0 and not ep.sample.correct)
                for ep, advantage, transition_count_for_episode in zip(
                    group,
                    final_advantages,
                    [
                        sum(1 for update in build_transition_updates(
                            [ep], reward_mode="result-only", train_turns="all"
                        ))
                        for ep in group
                    ],
                    strict=True,
                )
            )
            correct_negative_transitions += sum(
                transition_count_for_episode * (advantage < 0.0 and ep.sample.correct)
                for ep, advantage, transition_count_for_episode in zip(
                    group,
                    final_advantages,
                    [
                        sum(1 for update in build_transition_updates(
                            [ep], reward_mode="result-only", train_turns="all"
                        ))
                        for ep in group
                    ],
                    strict=True,
                )
            )
            summary["example_index"] = example_index
            homogeneous += int(summary["trajectory_nonzero"] == 0)
            task_summaries.append(summary)
        output["variants"][name] = {
            "homogeneous_groups": homogeneous,
            "mixed_groups": len(grouped) - homogeneous,
            "mean_nonzero_transition_fraction": sum(
                s["nonzero_transition_fraction"] for s in task_summaries
            ) / len(task_summaries),
            "total_absolute_coefficient_mass": sum(
                s["absolute_coefficient_mass"] for s in task_summaries
            ),
            "mean_positive_on_correct_fraction": sum(
                s["positive_on_correct_fraction"] for s in task_summaries
            ) / len(task_summaries),
            "mean_negative_on_wrong_fraction": sum(
                s["negative_on_wrong_fraction"] for s in task_summaries
            ) / len(task_summaries),
            "wrong_positive_trajectories": wrong_positive_trajectories,
            "correct_negative_trajectories": correct_negative_trajectories,
            "wrong_positive_transition_fraction": wrong_positive_transitions / max(1, total_transitions),
            "correct_negative_transition_fraction": correct_negative_transitions / max(1, total_transitions),
            "groups": task_summaries,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument("--clean-weight", type=float, default=0.25)
    parser.add_argument(
        "--policy-reduction",
        choices=("transition_mean", "trajectory_mean", "trajectory_token_mean"),
        default="trajectory_token_mean",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.pool,
        clean_weight=args.clean_weight,
        reduction=args.policy_reduction,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
