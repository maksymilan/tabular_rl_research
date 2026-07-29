#!/usr/bin/env python3
"""Replay verified trajectories, assign process rewards, and write an auditable report."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from process_credit import ProcessRewardConfig, score_verified_trajectory  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--expected-label",
        choices=("success", "failure", "mixed"),
        default="success",
        help="expected replay label for report exit invariants",
    )
    parser.add_argument("--penalty-cap", type=float, default=0.8)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument(
        "--denotation-comparison",
        choices=("bird-set",),
        default="bird-set",
    )
    parser.add_argument(
        "--grounding-audit-approved",
        action="store_true",
        help="mark a separately completed manual precision audit; structural coverage alone is insufficient",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_values = {}
    if args.config_json:
        config_values = {
            key: value
            for key, value in json.loads(args.config_json.read_text(encoding="utf-8")).items()
            if not key.startswith("_")
        }
    config_values.setdefault("penalty_cap", args.penalty_cap)
    config = ProcessRewardConfig(**config_values)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "scored_trajectories.jsonl"

    results = []
    with scored_path.open("w", encoding="utf-8") as sink:
        for index, trajectory in enumerate(rows, 1):
            result = score_verified_trajectory(
                trajectory,
                config,
                denotation_comparison=args.denotation_comparison,
            )
            results.append(result)
            sink.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            if not args.quiet:
                print(
                    f"[{index}/{len(rows)}] {result.trajectory_id} reward={result.total_reward:.4f} "
                    f"slice={len(result.back_slice_step_ids)} grounding={result.grounding_method}"
                )

    grounding = collections.Counter(result.grounding_method for result in results)
    tool_credit = collections.Counter()
    tool_penalty = collections.Counter()
    outcome_penalty_by_tool = collections.Counter()
    local_penalty_by_tool = collections.Counter()
    applied_positive_by_tool = collections.Counter()
    applied_negative_by_tool = collections.Counter()
    feature_hits = collections.Counter()
    feature_mass = collections.Counter()
    positive_steps = negative_steps = zero_steps = 0
    max_positive_shares = []
    max_negative_shares = []
    negative_steps_per_episode = []
    for result in results:
        max_positive_shares.append(max((step.c_positive for step in result.steps), default=0.0))
        max_negative_shares.append(max((step.c_negative for step in result.steps), default=0.0))
        negative_steps_per_episode.append(sum(step.reward < 0 for step in result.steps))
        for step in result.steps:
            tool = step.tool or step.features.get("error_type") or "unknown"
            tool_credit[tool] += step.c_positive
            tool_penalty[tool] += step.c_negative
            outcome_penalty_by_tool[tool] += step.p_outcome
            local_penalty_by_tool[tool] += step.p_local
            applied_positive_by_tool[tool] += max(0.0, step.reward)
            applied_negative_by_tool[tool] += min(0.0, step.reward)
            positive_steps += step.reward > 0
            negative_steps += step.reward < 0
            zero_steps += step.reward == 0
            for name in (
                "back_slice",
                "attempted_back_slice",
                "new_used_evidence",
                "search_reduction",
                "feedback_response",
                "target_table_delta",
                "target_column_delta",
                "target_row_delta",
                "target_potential_delta",
                "feedback_error_before",
                "feedback_empty_before",
                "action_changed_after_empty",
                "empty_result",
                "tool_error",
                "adjacent_repeat",
                "repeat_without_feedback",
                "legal_no_state_change",
                "ignored_feedback",
                "answer_format",
                "unsupported_guess",
                "empty_result_penalty",
                "verified_negative_evidence",
                "failure_responsibility",
            ):
                feature_hits[name] += float(step.features.get(name, 0)) > 0
                feature_mass[name] += float(step.features.get(name, 0))

    rewards = [result.total_reward for result in results]
    structured_count = sum(
        result.diagnostics.get("structured_grounding_available", False) for result in results
    )
    structured_rate = structured_count / len(results) if results else 0.0
    final_value_complete = sum(
        result.diagnostics.get("final_value_grounding_complete", False) for result in results
    )
    action_literal_complete = sum(
        result.diagnostics.get("action_literal_grounding_complete", False) for result in results
    )
    deterministic_complete = sum(
        result.diagnostics.get("deterministic_grounding_complete", False) for result in results
    )
    ungrounded_nonfallback = sum(
        not result.diagnostics.get("structured_grounding_available", False)
        and not result.fallback_terminal_credit
        for result in results
    )
    structural_grounding_gate = bool(results) and structured_rate >= 0.9 and not ungrounded_nonfallback
    if args.expected_label == "success":
        outcomes_match_expectation = all(result.correct for result in results)
    elif args.expected_label == "failure":
        outcomes_match_expectation = all(not result.correct for result in results)
    else:
        outcomes_match_expectation = True
    summary = {
        "input": str(args.input.resolve()),
        "expected_label": args.expected_label,
        "scored_output": str(scored_path.resolve()),
        "config": config.__dict__,
        "denotation_comparison": args.denotation_comparison,
        "trajectories": len(results),
        "replay_correct": sum(result.correct for result in results),
        "grounding_methods": dict(sorted(grounding.items())),
        "structured_grounding_rate": structured_rate,
        "final_value_grounding_complete": final_value_complete,
        "action_literal_grounding_complete": action_literal_complete,
        "deterministic_grounding_complete": deterministic_complete,
        "deterministic_grounding_complete_rate": (
            deterministic_complete / len(results) if results else 0.0
        ),
        "fallback_terminal_credit": sum(result.fallback_terminal_credit for result in results),
        "process_update": sum(result.process_update for result in results),
        "ungrounded_nonfallback": ungrounded_nonfallback,
        "reward": {
            "min": min(rewards) if rewards else 0.0,
            "p50": statistics.median(rewards) if rewards else 0.0,
            "p90": percentile(rewards, 0.9),
            "max": max(rewards) if rewards else 0.0,
            "mean": statistics.mean(rewards) if rewards else 0.0,
        },
        "step_reward_signs": {
            "positive": positive_steps,
            "negative": negative_steps,
            "zero": zero_steps,
        },
        "feature_hits": dict(sorted(feature_hits.items())),
        "feature_mass": {key: round(value, 6) for key, value in sorted(feature_mass.items())},
        "max_positive_credit_share": {
            "p50": statistics.median(max_positive_shares) if max_positive_shares else 0.0,
            "p90": percentile(max_positive_shares, 0.9),
            "max": max(max_positive_shares) if max_positive_shares else 0.0,
        },
        "max_negative_credit_share": {
            "p50": statistics.median(max_negative_shares) if max_negative_shares else 0.0,
            "p90": percentile(max_negative_shares, 0.9),
            "max": max(max_negative_shares) if max_negative_shares else 0.0,
        },
        "negative_steps_per_episode": {
            "p50": statistics.median(negative_steps_per_episode) if negative_steps_per_episode else 0.0,
            "p90": percentile(negative_steps_per_episode, 0.9),
            "max": max(negative_steps_per_episode) if negative_steps_per_episode else 0,
            "multi_step_episodes": sum(value > 1 for value in negative_steps_per_episode),
        },
        "positive_credit_by_tool": {
            key: round(value, 6) for key, value in sorted(tool_credit.items())
        },
        "negative_credit_by_tool": {
            key: round(value, 6) for key, value in sorted(tool_penalty.items())
        },
        "raw_outcome_penalty_by_tool": {
            key: round(value, 6) for key, value in sorted(outcome_penalty_by_tool.items())
        },
        "raw_local_penalty_by_tool": {
            key: round(value, 6) for key, value in sorted(local_penalty_by_tool.items())
        },
        "applied_positive_reward_by_tool": {
            key: round(value, 6) for key, value in sorted(applied_positive_by_tool.items())
        },
        "applied_negative_reward_by_tool": {
            key: round(value, 6) for key, value in sorted(applied_negative_by_tool.items())
        },
        "invariants": {
            "all_replay_correct": all(result.correct for result in results),
            "outcomes_match_expectation": outcomes_match_expectation,
            "all_totals_match_reward_identity": all(
                abs(
                    result.total_reward
                    - float(result.diagnostics.get("expected_total_reward"))
                ) < 1e-8
                for result in results
            ),
            "all_correct_totals_positive": all(
                not result.correct or result.total_reward > 0 for result in results
            ),
            "structural_grounding_gate_passed": structural_grounding_gate,
            "deterministic_grounding_gate_passed": bool(results) and deterministic_complete == len(results),
            "grounding_precision_audit_approved": args.grounding_audit_approved,
            "process_reward_ready": (
                structural_grounding_gate
                and deterministic_complete == len(results)
                and args.grounding_audit_approved
            ),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    required = (
        "outcomes_match_expectation",
        "all_totals_match_reward_identity",
        "all_correct_totals_positive",
    )
    return 0 if all(summary["invariants"][key] for key in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
