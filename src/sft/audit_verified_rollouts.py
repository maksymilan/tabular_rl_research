#!/usr/bin/env python3
"""Audit normalized, execution-verified rollout episodes before a scale-up decision."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_issues(episode: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    steps = episode.get("steps")
    generation = episode.get("rollout_generation") or {}
    episode_id = episode.get("trajectory_id", "unknown")
    if episode.get("label_status") != "verified":
        issues.append("label_status is not verified")
    if not isinstance(steps, list) or not steps:
        return issues + ["steps must be a non-empty list"]
    if steps[-1].get("tool_call", {}).get("tool") != "answer_from_context":
        issues.append("final legal action is not answer_from_context")
    if generation.get("context_mode") != "rolling-legal-history":
        issues.append("context_mode is not rolling-legal-history")
    if generation.get("history_turns") != 4:
        issues.append("history_turns is not 4")
    if generation.get("error_actions_are_sft_targets") is not False:
        issues.append("error-actions SFT policy is not explicitly false")
    gold_sql = (episode.get("source") or {}).get("gold_sql")
    step_numbers: list[int] = []
    for index, step in enumerate(steps, start=1):
        step_id = str(step.get("step_id", ""))
        try:
            number = int(step_id.removeprefix("step_"))
        except ValueError:
            issues.append(f"step {index}: invalid step_id")
            number = 0
        step_numbers.append(number)
        if not isinstance(step.get("think"), str) or not step["think"].strip():
            issues.append(f"step {index}: empty think")
        call = step.get("tool_call")
        if not isinstance(call, dict) or not call.get("tool") or not isinstance(call.get("arguments"), dict):
            issues.append(f"step {index}: invalid tool call")
        for field in ("tool_output", "environment_state_before", "environment_state"):
            if step.get(field) is None:
                issues.append(f"step {index}: missing {field}")
        error_before = step.get("last_tool_error_before")
        if bool(step.get("feedback_recovery")) != bool(error_before):
            issues.append(f"step {index}: feedback_recovery does not match LAST TOOL ERROR")
        if step.get("feedback_recovery"):
            actual = ((error_before or {}).get("error") or {}).get("type")
            if step.get("recovered_from_error_type") != actual:
                issues.append(f"step {index}: recovered error type does not match LAST TOOL ERROR")
        if gold_sql:
            visible = json.dumps(
                [step.get("environment_state_before"), step.get("last_tool_error_before")],
                ensure_ascii=False,
            )
            if gold_sql in visible:
                issues.append(f"step {index}: gold SQL leaks into model-visible context")
    action_count = generation.get("action_count")
    error_events = generation.get("error_events") or []
    if step_numbers != sorted(step_numbers) or len(step_numbers) != len(set(step_numbers)):
        issues.append("legal step ids are not strictly increasing")
    if isinstance(action_count, int) and action_count != len(steps) + len(error_events):
        issues.append(
            f"action count {action_count} != legal steps {len(steps)} + error events {len(error_events)}"
        )
    if isinstance(action_count, int):
        missing_actions = set(range(1, action_count + 1)) - set(step_numbers)
        error_actions = {event.get("action_index") for event in error_events}
        if missing_actions != error_actions:
            issues.append("step-id gaps do not exactly match excluded error actions")
    return [f"{episode_id}: {issue}" for issue in issues]


def audit(path: Path, expected_prompt_variant: str | None) -> dict[str, Any]:
    episodes = read_jsonl(path)
    issues = [issue for episode in episodes for issue in episode_issues(episode)]
    generations = [episode.get("rollout_generation") or {} for episode in episodes]
    error_events = [event for generation in generations for event in generation.get("error_events") or []]
    outcomes = Counter(generation.get("outcome", "unknown") for generation in generations)
    variants = Counter(generation.get("rolling_prompt_variant", "full") for generation in generations)
    report = {
        "input": str(path),
        "episodes": len(episodes),
        "verified_episodes": sum(episode.get("label_status") == "verified" for episode in episodes),
        "difficulty": dict(sorted(Counter(episode.get("difficulty", "unknown") for episode in episodes).items())),
        "outcomes": dict(sorted(outcomes.items())),
        "legal_step_targets": sum(len(episode.get("steps") or []) for episode in episodes),
        "feedback_recovery_targets": sum(
            bool(step.get("feedback_recovery"))
            for episode in episodes
            for step in episode.get("steps") or []
        ),
        "error_events": dict(sorted(Counter(event.get("error_type", "unknown") for event in error_events).items())),
        "teacher_prompt_variants": dict(sorted(variants.items())),
        "structural_issues": issues,
        "structural_gate": "pass" if not issues else "fail",
    }
    if expected_prompt_variant:
        report["required_prompt_variant_for_scale"] = expected_prompt_variant
        report["prompt_variant_gate"] = (
            "pass" if set(variants) == {expected_prompt_variant} else "fail"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--required-prompt-variant", choices=["full", "compact"], default=None)
    args = parser.parse_args()
    report = audit(args.input, args.required_prompt_variant)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["structural_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
