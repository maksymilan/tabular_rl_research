#!/usr/bin/env python3
"""Audit normalized, execution-verified rollout episodes before a scale-up decision."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training_result_quality import (  # noqa: E402
    EMPTY_RESULT_POLICY_VERSION,
    empty_result_target_reason,
    trajectory_has_empty_terminal_evidence,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_issues(episode: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    steps = episode.get("steps")
    generation = episode.get("rollout_generation") or {}
    native_bundle = episode.get("tool_scheme") == "native-tool-bundle"
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
    enforce_empty_result_policy = (
        generation.get("empty_result_policy") == EMPTY_RESULT_POLICY_VERSION
    )
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
        if not isinstance(step.get("think"), str) or (
            not native_bundle and not step["think"].strip()
        ):
            issues.append(f"step {index}: empty think")
        if native_bundle:
            if not isinstance(step.get("model_turn_index"), int):
                issues.append(f"step {index}: missing model_turn_index")
            if not isinstance(step.get("native_tool_call_id"), str):
                issues.append(f"step {index}: missing native_tool_call_id")
        call = step.get("tool_call")
        if not isinstance(call, dict) or not call.get("tool") or not isinstance(call.get("arguments"), dict):
            issues.append(f"step {index}: invalid tool call")
        for field in ("tool_output", "environment_state_before", "environment_state"):
            if step.get(field) is None:
                issues.append(f"step {index}: missing {field}")
        if (
            enforce_empty_result_policy
            and empty_result_target_reason(step) is not None
            and step.get("sft_target_eligible", True) is not False
        ):
            issues.append(f"step {index}: empty result remains SFT-target eligible")
        error_before = step.get("last_tool_error_before")
        if not native_bundle:
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
    if enforce_empty_result_policy and trajectory_has_empty_terminal_evidence(steps):
        issues.append("terminal evidence table is empty and cannot enter SFT")
    action_count = generation.get("action_count")
    primitive_count = generation.get("primitive_action_count")
    error_events = generation.get("error_events") or []
    # A native-bundle model turn may fail before it contains any callable primitive (for
    # example ``missing_tool_calls``).  Such a provider-turn error spends the model-turn budget
    # and remains auditable, but it has no primitive action index, creates no step-id gap, and is
    # deliberately absent from ``primitive_action_count``.  Prevalidation/execution failures do
    # carry an integer action index and are the only native errors counted at primitive granularity.
    primitive_error_events = (
        [event for event in error_events if isinstance(event.get("action_index"), int)]
        if native_bundle
        else error_events
    )
    if step_numbers != sorted(step_numbers) or len(step_numbers) != len(set(step_numbers)):
        issues.append("legal step ids are not strictly increasing")
    audited_count = primitive_count if native_bundle else action_count
    if isinstance(audited_count, int) and audited_count != len(steps) + len(primitive_error_events):
        issues.append(
            f"primitive count {audited_count} != legal steps {len(steps)} + "
            f"primitive error events {len(primitive_error_events)}"
        )
    if isinstance(audited_count, int):
        missing_actions = set(range(1, audited_count + 1)) - set(step_numbers)
        error_actions = {event.get("action_index") for event in primitive_error_events}
        if missing_actions != error_actions:
            issues.append("step-id gaps do not exactly match excluded error actions")
    if native_bundle:
        history = episode.get("provider_native_history")
        if not isinstance(history, list) or not history:
            issues.append("native bundle trajectory has no provider-native history")
        else:
            turn_indices = []
            for item_index, item in enumerate(history, start=1):
                turn_indices.append(item.get("model_turn_index"))
                assistant = item.get("assistant") or {}
                calls = assistant.get("tool_calls")
                tool_messages = item.get("tool_messages")
                if not isinstance(calls, list) or not isinstance(tool_messages, list):
                    issues.append(f"native history {item_index}: invalid calls/results")
                    continue
                call_ids = [call.get("id") for call in calls if isinstance(call, dict)]
                result_ids = [
                    message.get("tool_call_id")
                    for message in tool_messages
                    if isinstance(message, dict)
                ]
                if len(call_ids) != len(calls) or call_ids != result_ids:
                    issues.append(
                        f"native history {item_index}: call/result ids or order differ"
                    )
            if (
                any(not isinstance(index, int) for index in turn_indices)
                or turn_indices != sorted(turn_indices)
                or len(turn_indices) != len(set(turn_indices))
            ):
                issues.append("native history model_turn_index values are not strictly increasing")
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
        "tool_schemes": dict(sorted(Counter(episode.get("tool_scheme", "unknown") for episode in episodes).items())),
        "difficulty": dict(sorted(Counter(episode.get("difficulty", "unknown") for episode in episodes).items())),
        "outcomes": dict(sorted(outcomes.items())),
        "legal_step_targets": sum(len(episode.get("steps") or []) for episode in episodes),
        "sft_eligible_step_targets_after_empty_result_filter": sum(
            step.get("sft_target_eligible", True) is not False
            and empty_result_target_reason(step) is None
            for episode in episodes
            for step in episode.get("steps") or []
        ),
        "empty_result_context_only_steps": sum(
            empty_result_target_reason(step) is not None
            for episode in episodes
            for step in episode.get("steps") or []
        ),
        "empty_terminal_episodes": sum(
            trajectory_has_empty_terminal_evidence(episode.get("steps") or [])
            for episode in episodes
        ),
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
