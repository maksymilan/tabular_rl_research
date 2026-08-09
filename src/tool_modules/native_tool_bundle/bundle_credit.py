#!/usr/bin/env python3
"""Fact-only bundle/call statistics for later RL credit experiments.

These labels deliberately do not decide whether an observational call was semantically useful.
They record only causal facts visible in the trajectory: execution status, later references to a
produced handle/step, preservation of structured error feedback, and the next-turn recovery shape.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PERCEPTION_TOOLS = frozenset({"describe_table", "inspect_column", "read_subtable"})
def _strings(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_strings(item))
    return found


def _error_feedback_call_ids(record: dict[str, Any]) -> set[str]:
    preserved: set[str] = set()
    for item in record.get("provider_native_history") or []:
        for message in item.get("tool_messages") or []:
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if not isinstance(call_id, str) or not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                preserved.add(call_id)
    return preserved


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    turns = [
        turn
        for turn in (record.get("turns") or [])
        if isinstance((turn.get("parsed") or {}).get("calls"), list)
        and (turn.get("parsed") or {}).get("calls")
    ]
    feedback_ids = _error_feedback_call_ids(record)
    counters: Counter[str] = Counter()
    bundle_hist: Counter[str] = Counter()
    status_hist: Counter[str] = Counter()
    tool_hist: Counter[str] = Counter()
    utility_hist: Counter[str] = Counter()
    annotations: list[dict[str, Any]] = []

    later_argument_strings: list[list[set[str]]] = []
    for turn in turns:
        calls = (turn.get("parsed") or {}).get("calls") or []
        later_argument_strings.append([
            _strings(call.get("arguments")) for call in calls
        ])

    for turn_position, turn in enumerate(turns):
        calls = (turn.get("parsed") or {}).get("calls") or []
        results = turn.get("native_bundle_results") or []
        counters["turns_with_native_calls"] += 1
        bundle_hist[str(len(calls))] += 1
        if len(calls) == 1:
            counters["single_call_turns"] += 1
        else:
            counters["multi_call_turns"] += 1
        if len(calls) > 3:
            counters["soft_policy_over_three_turns"] += 1
        if any(call.get("tool") == "answer_from_context" for call in calls):
            counters["terminal_turns"] += 1
            if len(calls) != 1:
                counters["terminal_mixed_turns"] += 1

        result_by_id = {
            result.get("call_id"): result
            for result in results
            if isinstance(result, dict)
        }
        turn_had_error = False
        for call in calls:
            call_id = str(call.get("id") or "")
            tool = str(call.get("tool") or "")
            arguments = call.get("arguments") or {}
            result = result_by_id.get(call_id) or {}
            status = str(result.get("status") or "missing")
            status_hist[status] += 1
            tool_hist[tool] += 1
            counters["primitive_calls"] += 1

            annotation: dict[str, Any] = {
                "model_turn_index": turn.get("turn_index"),
                "native_tool_call_id": call_id,
                "tool": tool,
                "status": status,
            }
            if status == "error":
                turn_had_error = True
                counters["error_calls"] += 1
                preserved = call_id in feedback_ids
                counters[
                    "error_feedback_preserved" if preserved else "error_feedback_missing"
                ] += 1
                annotation["error_feedback_preserved"] = preserved
                annotation["utility_class"] = "error_feedback"
            elif status == "blocked":
                annotation["utility_class"] = "blocked_after_nonrecoverable_error"
            elif tool == "answer_from_context":
                correct = bool((result.get("output") or {}).get("correct"))
                annotation["terminal_correct"] = correct
                annotation["utility_class"] = (
                    "terminal_correct" if correct else "terminal_incorrect"
                )
            elif status == "success" and result.get("table"):
                identifiers = {
                    str(result["table"]),
                    str(result.get("step_id") or ""),
                } - {""}
                later_refs = 0
                for later_turn in later_argument_strings[turn_position + 1:]:
                    later_refs += sum(
                        bool(identifiers & argument_strings)
                        for argument_strings in later_turn
                    )
                annotation["produced_table"] = result["table"]
                annotation["later_reference_count"] = later_refs
                counters["created_handle_calls"] += 1
                if later_refs:
                    counters["created_handle_referenced_later"] += 1
                    annotation["utility_class"] = "downstream_reference"
                else:
                    counters["created_handle_unreferenced"] += 1
                    annotation["utility_class"] = "unreferenced_derived_result"
            elif status == "success" and tool in PERCEPTION_TOOLS:
                counters["perception_credit_unresolved"] += 1
                annotation["utility_class"] = "observational_credit_unresolved"
            elif status == "success" and tool == "plan":
                counters["control_credit_unresolved"] += 1
                annotation["utility_class"] = "control_credit_unresolved"
            else:
                annotation["utility_class"] = "execution_credit_unresolved"
            utility_hist[annotation["utility_class"]] += 1
            annotations.append(annotation)

        if turn_had_error:
            counters["error_turns"] += 1
            next_turn = turns[turn_position + 1] if turn_position + 1 < len(turns) else None
            if next_turn is not None:
                counters["error_turns_with_followup"] += 1
                next_results = next_turn.get("native_bundle_results") or []
                if any(result.get("status") == "success" for result in next_results):
                    counters["error_turns_with_successful_followup"] += 1
                current_error_calls = [
                    call
                    for call in calls
                    if (result_by_id.get(str(call.get("id") or "")) or {}).get("status")
                    == "error"
                ]
                next_calls = (next_turn.get("parsed") or {}).get("calls") or []
                corrected_same_tool = any(
                    prior.get("tool") == later.get("tool")
                    and prior.get("arguments") != later.get("arguments")
                    for prior in current_error_calls
                    for later in next_calls
                )
                counters[
                    "corrected_same_tool_next_turn"
                    if corrected_same_tool
                    else "changed_strategy_next_turn"
                ] += 1

    error_calls = counters["error_calls"]
    created_calls = counters["created_handle_calls"]
    return {
        "schema_version": "native-bundle-credit-stats-v1",
        "statistics_only_not_reward": True,
        "error_target_policy": (
            "preserve_error_turn_and_feedback_as_history; target_later_corrected_bundle"
        ),
        "counts": dict(sorted(counters.items())),
        "bundle_size_histogram": dict(sorted(bundle_hist.items(), key=lambda item: int(item[0]))),
        "call_status_histogram": dict(sorted(status_hist.items())),
        "tool_call_histogram": dict(sorted(tool_hist.items())),
        "utility_class_histogram": dict(sorted(utility_hist.items())),
        "error_feedback_preservation_rate": (
            counters["error_feedback_preserved"] / error_calls if error_calls else None
        ),
        "created_handle_later_reference_rate": (
            counters["created_handle_referenced_later"] / created_calls
            if created_calls
            else None
        ),
        "call_annotations": annotations,
    }


def aggregate_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summaries = list(summaries)
    counters: Counter[str] = Counter()
    bundle_hist: Counter[str] = Counter()
    status_hist: Counter[str] = Counter()
    tool_hist: Counter[str] = Counter()
    utility_hist: Counter[str] = Counter()
    for summary in summaries:
        counters.update(summary.get("counts") or {})
        bundle_hist.update(summary.get("bundle_size_histogram") or {})
        status_hist.update(summary.get("call_status_histogram") or {})
        tool_hist.update(summary.get("tool_call_histogram") or {})
        utility_hist.update(summary.get("utility_class_histogram") or {})
    error_calls = counters["error_calls"]
    created_calls = counters["created_handle_calls"]
    return {
        "schema_version": "native-bundle-credit-stats-corpus-v1",
        "statistics_only_not_reward": True,
        "records": len(summaries),
        "counts": dict(sorted(counters.items())),
        "bundle_size_histogram": dict(sorted(bundle_hist.items(), key=lambda item: int(item[0]))),
        "call_status_histogram": dict(sorted(status_hist.items())),
        "tool_call_histogram": dict(sorted(tool_hist.items())),
        "utility_class_histogram": dict(sorted(utility_hist.items())),
        "error_feedback_preservation_rate": (
            counters["error_feedback_preserved"] / error_calls if error_calls else None
        ),
        "created_handle_later_reference_rate": (
            counters["created_handle_referenced_later"] / created_calls
            if created_calls
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = aggregate_summaries(analyze_record(record) for record in records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
