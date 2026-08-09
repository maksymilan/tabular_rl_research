#!/usr/bin/env python3
"""Scheme-owned causal/provider audit for native-tool-bundle records."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DERIVED_HANDLE_RE = re.compile(
    r"^(?:project|filter|join|group|top|setop|derive)_\d+$"
)


def _collect_derived_refs(value: Any, refs: set[str]) -> None:
    if isinstance(value, str):
        if DERIVED_HANDLE_RE.fullmatch(value):
            refs.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_derived_refs(item, refs)
    elif isinstance(value, list):
        for item in value:
            _collect_derived_refs(item, refs)


def _canonical_arguments(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def audit_records(records: list[dict]) -> dict[str, Any]:
    issues: list[str] = []
    counts: Counter = Counter()
    original_assistants: dict[tuple[str, str], dict] = {}
    for record in records:
        trajectory_id = str(record.get("trajectory_id", "unknown"))
        no_plan_protocol = record.get("protocol_version") == "version54"
        if record.get("tool_scheme") != "native-tool-bundle":
            issues.append(f"{trajectory_id}: wrong tool_scheme")
        preserved_error_call_ids: set[str] = set()
        for history_item in record.get("provider_native_history") or []:
            for tool_message in history_item.get("tool_messages") or []:
                call_id = tool_message.get("tool_call_id")
                content = tool_message.get("content")
                if not isinstance(call_id, str) or not isinstance(content, str):
                    continue
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                    preserved_error_call_ids.add(call_id)
        gold_sql = record.get("gold_sql")
        for turn_index, turn in enumerate(record.get("turns") or []):
            counts["model_turns"] += 1
            visible = json.dumps(turn.get("model_input") or [], ensure_ascii=False)
            if isinstance(gold_sql, str) and gold_sql and gold_sql in visible:
                issues.append(f"{trajectory_id}/turn{turn_index}: gold SQL in model input")

            # Every prior native assistant in request history must preserve its original reasoning.
            for message in turn.get("model_input") or []:
                if message.get("role") != "assistant":
                    continue
                calls = message.get("tool_calls") or []
                if not calls:
                    issues.append(f"{trajectory_id}/turn{turn_index}: history assistant has no calls")
                    continue
                known = original_assistants.get(
                    (trajectory_id, str(calls[0].get("id")))
                )
                if no_plan_protocol and any(
                    (call.get("function") or {}).get("name") == "plan"
                    for call in calls
                ):
                    issues.append(
                        f"{trajectory_id}/turn{turn_index}: version54 history contains plan"
                    )
                if known is not None and message.get("reasoning_content") != known.get(
                    "reasoning_content"
                ):
                    issues.append(
                        f"{trajectory_id}/turn{turn_index}: reasoning history changed"
                    )

            provider_calls = turn.get("provider_native_tool_call") or []
            parsed_calls = (turn.get("parsed") or {}).get("calls") or []
            results = turn.get("native_bundle_results") or []
            if not provider_calls:
                counts["provider_turns_without_calls"] += 1
                continue
            if no_plan_protocol and any(
                (call.get("function") or {}).get("name") == "plan"
                for call in provider_calls
            ):
                issues.append(
                    f"{trajectory_id}/turn{turn_index}: version54 provider call contains plan"
                )
            counts["provider_calls"] += len(provider_calls)
            counts[f"bundle_size_{len(provider_calls)}"] += 1
            if len(provider_calls) > 1:
                counts["multi_call_turns"] += 1
            if isinstance(turn.get("raw_model_output"), str) and turn[
                "raw_model_output"
            ].strip():
                counts["nonempty_assistant_content_turns"] += 1
            call_ids = [call.get("id") for call in provider_calls]
            if len(call_ids) != len(set(call_ids)):
                issues.append(f"{trajectory_id}/turn{turn_index}: duplicate call ids")
            if len(provider_calls) != len(parsed_calls) or len(provider_calls) != len(results):
                issues.append(f"{trajectory_id}/turn{turn_index}: call/lowering/result count differs")
                continue
            original_assistants[(trajectory_id, str(call_ids[0]))] = {
                "reasoning_content": turn.get("provider_reasoning_content")
            }
            terminal_count = sum(
                call.get("function", {}).get("name") == "answer_from_context"
                for call in provider_calls
            )
            if terminal_count and len(provider_calls) != 1:
                issues.append(f"{trajectory_id}/turn{turn_index}: terminal mixed with work")

            bundle_initial = (results[0].get("environment_state_before") or {}) if results else {}
            initial_tables = set((bundle_initial.get("tables") or {}).keys())
            for call_position, (provider_call, parsed_call, result) in enumerate(
                zip(provider_calls, parsed_calls, results, strict=True)
            ):
                function = provider_call.get("function") or {}
                raw_arguments = _canonical_arguments(function.get("arguments"))
                if (
                    provider_call.get("id") != parsed_call.get("id")
                    or provider_call.get("id") != result.get("call_id")
                    or function.get("name") != parsed_call.get("tool")
                    or function.get("name") != result.get("tool")
                    or raw_arguments != parsed_call.get("arguments")
                    or raw_arguments != result.get("arguments")
                ):
                    issues.append(
                        f"{trajectory_id}/turn{turn_index}/call{call_position}: lowering changed"
                    )
                refs: set[str] = set()
                _collect_derived_refs(parsed_call.get("arguments"), refs)
                unseen_refs = refs - initial_tables
                if unseen_refs and result.get("status") == "success":
                    issues.append(
                        f"{trajectory_id}/turn{turn_index}/call{call_position}: "
                        "same-bundle unseen handle executed"
                    )
                if result.get("status") == "error":
                    counts["error_results"] += 1
                    if str(result.get("call_id") or "") in preserved_error_call_ids:
                        counts["preserved_error_feedback_results"] += 1
                    else:
                        issues.append(
                            f"{trajectory_id}/turn{turn_index}/call{call_position}: "
                            "structured error feedback missing from provider-native history"
                        )
                    if result.get("environment_state_before") != result.get("environment_state"):
                        issues.append(
                            f"{trajectory_id}/turn{turn_index}/call{call_position}: "
                            "rejected call changed state"
                        )

            # Validate the request transcript's one-result-per-call grouping.
            request_messages = turn.get("model_input") or []
            for message_index, message in enumerate(request_messages):
                if message.get("role") != "assistant":
                    continue
                expected = [
                    call.get("id") for call in (message.get("tool_calls") or [])
                ]
                actual = []
                cursor = message_index + 1
                while cursor < len(request_messages) and request_messages[cursor].get(
                    "role"
                ) == "tool":
                    actual.append(request_messages[cursor].get("tool_call_id"))
                    cursor += 1
                if expected != actual:
                    issues.append(
                        f"{trajectory_id}/turn{turn_index}: history call/result order differs"
                    )
    return {
        "schema_version": "native-tool-bundle-audit-v1",
        "records": len(records),
        "counts": dict(sorted(counts.items())),
        "issues": issues,
        "gate": "pass" if not issues else "fail",
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
    report = audit_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
