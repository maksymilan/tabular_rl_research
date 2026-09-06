#!/usr/bin/env python3
"""Offline size and next-action visibility audit for version46-version48 context profiles."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for relative in ("src/harness", "src/sft", "src/eval"):
    sys.path.insert(0, str(ROOT / relative))

from atomic_version48 import INTERPRET_BEFORE_ACT_SUFFIX  # noqa: E402
from protocol import (  # noqa: E402
    RESIDENT_STATE_PROFILE_HANDLE_CARDS,
    RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
    RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS,
    environment_state_message,
    tool_output_message,
)


STATE_MARKER = "CURRENT ENVIRONMENT STATE"
ERROR_MARKER = "\n\nLAST TOOL ERROR\n"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def numeric_summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def extract_state(content: str) -> tuple[dict | None, dict | None, int | None]:
    marker = content.find(STATE_MARKER)
    if marker < 0:
        return None, None, None
    json_start = content.find("\n", marker)
    if json_start < 0:
        return None, None, None
    json_start += 1
    decoder = json.JSONDecoder()
    try:
        state, consumed = decoder.raw_decode(content[json_start:])
    except json.JSONDecodeError:
        return None, None, None
    remainder = content[json_start + consumed:]
    last_error = None
    error_index = remainder.find(ERROR_MARKER)
    if error_index >= 0:
        try:
            last_error = json.loads(remainder[error_index + len(ERROR_MARKER):])
        except json.JSONDecodeError:
            last_error = None
    return state, last_error, marker


def replace_last_state(
    messages: list[dict],
    *,
    profile: str,
    latest_observation_full: bool,
    previous_success: dict | None,
) -> list[dict]:
    rendered = deepcopy(messages)
    user_index = next(
        (
            index
            for index in range(len(rendered) - 1, -1, -1)
            if rendered[index].get("role") == "user"
            and STATE_MARKER in str(rendered[index].get("content", ""))
        ),
        None,
    )
    if user_index is None:
        return rendered
    content = str(rendered[user_index]["content"])
    state, last_error, marker = extract_state(content)
    if state is None or marker is None:
        return rendered
    prefix = content[:marker].rstrip()
    if latest_observation_full and previous_success is not None:
        prefix_step = None
        if prefix.startswith("{"):
            try:
                prefix_step = json.JSONDecoder().raw_decode(prefix)[0].get("step_id")
            except (json.JSONDecodeError, AttributeError):
                prefix_step = None
        step_id = prefix_step or previous_success.get("step_id") or "previous_success"
        output = previous_success.get("tool_output")
        if isinstance(output, dict):
            prefix = tool_output_message(step_id, output)
    active_relation_refs = None
    if profile == RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE:
        known_tables = set((state.get("tables") or {}))
        active_relation_refs = set()

        def collect(value: Any) -> None:
            if isinstance(value, str):
                if value in known_tables:
                    active_relation_refs.add(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        for message in rendered:
            message_content = str(message.get("content", ""))
            payload = None
            if message.get("role") == "assistant":
                action_text = (
                    message_content.rsplit("</think>", 1)[1].strip()
                    if "</think>" in message_content
                    else message_content.strip()
                )
                try:
                    payload = json.loads(action_text)
                except json.JSONDecodeError:
                    payload = None
            elif message.get("role") == "user" and message_content.startswith("{"):
                try:
                    payload = json.JSONDecoder().raw_decode(message_content)[0]
                except json.JSONDecodeError:
                    payload = None
            collect(payload)
        collect(last_error)
    state_message = environment_state_message(
        state,
        last_error,
        resident_state_profile=profile,
        active_relation_refs=active_relation_refs,
    )
    rendered[user_index]["content"] = (
        f"{prefix}\n\n{state_message}" if prefix else state_message
    )
    return rendered


def input_chars(messages: list[dict]) -> int:
    return sum(len(str(item.get("content", ""))) for item in messages)


def visible_text(messages: list[dict]) -> str:
    return "\n".join(str(item.get("content", "")) for item in messages)


def leaves(value: Any, path: str = "arguments") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(leaves(item, f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(leaves(item, f"{path}[{index}]"))
        return out
    if isinstance(value, (str, int, float, bool)) and value not in ("", None):
        return [(path, value)]
    return []


def serialized_leaf(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def is_grounding_value_path(path: str) -> bool:
    """Select authored literal operands, excluding reconstructable table/column/operator names."""
    return (
        path.endswith(".value")
        or ".values[" in path
        or ".category_values[" in path
        or ".operands[" in path and path.endswith(".value")
    )


def audit(records: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    lost_events = {"version47": [], "version49": []}
    for record in records:
        previous_success = None
        for turn_index, turn in enumerate(record.get("turns") or []):
            original = turn.get("model_input")
            if not isinstance(original, list):
                continue
            v46 = replace_last_state(
                original,
                profile=RESIDENT_STATE_PROFILE_HANDLE_CARDS,
                latest_observation_full=False,
                previous_success=previous_success,
            )
            v47 = replace_last_state(
                original,
                profile=RESIDENT_STATE_PROFILE_HANDLE_CARDS_ARCHIVED_READS,
                latest_observation_full=True,
                previous_success=previous_success,
            )
            v48 = deepcopy(v47)
            if v48 and v48[0].get("role") == "system":
                v48[0]["content"] = str(v48[0].get("content", "")) + INTERPRET_BEFORE_ACT_SUFFIX
            v49 = replace_last_state(
                original,
                profile=RESIDENT_STATE_PROFILE_HANDLE_CARDS_ACTIVE_ARCHIVE,
                latest_observation_full=False,
                previous_success=previous_success,
            )

            original_text = visible_text(original)
            v47_text = visible_text(v47)
            parsed = turn.get("parsed") or {}
            lost_by_version = {}
            for version, candidate_text in (
                ("version47", v47_text),
                ("version49", visible_text(v49)),
            ):
                lost = []
                for path, value in leaves(parsed.get("arguments") or {}):
                    if not is_grounding_value_path(path):
                        continue
                    token = serialized_leaf(value)
                    if token in original_text and token not in candidate_text:
                        lost.append({"path": path, "value": value})
                lost_by_version[version] = lost
                if lost:
                    lost_events[version].append({
                        "trajectory_id": record.get("trajectory_id"),
                        "turn_index": turn_index,
                        "tool": parsed.get("tool"),
                        "lost_argument_leaves": lost,
                    })
            rows.append({
                "trajectory_id": record.get("trajectory_id"),
                "turn_index": turn_index,
                "correct": bool(record.get("correct")),
                "original_chars": input_chars(original),
                "version46_chars": input_chars(v46),
                "version47_chars": input_chars(v47),
                "version48_chars": input_chars(v48),
                "version49_chars": input_chars(v49),
                "version46_saved": input_chars(original) - input_chars(v46),
                "version47_saved": input_chars(original) - input_chars(v47),
                "version48_saved": input_chars(original) - input_chars(v48),
                "version49_saved": input_chars(original) - input_chars(v49),
                "version47_lost_next_action_leaf_count": len(lost_by_version["version47"]),
                "version49_lost_next_action_leaf_count": len(lost_by_version["version49"]),
            })

            if isinstance(turn.get("parsed"), dict) and isinstance(turn.get("tool_output"), dict):
                previous_success = {
                    "step_id": f"step_{turn_index + 1}",
                    "tool_output": turn["tool_output"],
                }

    final_rows = {}
    for row in rows:
        final_rows[row["trajectory_id"]] = row
    summary = {
        "records": len(records),
        "turns": len(rows),
        "final_turns": len(final_rows),
        "all_turn_context_chars": {
            key: numeric_summary([row[key] for row in rows])
            for key in (
                "original_chars", "version46_chars", "version47_chars", "version48_chars",
                "version49_chars", "version46_saved", "version47_saved", "version48_saved",
                "version49_saved",
            )
        },
        "final_turn_context_chars": {
            key: numeric_summary([row[key] for row in final_rows.values()])
            for key in (
                "original_chars", "version46_chars", "version47_chars", "version48_chars",
                "version49_chars", "version46_saved", "version47_saved", "version48_saved",
                "version49_saved",
            )
        },
        "next_action_visibility": {
            version: {
                "turns_with_argument_leaf_removed": len(events),
                "removed_argument_leaf_count": sum(
                    len(item["lost_argument_leaves"]) for item in events
                ),
            }
            for version, events in lost_events.items()
        },
    }
    return summary, lost_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, lost = audit(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "visibility_events": lost}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
