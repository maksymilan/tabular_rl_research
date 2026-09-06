#!/usr/bin/env python3
"""Replay lineage-aware asymmetric credit with deterministic error override.

This is an offline diagnostic over saved rollout JSONL.  It does not read or use ``gold_sql``.
The replay keeps the original group-standardized terminal advantage as its only scalar reward,
then applies the following per-turn precedence:

1. A Harness-owned deterministic model error receives an independent normalized penalty;
2. an infrastructure timeout receives zero policy credit;
3. a successful lineage action observed in both correct and incorrect trajectories keeps the
   positive occurrence and zeros the negative occurrence;
4. every other successful action keeps the original trajectory advantage.

The script emits an all-data summary plus a human-readable casebook for selected real K8 groups.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl" / "diagnostics"))

import rl.scenarios.diagnostics.audit_saam_lineage_replay as lineage  # noqa: E402
from rl.frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402
from rl.diagnostics.replay import (  # noqa: E402
    format_action,
    group_rollouts,
    local_error_statuses,
    read_rows,
    stable_action_digest,
)


SCHEMA_VERSION = "lineage-asymmetric-error-override-replay-v1"


_read_rows = read_rows


_group_rows = group_rollouts


def error_signature(turn: dict[str, Any], event: dict[str, Any]) -> str:
    parsed = turn.get("parsed")
    if isinstance(parsed, dict):
        action = {"tool": parsed.get("tool"), "arguments": parsed.get("arguments")}
    else:
        action = {
            "tool": event.get("attempted_tool"),
            "arguments": event.get("attempted_arguments"),
            "model_output": turn.get("model_output"),
        }
    return stable_action_digest(action)


_error_signature = error_signature


def _local_statuses(row: dict[str, Any]) -> list[dict[str, Any]]:
    return local_error_statuses(row, error_signature=error_signature)


def _short_action(event: dict[str, Any], turn: dict[str, Any]) -> str:
    if event.get("action_matchable"):
        return format_action(event)
    error = turn.get("error_event")
    if isinstance(error, dict):
        attempted = error.get("attempted_tool")
        if attempted:
            return f"INVALID({attempted})"
    parsed = turn.get("parsed")
    if isinstance(parsed, dict) and parsed.get("tool"):
        return f"UNRESOLVED({parsed['tool']})"
    return "INVALID_RESPONSE"


def _sign(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "zero"


def _mass(records: Iterable[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [float(record[field]) for record in records]
    return {
        "events": len(values),
        "positive_events": sum(value > 1e-12 for value in values),
        "negative_events": sum(value < -1e-12 for value in values),
        "zero_events": sum(abs(value) <= 1e-12 for value in values),
        "positive_mass": sum(max(value, 0.0) for value in values),
        "negative_mass": sum(max(-value, 0.0) for value in values),
    }


def _direct_conflict(records: Iterable[dict[str, Any]], field: str) -> float:
    grouped: dict[tuple[int, int, str], list[float]] = collections.defaultdict(list)
    for record in records:
        key = record.get("action_key")
        if not isinstance(key, str):
            continue
        grouped[(record["policy_global_step"], record["example_index"], key)].append(float(record[field]))
    total = 0.0
    for values in grouped.values():
        positive = sum(max(value, 0.0) for value in values)
        negative = sum(max(-value, 0.0) for value in values)
        total += 2.0 * min(positive, negative)
    return total


def replay(
    rows: list[dict[str, Any]],
    expected_group_size: int,
    *,
    error_penalty: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(error_penalty) or error_penalty <= 0.0:
        raise ValueError("error_penalty must be finite and positive")
    grouped = _group_rows(rows)
    records: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    group_mixed_keys: dict[tuple[int, int], set[str]] = {}

    prepared: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for group_key, group in sorted(grouped.items()):
        if len(group) != expected_group_size:
            raise ValueError(f"group {group_key} has {len(group)} trajectories, expected {expected_group_size}")
        eligible = [lineage._process_update(row) for row in group]
        advantages = standardized_group_advantages(
            [1.0 if row.get("correct") is True else 0.0 for row in group],
            eligible,
        )
        prepared_group: list[dict[str, Any]] = []
        successful_outcomes: dict[str, set[bool]] = collections.defaultdict(set)
        for row, is_eligible, advantage in zip(group, eligible, advantages, strict=True):
            events, lineage_summary = lineage.LineageReplay(row).replay()
            statuses = _local_statuses(row)
            if len(events) != len(statuses):
                raise ValueError(f"event/status length mismatch for {row.get('trajectory_id')}")
            turns = row.get("turns") or []
            event_rows: list[dict[str, Any]] = []
            for event, status, turn in zip(events, statuses, turns, strict=True):
                action_key = event.get("action_digest") if event.get("action_matchable") else None
                event_row = {
                    "event": event,
                    "turn": turn,
                    "status": status,
                    "action_key": action_key,
                }
                event_rows.append(event_row)
                if (
                    is_eligible
                    and action_key is not None
                    and status["kind"] == "success"
                ):
                    successful_outcomes[action_key].add(bool(row.get("correct")))
            prepared_group.append(
                {
                    "row": row,
                    "eligible": bool(is_eligible),
                    "advantage": float(advantage),
                    "events": event_rows,
                    "lineage_summary": lineage_summary,
                }
            )
        mixed_keys = {key for key, outcomes in successful_outcomes.items() if outcomes == {True, False}}
        group_mixed_keys[group_key] = mixed_keys
        prepared[group_key] = prepared_group

    for group_key, group in sorted(prepared.items()):
        mixed_keys = group_mixed_keys[group_key]
        for item in group:
            row = item["row"]
            advantage = float(item["advantage"])
            trajectory_decisions: collections.Counter[str] = collections.Counter()
            for event_item in item["events"]:
                event = event_item["event"]
                turn = event_item["turn"]
                status = event_item["status"]
                action_key = event_item["action_key"]
                if not item["eligible"]:
                    effective = 0.0
                    decision = "trajectory_ineligible_zero"
                elif status["kind"] == "infrastructure_timeout":
                    effective = 0.0
                    decision = "infrastructure_timeout_zero"
                elif status["local_error"]:
                    # Do not derive this only from A_i: an eligible group can be homogeneous
                    # after generation-length/non-semantic failures are excluded, giving A_i ==
                    # 0.  The Harness error is still high-confidence negative supervision.  The
                    # floor also prevents weakening an already stronger negative GRPO update on
                    # an incorrect trajectory.
                    effective = -max(abs(advantage), error_penalty)
                    decision = f"deterministic_error_negative:{status['kind']}"
                elif action_key in mixed_keys:
                    if bool(row.get("correct")):
                        effective = advantage
                        decision = "shared_success_correct_keep_positive"
                    else:
                        effective = 0.0
                        decision = "shared_success_wrong_suppress_negative"
                else:
                    effective = advantage
                    decision = (
                        "unmatched_success_keep_trajectory_positive"
                        if advantage > 0.0
                        else "unmatched_success_keep_trajectory_negative"
                        if advantage < 0.0
                        else "unmatched_success_keep_zero"
                    )
                trajectory_decisions[decision] += 1
                records.append(
                    {
                        "policy_global_step": int(row["policy_global_step"]),
                        "example_index": int(row["example_index"]),
                        "trajectory_id": str(row.get("trajectory_id") or ""),
                        "depth": int(event["depth"]),
                        "correct": bool(row.get("correct")),
                        "legal": bool(row.get("legal")),
                        "eligible": bool(item["eligible"]),
                        "tool": (
                            event.get("action", {}).get("tool")
                            if isinstance(event.get("action"), dict)
                            else (turn.get("parsed") or {}).get("tool")
                            if isinstance(turn.get("parsed"), dict)
                            else None
                        ),
                        "action": _short_action(event, turn),
                        "action_key": action_key,
                        "mixed_success_key": action_key in mixed_keys if action_key else False,
                        "local_status": status["kind"],
                        "error_type": status.get("error_type"),
                        "old_advantage": advantage if item["eligible"] else 0.0,
                        "effective_advantage": effective,
                        "old_sign": _sign(advantage if item["eligible"] else 0.0),
                        "effective_sign": _sign(effective),
                        "decision": decision,
                    }
                )
            trajectory_summaries.append(
                {
                    "policy_global_step": int(row["policy_global_step"]),
                    "example_index": int(row["example_index"]),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "correct": bool(row.get("correct")),
                    "legal": bool(row.get("legal")),
                    "eligible": bool(item["eligible"]),
                    "advantage": advantage if item["eligible"] else 0.0,
                    "turns": len(item["events"]),
                    "decisions": dict(sorted(trajectory_decisions.items())),
                    "lineage_issues": item["lineage_summary"]["issue_kinds"],
                }
            )

    eligible_records = [record for record in records if record["eligible"]]
    decision_counts = collections.Counter(record["decision"] for record in eligible_records)
    error_by_terminal = collections.Counter()
    error_flip_positive = 0
    error_flip_positive_mass = 0.0
    for record in eligible_records:
        if not record["decision"].startswith("deterministic_error_negative"):
            continue
        error_by_terminal["correct" if record["correct"] else "wrong"] += 1
        error_by_terminal[record["local_status"]] += 1
        if record["old_advantage"] > 0.0:
            error_flip_positive += 1
            error_flip_positive_mass += float(record["old_advantage"])

    tool_decisions: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    depth_decisions: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for record in eligible_records:
        tool_decisions[str(record.get("tool") or "__INVALID__")][record["decision"]] += 1
        depth_decisions[int(record["depth"])][record["decision"]] += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "gold_sql_read": False,
        "error_penalty": error_penalty,
        "groups": len(grouped),
        "trajectories": len(rows),
        "eligible_trajectories": sum(int(summary["eligible"]) for summary in trajectory_summaries),
        "events": len(records),
        "eligible_events": len(eligible_records),
        "mixed_success_action_keys": sum(len(values) for values in group_mixed_keys.values()),
        "decision_counts": dict(sorted(decision_counts.items())),
        "old_credit": _mass(eligible_records, "old_advantage"),
        "effective_credit": _mass(eligible_records, "effective_advantage"),
        "old_action_direct_conflict": _direct_conflict(eligible_records, "old_advantage"),
        "effective_action_direct_conflict": _direct_conflict(eligible_records, "effective_advantage"),
        "deterministic_errors_by_terminal_and_kind": dict(sorted(error_by_terminal.items())),
        "correct_trajectory_error_events_flipped_positive_to_negative": error_flip_positive,
        "correct_trajectory_positive_mass_flipped_to_negative": error_flip_positive_mass,
        "tool_decisions": {key: dict(sorted(value.items())) for key, value in sorted(tool_decisions.items())},
        "depth_decisions": {str(key): dict(sorted(value.items())) for key, value in sorted(depth_decisions.items())},
    }
    return {
        "summary": summary,
        "records": records,
        "trajectory_summaries": trajectory_summaries,
    }


def _parse_group(value: str) -> tuple[int, int]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("group must be POLICY_STEP:EXAMPLE_INDEX")
    return int(parts[0]), int(parts[1])


def _write_casebook(path: Path, result: dict[str, Any], selected: set[tuple[int, int]]) -> None:
    records = [
        record
        for record in result["records"]
        if (record["policy_global_step"], record["example_index"]) in selected
    ]
    trajectory_summaries = [
        summary
        for summary in result["trajectory_summaries"]
        if (summary["policy_global_step"], summary["example_index"]) in selected
    ]
    by_trajectory: dict[tuple[int, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_trajectory[(record["policy_global_step"], record["example_index"], record["trajectory_id"])].append(record)
    selected_eligible = [record for record in records if record["eligible"]]
    lines = [
        "# Lineage-aware asymmetric credit static replay casebook",
        "",
        f"- selected groups: {len(selected)}",
        f"- selected trajectories: {len(trajectory_summaries)}",
        f"- selected events: {len(records)}",
        f"- selected eligible events: {len(selected_eligible)}",
        "- gold_sql read: false",
        "",
        "Decision precedence: deterministic error negative > infrastructure zero > shared-success asymmetric credit > original trajectory credit.",
        "",
    ]
    for summary in sorted(
        trajectory_summaries,
        key=lambda item: (item["policy_global_step"], item["example_index"], item["trajectory_id"]),
    ):
        key = (summary["policy_global_step"], summary["example_index"], summary["trajectory_id"])
        lines.extend(
            [
                f"## step={key[0]} example={key[1]} trajectory={key[2]}",
                "",
                (
                    f"correct={int(summary['correct'])}, legal={int(summary['legal'])}, "
                    f"eligible={int(summary['eligible'])}, advantage={summary['advantage']:+.6f}, "
                    f"turns={summary['turns']}, lineage_issues={json.dumps(summary['lineage_issues'], ensure_ascii=False)}"
                ),
                "",
                "| depth | action | local status | mixed | old A | effective A | decision |",
                "|---:|---|---|:---:|---:|---:|---|",
            ]
        )
        for record in sorted(by_trajectory[key], key=lambda item: item["depth"]):
            action = str(record["action"]).replace("|", "\\|")
            lines.append(
                f"| {record['depth']} | `{action}` | {record['local_status']} | "
                f"{int(record['mixed_success_key'])} | {record['old_advantage']:+.6f} | "
                f"{record['effective_advantage']:+.6f} | {record['decision']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--select-group", action="append", type=_parse_group, default=[])
    parser.add_argument("--error-penalty", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--casebook", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_rows(args.rollouts)
    result = replay(rows, args.expected_group_size, error_penalty=args.error_penalty)
    selected = set(args.select_group)
    available = set(_group_rows(rows))
    missing = selected - available
    if missing:
        raise ValueError(f"selected groups not found: {sorted(missing)}")
    if not selected:
        selected = set(sorted(available)[:8])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_casebook(args.casebook, result, selected)
    compact = dict(result["summary"])
    compact["selected_groups"] = sorted([list(key) for key in selected])
    compact["selected_trajectories"] = sum(
        (summary["policy_global_step"], summary["example_index"]) in selected
        for summary in result["trajectory_summaries"]
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
