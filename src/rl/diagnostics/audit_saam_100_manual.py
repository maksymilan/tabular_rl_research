#!/usr/bin/env python3
"""Manual-style, per-trajectory audit for a fixed 100 Gate60 rollouts.

This diagnostic intentionally ignores the raw ``gold_sql`` field.  It uses only actor actions,
Harness observations/derivations, terminal correctness, and the existing lineage replay helper.
It prints compact action traces so every selected trajectory can be inspected rather than only
reporting aggregate counts.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl" / "diagnostics"))

import audit_saam_lineage_replay as audit  # noqa: E402
from frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(int(row["policy_global_step"]), int(row["example_index"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: str(item.get("trajectory_id", "")))
    return grouped


def _node(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    kind = value.get("kind")
    if kind == "source":
        return str(value.get("table", "?"))
    if kind == "relation":
        inputs = [
            _node(item.get("ref"))
            for item in value.get("inputs", [])
            if isinstance(item, dict)
        ]
        return f"{value.get('operator', '?')}({','.join(inputs)})"
    if kind == "column":
        return f"{_node(value.get('relation'))}.{value.get('column', '?')}"
    if kind in {"unresolved", "unresolved_handle", "unresolved_step"}:
        return "UNRES"
    return str(kind or "?")


def _condition(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    if "and" in value:
        return "AND[" + ",".join(_condition(item) for item in value["and"]) + "]"
    if "or" in value:
        return "OR[" + ",".join(_condition(item) for item in value["or"]) + "]"
    if "not" in value:
        return "NOT[" + _condition(value["not"]) + "]"
    column = value.get("column", value.get("column_value", "?"))
    op = value.get("op", "?")
    raw = value.get("value", value.get("values", value.get("in_table", "")))
    if isinstance(raw, dict):
        raw = "REF"
    return f"{column}{op}{raw}"


def _action(event: dict[str, Any]) -> str:
    action = event.get("action") or {}
    tool = action.get("tool", "?")
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return tool
    if tool == "describe_table":
        return "D(" + ",".join(map(str, args.get("tables", []))) + ")"
    if tool == "condition_filter":
        return f"F({_node(args.get('table'))};{_condition(args.get('conditions'))};ret={args.get('return_columns')})"
    if tool == "inspect_column":
        return f"I({_node(args.get('table'))}.{args.get('column')})"
    if tool == "read_subtable":
        return f"R({_node(args.get('table'))};cols={args.get('columns')};n={args.get('limit')})"
    if tool == "join_tables":
        joins = []
        for item in args.get("joins", []):
            edges = []
            for edge in item.get("on", []):
                left = edge.get("left")
                left = _node(left) if isinstance(left, dict) else str(left)
                edges.append(f"{left}={edge.get('right')}")
            joins.append(f"{_node(item.get('table'))}[{','.join(edges)}]")
        return f"J({_node(args.get('base'))};{'|'.join(joins)})"
    if tool == "project":
        return f"P({_node(args.get('table'))};expr={args.get('expressions')};d={args.get('distinct')})"
    if tool == "group_aggregate":
        aggs = ",".join(
            f"{item.get('op')}:{item.get('column')}"
            for item in args.get("aggregations", [])
            if isinstance(item, dict)
        )
        return f"G({_node(args.get('table'))};gb={args.get('group_by')};{aggs})"
    if tool == "extreme_value_select":
        return f"E({_node(args.get('table'))};ord={args.get('order_by')};k={args.get('top_k')};ret={args.get('return_columns')})"
    if tool == "scalar_compute":
        return f"C({args.get('operation')};n={len(args.get('operands', []))})"
    if tool == "set_op":
        return f"S({_node(args.get('left'))},{_node(args.get('right'))};{args.get('op')})"
    if tool == "answer_from_context":
        evidence = args.get("evidence") or {}
        return f"A({_node(evidence.get('table'))})"
    return tool


def _prepare(rows: list[dict[str, Any]]) -> tuple[dict[tuple[int, int, str], Any], dict[str, set[bool]], dict[str, set[bool]], dict[str, set[bool]]]:
    grouped = _groups(rows)
    infos: dict[tuple[int, int, str], Any] = {}
    lineage_action_outcomes: dict[str, set[bool]] = collections.defaultdict(set)
    strict_outcomes: dict[str, set[bool]] = collections.defaultdict(set)
    literal_outcomes: dict[str, set[bool]] = collections.defaultdict(set)
    for _, group in sorted(grouped.items()):
        advantages = standardized_group_advantages(
            [1.0 if row.get("correct") is True else 0.0 for row in group],
            [audit._process_update(row) for row in group],
        )
        for row, advantage in zip(group, advantages, strict=True):
            replay = audit.LineageReplay(row)
            events, summary = replay.replay()
            eligible = audit._process_update(row)
            for event in events:
                event["training_eligible"] = eligible
            if not eligible:
                row_key = (
                    int(row["policy_global_step"]),
                    int(row["example_index"]),
                    str(row["trajectory_id"]),
                )
                if row_key in infos:
                    raise ValueError(f"duplicate rollout identity: {row_key}")
                infos[row_key] = {"row": row, "events": events, "summary": summary}
                continue
            for event in events:
                event["advantage"] = float(advantage)
                if event.get("action_matchable"):
                    action_key = f"{event['policy_global_step']}|{event['example_index']}|{event['action_digest']}"
                    lineage_action_outcomes[action_key].add(bool(row.get("correct")))
                    literal = audit._literal_action_signature(event)
                    if literal is not None:
                        literal_key = f"{event['policy_global_step']}|{event['example_index']}|{literal}"
                        literal_outcomes[literal_key].add(bool(row.get("correct")))
                if event.get("matched"):
                    strict_key = f"{event['policy_global_step']}|{event['example_index']}|{event['state_digest']}|{event['action_digest']}"
                    strict_outcomes[strict_key].add(bool(row.get("correct")))
            row_key = (
                int(row["policy_global_step"]),
                int(row["example_index"]),
                str(row["trajectory_id"]),
            )
            if row_key in infos:
                raise ValueError(f"duplicate rollout identity: {row_key}")
            infos[row_key] = {"row": row, "events": events, "summary": summary}
    return infos, lineage_action_outcomes, strict_outcomes, literal_outcomes


def _select(rows: list[dict[str, Any]], per_step: int) -> list[dict[str, Any]]:
    grouped = _groups(rows)
    mixed = {
        key
        for key, group in grouped.items()
        if {bool(row.get("correct")) for row in group} == {True, False}
    }
    selected: list[dict[str, Any]] = []
    for step in sorted({key[0] for key in grouped}):
        candidates = sorted(
            [
                row
                for row in rows
                if (int(row["policy_global_step"]), int(row["example_index"])) in mixed
                and int(row["policy_global_step"]) == step
            ],
            key=lambda row: (int(row["example_index"]), str(row.get("trajectory_id", ""))),
        )
        selected.extend(candidates[:per_step])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--per-step", type=int, default=25)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollouts.open(encoding="utf-8") if line.strip()]
    infos, action_outcomes, strict_outcomes, literal_outcomes = _prepare(rows)
    selected = _select(rows, args.per_step)
    if args.step is not None:
        selected = [row for row in selected if int(row["policy_global_step"]) == args.step]
    action_mixed = {key for key, values in action_outcomes.items() if values == {True, False}}
    strict_mixed = {key for key, values in strict_outcomes.items() if values == {True, False}}
    literal_mixed = {key for key, values in literal_outcomes.items() if values == {True, False}}
    print(
        f"selected={len(selected)} eligible={sum(audit._process_update(row) for row in selected)} "
        f"per_step={args.per_step} action_mixed_keys={len(action_mixed)} "
        f"strict_mixed_keys={len(strict_mixed)} literal_mixed_keys={len(literal_mixed)}"
    )
    for row in selected:
        tid = str(row["trajectory_id"])
        row_key = (int(row["policy_global_step"]), int(row["example_index"]), tid)
        info = infos[row_key]
        strict_depths: list[int] = []
        action_depths: list[int] = []
        literal_mixed_depths: list[int] = []
        false_literal_depths: list[int] = []
        missed_literal_depths: list[int] = []
        actions: list[str] = []
        for event in info["events"]:
            if not event.get("action_matchable"):
                actions.append(f"{event['depth']}:UNMATCHED")
                continue
            action_key = f"{event['policy_global_step']}|{event['example_index']}|{event['action_digest']}"
            strict_key = f"{event['policy_global_step']}|{event['example_index']}|{event['state_digest']}|{event['action_digest']}" if event.get("matched") else None
            literal = audit._literal_action_signature(event)
            literal_key = f"{event['policy_global_step']}|{event['example_index']}|{literal}" if literal is not None else None
            flags = ""
            if strict_key is not None and strict_key in strict_mixed:
                flags += "S"
                strict_depths.append(int(event["depth"]))
            if action_key in action_mixed:
                flags += "M"
                action_depths.append(int(event["depth"]))
            if literal_key in literal_mixed:
                flags += "L"
                literal_mixed_depths.append(int(event["depth"]))
                if action_key not in action_mixed:
                    flags += "X"
                    false_literal_depths.append(int(event["depth"]))
            if action_key in action_mixed and literal_key not in literal_mixed:
                flags += "G"
                missed_literal_depths.append(int(event["depth"]))
            actions.append(f"{event['depth']}:{_action(event)}" + (f"[{flags}]" if flags else ""))
        summary = info["summary"]
        issues = ",".join(f"{key}:{value}" for key, value in summary["issue_kinds"].items()) or "-"
        print(
            f"{row['policy_global_step']}|{row['example_index']}|{tid}|eligible={int(audit._process_update(row))}|y={int(bool(row.get('correct')))}|legal={int(bool(row.get('legal')))}|turns={len(row.get('turns', []))}|err={len(row.get('error_events') or [])}|issues={issues}"
        )
        print("  " + " | ".join(actions))
        print(f"  flags S={strict_depths} M={action_depths} L={literal_mixed_depths} X={false_literal_depths} G={missed_literal_depths}")


if __name__ == "__main__":
    main()
