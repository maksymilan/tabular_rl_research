#!/usr/bin/env python3
"""Summarize tool usage for table-RL trajectory or SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def iter_jsonl(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def walk_conditions(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_conditions(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_conditions(child)


def calls_from_trajectory(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for step in record.get("steps") or []:
        tool_call = step.get("tool_call") or {}
        tool = tool_call.get("tool")
        if tool:
            calls.append((tool, tool_call.get("arguments") or {}))
    return calls


def calls_from_sft(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    conversations = record.get("conversations") or record.get("messages") or []
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in conversations:
        role = message.get("from") or message.get("role")
        if role not in {"assistant", "gpt"}:
            continue
        content = message.get("value") or message.get("content") or ""
        for match in TOOL_BLOCK_RE.finditer(content):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            tool = payload.get("tool")
            if tool:
                calls.append((tool, payload.get("arguments") or {}))
    return calls


def feature_tags(tool: str, args: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    if tool == "join_tables":
        n_tables = len(args.get("tables") or [])
        tags.add("nway_join_3plus" if n_tables >= 3 else "join_2way")
    elif tool == "condition_filter":
        for cond in walk_conditions(args.get("conditions")):
            op = str(cond.get("op", "")).lower()
            if "value_ref" in cond:
                tags.add("value_ref_filter")
            if "in_table" in cond:
                tags.add("in_table_filter")
            if op in {"like", "not like"}:
                tags.add("like_filter")
    elif tool == "set_op":
        op = str(args.get("op", "")).lower() or "unknown"
        tags.add(f"set_op_{op}")
    elif tool == "group_aggregate":
        group_by = args.get("group_by") or []
        aggregations = args.get("aggregations") or []
        if not group_by:
            tags.add("whole_table_group_aggregate")
        if len(aggregations) >= 2:
            tags.add("multi_aggregate")
            tags.add("grouped_multi_agg" if group_by else "whole_table_multi_agg")
    elif tool == "aggregate":
        tags.add("single_scalar_aggregate")
    elif tool == "extreme_value_select":
        tags.add("extreme_value_select")
    elif tool == "read_subtable":
        tags.add("read_subtable")
    elif tool == "describe_table":
        tables = args.get("tables")
        if isinstance(tables, list) and len(tables) >= 2:
            tags.add("multi_table_describe")
    return tags


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[idx]


def analyze(paths: list[Path]) -> dict[str, Any]:
    records = 0
    total_calls = 0
    tool_counts: Counter[str] = Counter()
    traj_with_tool: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    step_counts: list[int] = []

    for _path, _line_no, record in iter_jsonl(paths):
        records += 1
        calls = calls_from_trajectory(record)
        if not calls:
            calls = calls_from_sft(record)
        step_counts.append(len(calls))
        local_tools: Counter[str] = Counter()
        local_features: set[str] = set()
        for tool, args in calls:
            total_calls += 1
            tool_counts[tool] += 1
            local_tools[tool] += 1
            local_features.update(feature_tags(tool, args))
        traj_with_tool.update(local_tools.keys())
        feature_counts.update(local_features)

    avg_steps = (sum(step_counts) / records) if records else 0.0
    return {
        "files": [str(p) for p in paths],
        "records": records,
        "total_tool_calls": total_calls,
        "step_counts": {
            "avg": avg_steps,
            "min": min(step_counts) if step_counts else None,
            "max": max(step_counts) if step_counts else None,
            "p50": percentile(step_counts, 0.50),
            "p75": percentile(step_counts, 0.75),
            "p90": percentile(step_counts, 0.90),
            "p95": percentile(step_counts, 0.95),
            "p99": percentile(step_counts, 0.99),
        },
        "tools": [
            {
                "tool": tool,
                "calls": count,
                "call_pct": (100.0 * count / total_calls) if total_calls else 0.0,
                "trajectories": traj_with_tool[tool],
                "trajectory_pct": (100.0 * traj_with_tool[tool] / records) if records else 0.0,
                "avg_per_trajectory": (count / records) if records else 0.0,
            }
            for tool, count in tool_counts.most_common()
        ],
        "features": [
            {
                "feature": feature,
                "trajectories": count,
                "trajectory_pct": (100.0 * count / records) if records else 0.0,
            }
            for feature, count in feature_counts.most_common()
        ],
    }


def print_text(stats: dict[str, Any]) -> None:
    steps = stats["step_counts"]
    print("Files:", ", ".join(stats["files"]))
    print("Records:", stats["records"])
    print("Total tool calls:", stats["total_tool_calls"])
    print(
        "Steps per trajectory:",
        f"avg={steps['avg']:.3f}",
        f"min={steps['min']}",
        f"p50={steps['p50']}",
        f"p75={steps['p75']}",
        f"p90={steps['p90']}",
        f"p95={steps['p95']}",
        f"p99={steps['p99']}",
        f"max={steps['max']}",
    )
    print()
    print("TOOL\tCALLS\tCALL_%\tTRAJ_WITH\tTRAJ_%\tAVG_PER_TRAJ")
    for row in stats["tools"]:
        print(
            f"{row['tool']}\t{row['calls']}\t{row['call_pct']:.2f}\t"
            f"{row['trajectories']}\t{row['trajectory_pct']:.2f}\t"
            f"{row['avg_per_trajectory']:.3f}"
        )
    print()
    print("FEATURE\tTRAJ_WITH\tTRAJ_%")
    for row in stats["features"]:
        print(f"{row['feature']}\t{row['trajectories']}\t{row['trajectory_pct']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="trajectory or SFT JSONL file(s)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tab-separated text")
    args = parser.parse_args()

    missing = [str(p) for p in args.jsonl if not p.exists()]
    if missing:
        raise SystemExit("Missing file(s): " + ", ".join(missing))

    stats = analyze(args.jsonl)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print_text(stats)


if __name__ == "__main__":
    main()
