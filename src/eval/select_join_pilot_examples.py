#!/usr/bin/env python3
"""Freeze a hard join pilot cohort without using gold SQL or terminal correctness for selection.

The candidate pool is behavior-conditioned: a prior causal model↔harness trajectory must have
actually attempted join_tables. Ranking uses only model actions and harness-local error/audit
signals. Full source examples (which contain hidden gold SQL for terminal scoring) are joined back
*after* the selected ids have been frozen.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FORBIDDEN_SELECTION_FIELDS = frozenset({
    "gold_sql", "gold_exec_results", "difficulty", "difficulty_proxy",
    "correct", "legal", "failure_type", "fail",
})


def _join_arity(arguments: dict) -> int:
    if isinstance(arguments.get("joins"), list):
        return 1 + len(arguments["joins"])
    if isinstance(arguments.get("tables"), list):
        return len(arguments["tables"])
    if arguments.get("left") is not None and arguments.get("right") is not None:
        return 2
    return 0


def _join_edges(arguments: dict) -> int:
    if isinstance(arguments.get("joins"), list):
        return sum(
            len(item.get("on") or [])
            for item in arguments["joins"]
            if isinstance(item, dict)
        )
    on = arguments.get("on")
    if not isinstance(on, list):
        return 0
    return sum(len(item) if isinstance(item, list) else 1 for item in on)


def behavior_signals(record: dict) -> dict | None:
    """Extract ranking signals without consulting any terminal/gold fields."""
    turns = record.get("turns") or []
    join_arguments = [
        (turn.get("parsed") or {}).get("arguments") or {}
        for turn in turns
        if (turn.get("parsed") or {}).get("tool") == "join_tables"
    ]
    if not join_arguments:
        return None
    error_events = record.get("error_events") or []
    join_local_errors = sum(
        event.get("attempted_tool") == "join_tables"
        and event.get("error_type") in {"argument_validation_error", "execution_error"}
        for event in error_events
    )
    arities = [_join_arity(arguments) for arguments in join_arguments]
    edges = [_join_edges(arguments) for arguments in join_arguments]
    table_refs: set[str] = set()
    for arguments in join_arguments:
        table_refs.update(
            item for item in arguments.get("tables") or [] if isinstance(item, str)
        )
        for key in ("left", "right", "base"):
            if isinstance(arguments.get(key), str):
                table_refs.add(arguments[key])
        table_refs.update(
            item.get("table") for item in arguments.get("joins") or []
            if isinstance(item, dict) and isinstance(item.get("table"), str)
        )
    action_tools = [
        (turn.get("parsed") or {}).get("tool")
        for turn in turns
        if isinstance((turn.get("parsed") or {}).get("tool"), str)
    ]
    signals = {
        "trajectory_id": record.get("trajectory_id"),
        "example_index": record.get("example_index"),
        "db_id": record.get("db_id"),
        "join_local_errors": int(join_local_errors),
        "join_calls": len(join_arguments),
        "max_join_arity": max(arities),
        "max_join_edges": max(edges),
        "distinct_join_inputs": len(table_refs),
        "action_count": len(turns),
        "distinct_action_tools": len(set(action_tools)),
    }
    # The score is diagnostic; deterministic ordering below uses the component tuple directly.
    signals["hardness_score"] = (
        1000 * signals["join_local_errors"]
        + 100 * max(0, signals["max_join_arity"] - 2)
        + 20 * max(0, signals["join_calls"] - 1)
        + 5 * signals["max_join_edges"]
        + min(signals["action_count"], 99)
        + signals["distinct_action_tools"]
    )
    return signals


def select_ids(records: list[dict], count: int) -> list[dict]:
    candidates = [signals for record in records if (signals := behavior_signals(record))]
    candidates.sort(
        key=lambda item: (
            -item["join_local_errors"],
            -item["max_join_arity"],
            -item["join_calls"],
            -item["max_join_edges"],
            -item["action_count"],
            -item["distinct_join_inputs"],
            str(item["trajectory_id"]),
        )
    )
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} behavior-confirmed join candidates for count={count}")
    return candidates[:count]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    selected = select_ids(_read_jsonl(args.behavior), args.count)
    # Selection is complete at this boundary. Only now load full examples containing verifier-only
    # gold fields, and use them strictly as opaque payloads keyed by example_id.
    source = _read_jsonl(args.source_examples)
    by_id = {
        item.get("example_id") or item.get("instance_id"): item
        for item in source
    }
    missing = [item["trajectory_id"] for item in selected if item["trajectory_id"] not in by_id]
    if missing:
        raise ValueError(f"selected ids missing from source examples: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(by_id[item["trajectory_id"]], ensure_ascii=False) + "\n")

    manifest = {
        "method": "behavior_conditioned_join_pilot_no_gold_selection",
        "behavior_source": str(args.behavior),
        "source_examples": str(args.source_examples),
        "output": str(args.output),
        "count": len(selected),
        "candidate_count": sum(
            behavior_signals(record) is not None for record in _read_jsonl(args.behavior)
        ),
        "selection_fields": [
            "prior parsed join_tables arguments",
            "join-local argument/execution errors",
            "action count",
            "distinct action tools",
        ],
        "forbidden_selection_fields": sorted(FORBIDDEN_SELECTION_FIELDS),
        "db_histogram": dict(sorted(Counter(item["db_id"] for item in selected).items())),
        "selected": [{"rank": rank, **item} for rank, item in enumerate(selected, 1)],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".selection.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(manifest_path),
        "selected": len(selected),
        "candidates": manifest["candidate_count"],
    }))


if __name__ == "__main__":
    main()
