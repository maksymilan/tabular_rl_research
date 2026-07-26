#!/usr/bin/env python3
"""Build compact, replay-auditable views of failed action-block trajectories.

This is an offline evaluation utility.  Gold SQL is included only as audit
evidence; it is never rendered into a model request or an environment state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


_NOISY_OUTPUT_KEYS = {
    "environment_state",
    "environment_state_before",
    "derivation",
}


def _read_jsonl(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                trajectory_id = record.get("trajectory_id")
                if not isinstance(trajectory_id, str) or not trajectory_id:
                    raise ValueError(
                        f"{path}:{line_number}: missing trajectory_id"
                    )
                if trajectory_id in records:
                    raise ValueError(
                        f"duplicate trajectory_id {trajectory_id!r}"
                    )
                records[trajectory_id] = record
    return records


def _bounded_rows(value: Any, limit: int) -> Any:
    if not isinstance(value, list):
        return value
    return value[:limit]


def _compact_output(output: Any, row_limit: int) -> Any:
    if not isinstance(output, dict):
        return output

    compact: dict[str, Any] = {}
    for key, value in output.items():
        if key in _NOISY_OUTPUT_KEYS:
            continue
        if key == "tables" and isinstance(value, list):
            compact[key] = [
                {
                    "table_name": table.get("table_name"),
                    "row_count": table.get("row_count"),
                    "columns": [
                        column.get("name")
                        for column in table.get("columns") or []
                    ],
                    "foreign_keys": table.get("foreign_keys") or [],
                }
                for table in value
            ]
            continue
        if key == "rows":
            compact[key] = _bounded_rows(value, row_limit)
            if isinstance(value, list) and len(value) > row_limit:
                compact["rows_truncated_for_audit"] = True
            continue
        compact[key] = value
    return compact


def _compact_result(result: dict[str, Any], row_limit: int) -> dict[str, Any]:
    authored = result.get("arguments")
    resolved = result.get("resolved_arguments")
    compact = {
        "call_id": result.get("call_id"),
        "step_id": result.get("step_id"),
        "tool": result.get("tool"),
        "status": result.get("status"),
        "dependencies": result.get("dependencies") or [],
        "arguments": authored,
        "output": _compact_output(result.get("output"), row_limit),
    }
    if resolved is not None and resolved != authored:
        compact["resolved_arguments"] = resolved
    for key in (
        "blocked_by",
        "root_causes",
        "error_type",
        "message",
        "details",
        "interface_resolutions",
        "terminal_projection",
    ):
        if result.get(key) is not None:
            compact[key] = result[key]
    return compact


def _compact_turn(turn: dict[str, Any], row_limit: int) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "turn_index": turn.get("turn_index"),
        "reasoning": turn.get("provider_reasoning_content"),
        "action": turn.get("parsed"),
    }
    batch_results = turn.get("batch_results")
    if isinstance(batch_results, list):
        compact["results"] = [
            _compact_result(result, row_limit)
            for result in batch_results
        ]
    if turn.get("terminal_result") is not None:
        compact["terminal_result"] = turn["terminal_result"]
    for key in ("root_error_count", "blocked_count"):
        if turn.get(key):
            compact[key] = turn[key]
    return compact


def _compact_baseline(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "correct": bool(record.get("correct")),
        "legal": bool(record.get("legal")),
        "failure_type": record.get("failure_type"),
        "pred_sample": record.get("pred_sample"),
        "gold_sample": record.get("gold_sample"),
        "errors": record.get("errors"),
    }


def _audit_record(
    record: dict[str, Any],
    *,
    baseline: dict[str, Any] | None,
    row_limit: int,
) -> dict[str, Any]:
    return {
        "trajectory_id": record["trajectory_id"],
        "example_index": record.get("example_index"),
        "db_id": record.get("db_id"),
        "difficulty": record.get("difficulty"),
        "question": record.get("question"),
        "gold_sql": record.get("gold_sql"),
        "outcome": {
            "correct": bool(record.get("correct")),
            "legal": bool(record.get("legal")),
            "failure_type": record.get("failure_type"),
            "pred_sample": record.get("pred_sample"),
            "gold_sample": record.get("gold_sample"),
            "model_turns": record.get("model_turns"),
            "atomic_actions": record.get("atomic_actions"),
            "errors": record.get("errors"),
            "blocked_nodes": record.get("blocked_nodes"),
            "interface_resolutions": record.get("interface_resolutions"),
        },
        "baseline": _compact_baseline(baseline),
        "error_events": record.get("error_events") or [],
        "interface_resolution_events": (
            record.get("interface_resolution_events") or []
        ),
        "turns": [
            _compact_turn(turn, row_limit)
            for turn in record.get("turns") or []
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, default=5)
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="include successful trajectories as well as failures",
    )
    args = parser.parse_args()

    if args.row_limit < 1:
        parser.error("--row-limit must be positive")

    records = _read_jsonl(args.input)
    baseline_records = _read_jsonl(args.baseline) if args.baseline else {}
    selected = [
        records[trajectory_id]
        for trajectory_id in sorted(records)
        if args.include_correct or not records[trajectory_id].get("correct")
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in selected:
            compact = _audit_record(
                record,
                baseline=baseline_records.get(record["trajectory_id"]),
                row_limit=args.row_limit,
            )
            stream.write(json.dumps(compact, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_records": len(records),
                "audit_records": len(selected),
                "baseline_records": len(baseline_records),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
