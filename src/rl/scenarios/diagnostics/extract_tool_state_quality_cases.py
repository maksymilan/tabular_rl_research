#!/usr/bin/env python3
"""Extract a compact, human-auditable casebook from frozen rollout records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_KEYS = {
    "table", "tables", "columns", "row_count", "rows", "values", "value", "stats",
    "distinct_values", "is_truncated", "derivation", "final_answer", "error", "message",
}


def compact(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "<depth-limit>"
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…"
    if isinstance(value, list):
        retained = [compact(item, depth=depth + 1) for item in value[:30]]
        if len(value) > 30:
            retained.append(f"<{len(value) - 30} more items>")
        return retained
    if isinstance(value, dict):
        return {str(key): compact(item, depth=depth + 1) for key, item in value.items()}
    return value


def compact_output(output: Any) -> Any:
    if not isinstance(output, dict):
        return compact(output)
    selected = {key: value for key, value in output.items() if key in OUTPUT_KEYS}
    return compact(selected or output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-id", action="append")
    selection.add_argument(
        "--task-id-file", type=Path,
        help="JSON list containing task-id strings or objects with a task_id field",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.task_id_file is not None:
        payload = json.loads(args.task_id_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("task-id-file must contain a JSON list")
        wanted = {
            str(item.get("task_id")) if isinstance(item, dict) else str(item)
            for item in payload
        }
    else:
        wanted = set(args.task_id or [])
    if not wanted or "None" in wanted:
        raise ValueError("at least one non-null task id is required")
    scores = {}
    with args.scores.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            task_id = str(row.get("task_id"))
            if task_id in wanted:
                scores[task_id] = row

    cases = []
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            environment = row.get("environment") or {}
            task_id = str(environment.get("task_id") or environment.get("example_id"))
            if task_id not in wanted:
                continue
            record = ((row.get("sample") or {}).get("audit_record")) or {}
            turns = []
            for turn in record.get("turns") or []:
                parsed = turn.get("parsed") or {}
                if not parsed.get("tool") and not turn.get("execution_error_type"):
                    continue
                turns.append({
                    "turn_index": turn.get("turn_index"),
                    "think": compact(parsed.get("think") or ""),
                    "tool": parsed.get("tool"),
                    "arguments": compact(parsed.get("arguments") or {}),
                    "tool_output": compact_output(turn.get("tool_output")),
                    "execution_error_type": turn.get("execution_error_type"),
                    "execution_error": compact(turn.get("execution_error")),
                })
            cases.append({
                "task_id": task_id,
                "environment": compact(environment),
                "correct": bool(record.get("correct")),
                "legal": bool(record.get("legal")),
                "failure_type": record.get("failure_type"),
                "gold_sample": compact(record.get("gold_sample")),
                "pred_sample": compact(record.get("pred_sample")),
                "outcome": compact(record.get("outcome")),
                "error_events": compact(record.get("error_events") or []),
                "score": scores.get(task_id),
                "turns": turns,
            })
            wanted.remove(task_id)
            if not wanted:
                break
    if wanted:
        raise ValueError(f"missing requested task ids: {sorted(wanted)}")
    cases.sort(key=lambda item: int((item.get("score") or {}).get("position", 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
