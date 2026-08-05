#!/usr/bin/env python3
"""Quantify model-visible context behavior in version24 fixed-200 trajectories.

This audit reads only recorded model inputs/actions and harness-owned resident state. It does not
open a database, execute gold SQL, or reinterpret verifier answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any


SRC = Path(__file__).resolve().parents[1]
for dependency in (SRC / "sft", SRC / "harness"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from protocol import _compact_state_columns  # noqa: E402


STATE_MARKER = "CURRENT ENVIRONMENT STATE\n"
ERROR_MARKER = "\n\nLAST TOOL ERROR\n"
PERCEPTION_TOOLS = {"describe_table", "inspect_column", "read_subtable", "inspect_rows"}


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records.append(json.loads(line))
    return records


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_state(model_input: list[dict[str, Any]]) -> tuple[dict[str, Any], int, int]:
    for message in reversed(model_input):
        content = message.get("content")
        if not isinstance(content, str) or STATE_MARKER not in content:
            continue
        state_and_error = content.rsplit(STATE_MARKER, 1)[1]
        if ERROR_MARKER in state_and_error:
            state_text, error_text = state_and_error.split(ERROR_MARKER, 1)
        else:
            state_text, error_text = state_and_error, ""
        state = json.loads(state_text)
        if not isinstance(state, dict):
            raise ValueError("resident state is not a JSON object")
        state_section_chars = len(STATE_MARKER) + len(state_text)
        if error_text:
            state_section_chars += len(ERROR_MARKER) + len(error_text)
        return state, state_section_chars, len(error_text)
    return {"plan": [], "tables": {}, "values": {}}, 0, 0


def context_stats(turn: dict[str, Any]) -> dict[str, Any]:
    model_input = turn.get("model_input") or []
    state, state_section_chars, error_chars = extract_state(model_input)
    system_chars = sum(
        len(message.get("content", ""))
        for message in model_input
        if message.get("role") == "system"
    )
    assistant_history_chars = sum(
        len(message.get("content", ""))
        for message in model_input
        if message.get("role") == "assistant"
    )
    total_chars = sum(len(message.get("content", "")) for message in model_input)
    return {
        "total_chars": total_chars,
        "system_chars": system_chars,
        "assistant_history_chars": assistant_history_chars,
        "state_section_chars": state_section_chars,
        "last_error_chars": error_chars,
        "history_pairs": sum(message.get("role") == "assistant" for message in model_input),
        "state": state,
    }


def canonical_action(turn: dict[str, Any]) -> str | None:
    parsed = turn.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tool"), str):
        return None
    return compact_json({"tool": parsed["tool"], "arguments": parsed.get("arguments") or {}})


def evicted_unique_read_count(turns: list[dict[str, Any]]) -> int:
    """Count unique per-table reads beyond EnvironmentState's retained latest four."""
    reads_by_table: dict[str, set[str]] = defaultdict(set)
    for turn in turns:
        parsed = turn.get("parsed")
        if not isinstance(parsed, dict) or parsed.get("tool") not in {
            "read_subtable", "inspect_rows"
        }:
            continue
        args = parsed.get("arguments") or {}
        table = args.get("table")
        if not isinstance(table, str):
            continue
        comparable = {
            "columns": list(args.get("columns") or []),
            "conditions": args.get("conditions"),
            "order_by": list(args.get("order_by") or []),
            "offset": args.get("offset") or 0,
            "limit": args.get("limit"),
        }
        reads_by_table[table].add(compact_json(comparable))
    return sum(max(0, len(reads) - 4) for reads in reads_by_table.values())


def duplicate_read_count(state: dict[str, Any]) -> int:
    duplicate_count = 0
    for entry in (state.get("tables") or {}).values():
        if not isinstance(entry, dict):
            continue
        seen: set[str] = set()
        for raw_read in entry.get("reads") or []:
            if not isinstance(raw_read, dict):
                continue
            read = deepcopy(raw_read)
            read.pop("from_step", None)
            read.pop("equivalent_from_steps", None)
            read["columns"] = list(read.get("columns") or [])
            read["conditions"] = deepcopy(read.get("conditions"))
            read["order_by"] = list(read.get("order_by") or [])
            read["offset"] = read.get("offset") or 0
            if read.get("note") is None and read.get("limit") is None:
                read["limit"] = 20
            key = compact_json(read)
            if key in seen:
                duplicate_count += 1
            seen.add(key)
    return duplicate_count


def derived_dependencies(state: dict[str, Any], table_name: str) -> set[str]:
    tables = state.get("tables") or {}
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        entry = tables.get(name)
        if not isinstance(entry, dict):
            return
        derivation = entry.get("derivation")
        if not isinstance(derivation, dict):
            return
        for item in derivation.get("inputs") or []:
            if not isinstance(item, dict) or item.get("kind") != "table":
                continue
            ref = item.get("ref")
            if isinstance(ref, str):
                visit(ref)

    visit(table_name)
    return visited


def state_component_chars(state: dict[str, Any], field: str) -> int:
    """Return exact compact-JSON characters removed by dropping one resident component."""
    original_chars = len(compact_json(state))
    stripped = deepcopy(state)
    if field in {"plan", "values"}:
        stripped[field] = [] if field == "plan" else {}
    else:
        for entry in (stripped.get("tables") or {}).values():
            if isinstance(entry, dict):
                entry.pop(field, None)
    return original_chars - len(compact_json(stripped))


def final_state_stats(record: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    tables = state.get("tables") or {}
    derived = {
        name
        for name, entry in tables.items()
        if isinstance(entry, dict) and entry.get("kind") != "source"
    }
    zero_filters = {
        name
        for name, entry in tables.items()
        if isinstance(entry, dict)
        and entry.get("row_count") == 0
        and isinstance(entry.get("derivation"), dict)
        and entry["derivation"].get("operator") == "condition_filter"
    }
    reads = sum(
        len(entry.get("reads") or [])
        for entry in tables.values()
        if isinstance(entry, dict)
    )
    read_rows = sum(
        len(read.get("rows") or [])
        for entry in tables.values()
        if isinstance(entry, dict)
        for read in (entry.get("reads") or [])
        if isinstance(read, dict)
    )

    terminal_table = None
    turns = record.get("turns") or []
    if turns:
        parsed = turns[-1].get("parsed")
        if isinstance(parsed, dict) and parsed.get("tool") == "answer_from_context":
            evidence = (parsed.get("arguments") or {}).get("evidence")
            if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
                terminal_table = evidence["table"]
    reachable = derived_dependencies(state, terminal_table) if terminal_table else set()
    reachable_derived = derived & reachable
    unreachable_derived = derived - reachable_derived if terminal_table else set()
    unreachable_chars = sum(
        len(compact_json(tables[name])) for name in unreachable_derived
    )
    state_json_chars = len(compact_json(state))
    version39_compacted_state_chars = len(compact_json(_compact_state_columns(state)))
    return {
        "resident_tables": len(tables),
        "derived_handles": len(derived),
        "zero_row_filter_handles": len(zero_filters),
        "resident_reads": reads,
        "resident_read_rows": read_rows,
        "equivalent_duplicate_reads": duplicate_read_count(state),
        "terminal_evidence_table": terminal_table,
        "reachable_derived_handles": len(reachable_derived),
        "unreachable_derived_handles": len(unreachable_derived),
        "unreachable_derived_chars": unreachable_chars,
        "state_json_chars": state_json_chars,
        "version39_compacted_state_chars": version39_compacted_state_chars,
        "version39_compaction_saved_chars": (
            state_json_chars - version39_compacted_state_chars
        ),
        "state_read_chars": state_component_chars(state, "reads"),
        "state_schema_chars": state_component_chars(state, "schema"),
        "state_inspected_column_chars": state_component_chars(
            state, "inspected_columns"
        ),
        "state_derivation_chars": state_component_chars(state, "derivation"),
        "state_plan_chars": state_component_chars(state, "plan"),
        "state_values_chars": state_component_chars(state, "values"),
    }


def task_row(record: dict[str, Any]) -> dict[str, Any]:
    turns = record.get("turns") or []
    per_turn = [context_stats(turn) for turn in turns]
    final_context = per_turn[-1] if per_turn else {
        "total_chars": 0,
        "system_chars": 0,
        "assistant_history_chars": 0,
        "state_section_chars": 0,
        "last_error_chars": 0,
        "history_pairs": 0,
        "state": {"plan": [], "tables": {}, "values": {}},
    }
    actions = [action for turn in turns if (action := canonical_action(turn))]
    action_counts = Counter(actions)
    repeated_actions = sum(count - 1 for count in action_counts.values() if count > 1)
    repeated_action_kinds = {
        json.loads(action)["tool"]
        for action, count in action_counts.items()
        if count > 1
    }
    prompt_tokens = int((record.get("usage") or {}).get("prompt_tokens", 0))
    row = {
        "example_index": record.get("example_index"),
        "trajectory_id": record.get("trajectory_id"),
        "correct": bool(record.get("correct")),
        "legal": bool(record.get("legal")),
        "steps": int(record.get("steps") or len(turns)),
        "errors": int(record.get("errors") or 0),
        "failure_type": record.get("failure_type"),
        "prompt_tokens": prompt_tokens,
        "initial_context_chars": per_turn[0]["total_chars"] if per_turn else 0,
        "final_context_chars": final_context["total_chars"],
        "max_context_chars": max((item["total_chars"] for item in per_turn), default=0),
        "final_state_section_chars": final_context["state_section_chars"],
        "final_assistant_history_chars": final_context["assistant_history_chars"],
        "final_history_pairs": final_context["history_pairs"],
        "max_history_pairs": max((item["history_pairs"] for item in per_turn), default=0),
        "turns_with_last_error": sum(item["last_error_chars"] > 0 for item in per_turn),
        "repeated_exact_actions": repeated_actions,
        "repeated_exact_action_tools": sorted(repeated_action_kinds),
        "repeated_perception_action": bool(repeated_action_kinds & PERCEPTION_TOOLS),
        "evicted_unique_reads": evicted_unique_read_count(turns),
    }
    row.update(final_state_stats(record, final_context["state"]))
    return row


def step_bucket(steps: int) -> str:
    if steps <= 4:
        return "1-4"
    if steps <= 8:
        return "5-8"
    if steps <= 12:
        return "9-12"
    if steps <= 16:
        return "13-16"
    return "17+"


def numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 2),
        "max": max(values),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_rows = [row for row in rows if row["correct"]]
    failures = [row for row in rows if not row["correct"]]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[step_bucket(row["steps"])].append(row)
    bucket_summary = {}
    for label in ("1-4", "5-8", "9-12", "13-16", "17+"):
        group = buckets[label]
        correct = sum(row["correct"] for row in group)
        bucket_summary[label] = {
            "tasks": len(group),
            "correct": correct,
            "accuracy": round(correct / len(group), 4) if group else None,
            "mean_final_context_chars": round(
                mean(row["final_context_chars"] for row in group), 2
            ) if group else 0,
        }
    def outcome_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tasks": len(group),
            "mean_steps": round(mean(row["steps"] for row in group), 2),
            "mean_final_context_chars": round(
                mean(row["final_context_chars"] for row in group), 2
            ),
            "mean_final_state_chars": round(
                mean(row["final_state_section_chars"] for row in group), 2
            ),
            "mean_derived_handles": round(
                mean(row["derived_handles"] for row in group), 2
            ),
            "mean_unreachable_derived_handles": round(
                mean(row["unreachable_derived_handles"] for row in group), 2
            ),
            "tasks_with_unreachable_derived_handles": sum(
                row["unreachable_derived_handles"] > 0 for row in group
            ),
            "tasks_with_repeated_exact_actions": sum(
                row["repeated_exact_actions"] > 0 for row in group
            ),
            "tasks_with_zero_row_filter_handles": sum(
                row["zero_row_filter_handles"] > 0 for row in group
            ),
        }

    return {
        "all_tasks": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "failures": len(failures),
        "outcome_context_comparison": {
            "correct": outcome_metrics(correct_rows),
            "failure": outcome_metrics(failures),
        },
        "accuracy_by_episode_length": bucket_summary,
        "failure_steps": numeric_summary([row["steps"] for row in failures]),
        "failure_prompt_tokens": numeric_summary([row["prompt_tokens"] for row in failures]),
        "failure_final_context_chars": numeric_summary(
            [row["final_context_chars"] for row in failures]
        ),
        "failure_final_state_chars": numeric_summary(
            [row["final_state_section_chars"] for row in failures]
        ),
        "failure_state_component_mean_chars": {
            field: round(mean(row[field] for row in failures), 2)
            for field in (
                "state_read_chars",
                "state_schema_chars",
                "state_inspected_column_chars",
                "state_derivation_chars",
                "state_plan_chars",
                "state_values_chars",
            )
        },
        "failures_steps_ge_10": sum(row["steps"] >= 10 for row in failures),
        "failures_steps_ge_15": sum(row["steps"] >= 15 for row in failures),
        "failures_with_repeated_exact_actions": sum(
            row["repeated_exact_actions"] > 0 for row in failures
        ),
        "failures_with_repeated_perception": sum(
            row["repeated_perception_action"] for row in failures
        ),
        "failures_with_duplicate_resident_reads": sum(
            row["equivalent_duplicate_reads"] > 0 for row in failures
        ),
        "failures_with_evicted_unique_reads": sum(
            row["evicted_unique_reads"] > 0 for row in failures
        ),
        "failures_with_zero_row_filter_handles": sum(
            row["zero_row_filter_handles"] > 0 for row in failures
        ),
        "failures_changed_by_version39_state_compaction": sum(
            row["version39_compaction_saved_chars"] > 0 for row in failures
        ),
        "failure_version39_compaction_saved_chars": numeric_summary(
            [row["version39_compaction_saved_chars"] for row in failures]
        ),
        "failures_with_unreachable_derived_handles": sum(
            row["unreachable_derived_handles"] > 0 for row in failures
        ),
        "failure_unreachable_derived_handles": numeric_summary(
            [row["unreachable_derived_handles"] for row in failures if row["legal"]]
        ),
        "context_overflow_failures": sum(
            row["failure_type"] == "context_overflow" for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if len(records) != 200:
        raise ValueError(f"expected fixed-200 records, found {len(records)}")
    rows = [task_row(record) for record in records]
    rows.sort(key=lambda row: int(row["example_index"]))
    failures = [row for row in rows if not row["correct"]]
    if len(failures) != 55:
        raise ValueError(f"expected 55 failures, found {len(failures)}")
    summary = build_summary(rows)
    summary["api_context_retry_count"] = sum(
        int((record.get("usage") or {}).get("api_context_retries", 0))
        for record in records
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "version24_context_management_summary.json"
    all_task_path = args.output_dir / "version24_all_context_metrics.jsonl"
    task_path = args.output_dir / "version24_failure_context_metrics.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path, selected_rows in ((all_task_path, rows), (task_path, failures)):
        with path.open("w", encoding="utf-8") as target:
            for row in selected_rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary, "summary_path": str(summary_path),
                      "all_metrics_path": str(all_task_path),
                      "failure_metrics_path": str(task_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
