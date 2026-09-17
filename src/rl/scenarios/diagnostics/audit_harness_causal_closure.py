#!/usr/bin/env python3
"""Audit Harness causal-closure coverage without using Gold SQL or model reasoning.

This diagnostic has two deliberately separate notions of coverage:

* resolved artifact closure: an action is on the backward relation lineage of the
  terminal evidence handle, using only Harness-authored derivation metadata;
* observation support candidates: a later literal happens to equal an observed
  value.  These are *not* treated as dependencies because the value may be task
  supplied or model guessed.  They are emitted for manual review only.

The output is read-only evidence for reward design.  It does not modify rollout
records and never emits reward values.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OBSERVER_TOOLS = {"plan", "describe_table", "inspect_column", "read_subtable"}
PRODUCER_TOOLS = {
    "condition_filter", "project", "join_tables", "group_aggregate",
    "scalar_compute", "extreme_value_select", "set_op",
}


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _step_number(value: str) -> int | None:
    if not value.startswith("step_"):
        return None
    try:
        return int(value[5:]) - 1
    except ValueError:
        return None


def _terminal_index(turns: list[dict[str, Any]]) -> tuple[int | None, str | None]:
    result: tuple[int | None, str | None] = (None, None)
    for index, turn in enumerate(turns):
        parsed = turn.get("parsed") or {}
        if parsed.get("tool") != "answer_from_context":
            continue
        evidence = (parsed.get("arguments") or {}).get("evidence") or {}
        if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
            result = (index, evidence["table"])
    return result


def _literal_values(value: Any, *, field: str | None = None) -> list[Any]:
    """Collect only argument values, not tool names, column names, or operators."""
    if isinstance(value, dict):
        result: list[Any] = []
        for key, child in value.items():
            if key in {"value_ref", "table", "base", "left", "right", "tables", "evidence", "column", "op", "as", "result_name"}:
                continue
            result.extend(_literal_values(child, field=key))
        return result
    if isinstance(value, list):
        return [item for child in value for item in _literal_values(child, field=field)]
    if field in {"value", "low", "high", "values"} and (isinstance(value, (str, int, float, bool)) or value is None):
        return [value]
    return []


def _observed_values(turn: dict[str, Any]) -> list[Any]:
    output = turn.get("tool_output") or {}
    values: list[Any] = []
    if isinstance(output.get("rows"), list):
        for row in output["rows"]:
            if isinstance(row, list):
                values.extend(row)
    frequent = output.get("frequent_values")
    if isinstance(frequent, list):
        values.extend(frequent)
    return values


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _artifact_graph(turns: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, set[int]], dict[int, dict[str, Any]]]:
    producers: dict[str, int] = {}
    parents: dict[int, set[int]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for index, turn in enumerate(turns):
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        output = turn.get("tool_output") or {}
        handle = output.get("table")
        if tool not in PRODUCER_TOOLS or not isinstance(handle, str):
            continue
        producers[handle] = index
        derivation = output.get("derivation")
        metadata[index] = {
            "tool": tool,
            "handle": handle,
            "has_derivation": isinstance(derivation, dict),
            "input_handles": [],
            "input_steps": [],
            "constant_inputs": [],
        }
        if not isinstance(derivation, dict):
            continue
        for item in derivation.get("inputs") or []:
            if not isinstance(item, dict):
                continue
            ref = item.get("ref")
            if isinstance(ref, str):
                metadata[index]["input_handles"].append(ref)
                if ref in producers:
                    parents.setdefault(index, set()).add(producers[ref])
            if item.get("kind") == "constant":
                metadata[index]["constant_inputs"].append(item.get("value"))
        for node in _walk((parsed.get("arguments") or {})):
            if isinstance(node, dict) and isinstance(node.get("value_ref"), str):
                parent = _step_number(node["value_ref"])
                if parent is not None and parent < index:
                    metadata[index]["input_steps"].append(parent)
                    parents.setdefault(index, set()).add(parent)
    return producers, parents, metadata


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    turns = [turn for turn in row.get("turns") or [] if isinstance(turn, dict)]
    terminal_index, terminal_handle = _terminal_index(turns)
    producers, parents, metadata = _artifact_graph(turns)
    closure: set[int] = set()
    if terminal_handle in producers:
        pending = [producers[terminal_handle]]
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            pending.extend(parents.get(current, ()))

    observer_indices = [
        index for index, turn in enumerate(turns)
        if (turn.get("parsed") or {}).get("tool") in OBSERVER_TOOLS
    ]
    observed_by_index = {
        index: {_value_key(value) for value in _observed_values(turns[index])}
        for index in observer_indices
    }
    question_text = " ".join(
        str(row.get(key) or "") for key in ("question", "external_knowledge")
    )
    candidates: list[dict[str, Any]] = []
    for later_index in range(len(turns)):
        parsed = turns[later_index].get("parsed") or {}
        args = parsed.get("arguments") or {}
        later_values = {_value_key(value) for value in _literal_values(args)}
        if not later_values:
            continue
        for observer_index, values in observed_by_index.items():
            if observer_index >= later_index:
                continue
            overlap = later_values & values
            if not overlap:
                continue
            matched = [json.loads(value) for value in sorted(overlap)]
            candidates.append({
                "observer_turn": observer_index,
                "later_turn": later_index,
                "observer_tool": (turns[observer_index].get("parsed") or {}).get("tool"),
                "later_tool": parsed.get("tool"),
                "matched_values": matched[:8],
                "task_text_overlap": [value for value in matched if str(value) and str(value) in question_text],
                "status": "candidate_manual_review",
            })

    explicit_refs: list[dict[str, Any]] = []
    for later_index, turn in enumerate(turns):
        for node in _walk(((turn.get("parsed") or {}).get("arguments") or {})):
            if not isinstance(node, dict) or not isinstance(node.get("value_ref"), str):
                continue
            parent = _step_number(node["value_ref"])
            if parent is None or parent >= later_index:
                continue
            explicit_refs.append({
                "from_turn": later_index,
                "to_turn": parent,
                "to_tool": (turns[parent].get("parsed") or {}).get("tool"),
                "to_observer": parent in observer_indices,
                "later_action_success": (
                    turns[later_index].get("execution_error_type") is None
                    and bool(turns[later_index].get("tool_output"))
                ),
                "status": "resolved_explicit_value_ref",
            })

    observation_support_edges: list[dict[str, Any]] = []
    prior_observation_values: Counter[str] = Counter()
    for observer_index in observer_indices:
        output = turns[observer_index].get("tool_output") or {}
        values = []
        for observed_row in output.get("rows") or []:
            if isinstance(observed_row, list):
                values.extend(observed_row)
        values.extend(output.get("frequent_values") or [])
        for value in values:
            prior_observation_values[_value_key(value)] += 1
    for candidate in candidates:
        observer_index = int(candidate["observer_turn"])
        later_index = int(candidate["later_turn"])
        if candidate["observer_tool"] != "read_subtable":
            continue
        if later_index not in closure or later_index >= len(turns):
            continue
        later_turn = turns[later_index]
        if later_turn.get("execution_error_type") is not None or not later_turn.get("tool_output"):
            continue
        for value in candidate["matched_values"]:
            key = _value_key(value)
            if prior_observation_values[key] != 1:
                continue
            if str(value) and str(value) in question_text:
                continue
            observation_support_edges.append({
                **candidate,
                "matched_value": value,
                "status": "conservative_support_candidate",
            })

    unknown_literal_actions = []
    constant_input_actions = []
    for index, turn in enumerate(turns):
        tool = (turn.get("parsed") or {}).get("tool")
        if tool not in PRODUCER_TOOLS:
            continue
        args = (turn.get("parsed") or {}).get("arguments") or {}
        ordinary = [value for value in _literal_values(args) if value not in {None, True, False, ""}]
        has_value_ref = any(
            isinstance(node, dict) and isinstance(node.get("value_ref"), str)
            for node in _walk(args)
        )
        if ordinary and not has_value_ref:
            unknown_literal_actions.append({
                "turn": index,
                "tool": tool,
                "values": ordinary[:8],
                "status": "unknown_or_task_constant",
            })
        constant_inputs = (metadata.get(index) or {}).get("constant_inputs") or []
        if constant_inputs:
            constant_input_actions.append({
                "turn": index,
                "tool": tool,
                "values": constant_inputs[:8],
                "status": "harness_explicit_constant",
            })

    return {
        "trajectory_id": row.get("trajectory_id"),
        "example_index": row.get("example_index"),
        "correct": bool(row.get("correct")),
        "legal": bool(row.get("legal")),
        "failure_type": row.get("failure_type"),
        "turn_count": len(turns),
        "terminal_index": terminal_index,
        "terminal_handle": terminal_handle,
        "terminal_resolved": terminal_handle in producers if terminal_handle else False,
        "producer_count": len(producers),
        "artifact_closure_turns": sorted(closure),
        "artifact_closure_count": len(closure),
        "observer_turns": observer_indices,
        "observer_count": len(observer_indices),
        "observer_candidate_edges": candidates,
        "conservative_observation_support_edges": observation_support_edges,
        "explicit_value_refs": explicit_refs,
        "unknown_literal_actions": unknown_literal_actions,
        "constant_input_actions": constant_input_actions,
        "explicit_value_ref_count": len(explicit_refs),
        "explicit_value_ref_to_observer_count": sum(
            int(ref["to_observer"]) for ref in explicit_refs
        ),
        "explicit_value_ref_to_observer_success_count": sum(
            int(ref["to_observer"] and ref["later_action_success"])
            for ref in explicit_refs
        ),
        "candidate_decision_edge_count": len(candidates),
        "metadata": metadata,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pct(value: int, total: int) -> float:
        return round(100.0 * value / total, 3) if total else 0.0

    total = len(rows)
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    closure = [row for row in rows if row["terminal_resolved"]]
    obs = [row for row in rows if row["observer_count"]]
    resolved_refs = [row for row in rows if row["explicit_value_ref_count"]]
    candidates = [row for row in rows if row["candidate_decision_edge_count"]]
    return {
        "trajectories": total,
        "correct": len(correct),
        "wrong": len(wrong),
        "terminal_resolved": len(closure),
        "terminal_resolved_pct": pct(len(closure), total),
        "terminal_resolved_correct": sum(row["correct"] for row in closure),
        "terminal_resolved_wrong": sum(not row["correct"] for row in closure),
        "observer_trajectories": len(obs),
        "explicit_value_ref_trajectories": len(resolved_refs),
        "candidate_observation_decision_trajectories": len(candidates),
        "explicit_value_ref_edges": sum(row["explicit_value_ref_count"] for row in rows),
        "explicit_value_ref_to_observer_edges": sum(
            row["explicit_value_ref_to_observer_count"] for row in rows
        ),
        "explicit_value_ref_to_observer_success_edges": sum(
            row["explicit_value_ref_to_observer_success_count"] for row in rows
        ),
        "candidate_edges_manual_only": sum(row["candidate_decision_edge_count"] for row in rows),
        "conservative_support_edges_manual_only": sum(
            len(row["conservative_observation_support_edges"]) for row in rows
        ),
        "conservative_support_trajectories_manual_only": sum(
            bool(row["conservative_observation_support_edges"]) for row in rows
        ),
        "unknown_literal_actions": sum(len(row["unknown_literal_actions"]) for row in rows),
        "harness_explicit_constant_actions": sum(len(row["constant_input_actions"]) for row in rows),
        "artifact_closure_turns": sum(row["artifact_closure_count"] for row in rows),
        "parsed_turns": sum(row["turn_count"] for row in rows),
        "resolved_artifact_action_fraction": pct(
            sum(row["artifact_closure_count"] for row in rows),
            sum(row["turn_count"] for row in rows),
        ),
        "correct_summary": {
            "trajectories": len(correct),
            "terminal_resolved": sum(row["terminal_resolved"] for row in correct),
            "observer_trajectories": sum(row["observer_count"] > 0 for row in correct),
            "explicit_value_ref_trajectories": sum(row["explicit_value_ref_count"] > 0 for row in correct),
            "candidate_decision_edge_trajectories": sum(row["candidate_decision_edge_count"] > 0 for row in correct),
            "conservative_support_trajectories_manual_only": sum(bool(row["conservative_observation_support_edges"]) for row in correct),
            "unknown_literal_actions": sum(len(row["unknown_literal_actions"]) for row in correct),
            "harness_explicit_constant_actions": sum(len(row["constant_input_actions"]) for row in correct),
        },
        "wrong_summary": {
            "trajectories": len(wrong),
            "terminal_resolved": sum(row["terminal_resolved"] for row in wrong),
            "observer_trajectories": sum(row["observer_count"] > 0 for row in wrong),
            "explicit_value_ref_trajectories": sum(row["explicit_value_ref_count"] > 0 for row in wrong),
            "candidate_decision_edge_trajectories": sum(row["candidate_decision_edge_count"] > 0 for row in wrong),
            "conservative_support_trajectories_manual_only": sum(bool(row["conservative_observation_support_edges"]) for row in wrong),
            "unknown_literal_actions": sum(len(row["unknown_literal_actions"]) for row in wrong),
            "harness_explicit_constant_actions": sum(len(row["constant_input_actions"]) for row in wrong),
        },
    }


def select_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    bins = [
        lambda row: row["correct"] and row["terminal_resolved"] and row["candidate_decision_edge_count"] > 0,
        lambda row: row["correct"] and row["terminal_resolved"] and row["observer_count"] > 0 and not row["candidate_decision_edge_count"],
        lambda row: row["correct"] and row["terminal_resolved"] and row["unknown_literal_actions"],
        lambda row: (not row["correct"]) and row["legal"] and row["terminal_resolved"],
        lambda row: row["correct"] and not row["terminal_resolved"],
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for predicate in bins:
        for row in rows:
            trajectory_id = str(row.get("trajectory_id"))
            if trajectory_id in used or not predicate(row):
                continue
            selected.append(row)
            used.add(trajectory_id)
            if len(selected) >= limit:
                return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-limit", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    all_rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}
    for path in args.rollouts:
        run_name = f"{path.parent.name}_{path.stem}"
        rows = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        rows.append(audit_row(json.loads(line)))
                    except Exception as exc:
                        raise SystemExit(f"{path}:{line_number}: {type(exc).__name__}: {exc}") from exc
        run_summaries[run_name] = summarize(rows)
        all_rows.extend({"run": run_name, **row} for row in rows)
    output = {
        "schema_version": "harness-causal-closure-audit-v1",
        "policy": {
            "uses_gold_sql": False,
            "uses_model_reasoning": False,
            "candidate_edges_are_reward_eligible": False,
            "unknown_literals_are_rewarded": False,
        },
        "runs": run_summaries,
        "overall": summarize(all_rows),
        "manual_cases": select_cases(all_rows, args.manual_limit),
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "overall": output["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
