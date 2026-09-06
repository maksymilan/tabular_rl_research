#!/usr/bin/env python3
"""Analyze the expanded human audit and conservative critical-error guards."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        rank = ((index + 1) + stop) / 2
        for original, _ in ordered[index:stop]:
            result[original] = rank
        index = stop
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _observed_row_count(output: Any) -> int | None:
    if not isinstance(output, dict):
        return None
    if isinstance(output.get("row_count"), int):
        return int(output["row_count"])
    if isinstance(output.get("rows"), list):
        return len(output["rows"])
    return None


def _gold_limit(sql: str) -> int | None:
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    limit = tree.args.get("limit")
    expression = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(expression, exp.Literal) and not expression.is_string:
        try:
            value = int(str(expression.this))
            return value if value > 0 else None
        except ValueError:
            return None
    return None


def local_rank_scope_violation_from_steps(
    steps: list[dict[str, Any]], *, gold_sql: str,
) -> dict[str, Any] | None:
    """Detect one narrow, forward-observed rank-scope violation.

    Gold applies a literal LIMIT k to the completed query, while the trajectory applies top_k=k
    first, directly uses that handle as the base of a join, and cites a forward chain that still
    contains more than k rows.  The implementation follows only explicit tool arguments and
    outputs in execution order; it does not construct or score a backward dependency slice.
    """
    terminal: str | None = None
    gold_limit = _gold_limit(gold_sql)
    if gold_limit is None:
        return None
    states: dict[str, dict[str, Any]] = {}
    unary_tools = {
        "condition_filter",
        "project",
        "group_aggregate",
        "extreme_value_select",
    }
    for step in steps:
        call = step.get("tool_call") or {}
        tool = call.get("tool")
        arguments = call.get("arguments") or {}
        output = step.get("tool_output") or {}
        if tool == "answer_from_context":
            evidence = arguments.get("evidence") or {}
            if isinstance(evidence.get("table"), str):
                terminal = evidence["table"]
            continue
        handle = output.get("table")
        rows = _observed_row_count(output)
        if tool == "extreme_value_select" and isinstance(handle, str):
            top_k = arguments.get("top_k")
            if isinstance(top_k, int) and top_k == gold_limit:
                states[handle] = {
                    "top_handle": handle,
                    "current_handle": handle,
                    "top_k": top_k,
                    "current_rows": rows,
                    "join_handle": None,
                    "join_rows": None,
                }

        for state in list(states.values()):
            current = state["current_handle"]
            follows_current = False
            if tool == "join_tables" and arguments.get("base") == current:
                follows_current = True
            elif tool in unary_tools and arguments.get("table") == current:
                follows_current = True
            if not follows_current or not isinstance(handle, str):
                continue
            state["current_handle"] = handle
            state["current_rows"] = rows
            if tool == "join_tables" and rows is not None and rows > gold_limit:
                state["join_handle"] = handle
                state["join_rows"] = rows

    if terminal is None:
        return None
    violations = []
    for state in states.values():
        terminal_rows = state["current_rows"]
        if (
            state["current_handle"] == terminal
            and state["join_handle"] is not None
            and terminal_rows is not None
            and terminal_rows > gold_limit
        ):
            violations.append({
                "handle": state["top_handle"],
                "join_handle": state["join_handle"],
                "join_rows": state["join_rows"],
                "top_k": state["top_k"],
                "gold_limit": gold_limit,
                "terminal_rows": terminal_rows,
            })
    if not violations:
        return None
    strongest = min(violations, key=lambda item: item["top_k"] / item["terminal_rows"])
    return {
        **strongest,
        "ratio": strongest["top_k"] / strongest["terminal_rows"],
        "terminal": terminal,
    }


# Kept as a compatibility name for the frozen diagnostic scripts created in this audit.
rank_expansion_from_steps = local_rank_scope_violation_from_steps


def rank_expansion_diagnostic(case: dict[str, Any]) -> dict[str, Any] | None:
    steps = []
    for turn in case.get("turns") or []:
        steps.append({
            "tool_call": {"tool": turn.get("tool"), "arguments": turn.get("arguments") or {}},
            "tool_output": turn.get("tool_output"),
        })
    return local_rank_scope_violation_from_steps(
        steps,
        gold_sql=str((case.get("environment") or {}).get("gold_sql") or ""),
    )


def _value_score(score: dict[str, Any]) -> float | None:
    details = ((score.get("terminal_overlap") or {}).get("per_category") or {}).get("value")
    return float(details["score"]) if isinstance(details, dict) and details.get("score") is not None else None


def analyze(
    selections: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_by_id = {str(item["task_id"]): item for item in selections}
    case_by_id = {str(item["task_id"]): item for item in cases}
    annotation_by_id = {str(item["task_id"]): item for item in annotations}
    ids = set(selection_by_id)
    if ids != set(case_by_id) or ids != set(annotation_by_id):
        raise ValueError("selection, casebook, and annotation task ids must match exactly")

    rows = []
    for task_id in sorted(ids):
        selection = selection_by_id[task_id]
        case = case_by_id[task_id]
        annotation = annotation_by_id[task_id]
        current = float(selection["score"])
        value = _value_score(case.get("score") or {})
        value_factor = 1.0 if value is None else 0.5 + 0.5 * value
        rank_expansion = rank_expansion_diagnostic(case)
        rank_factor = 1.0 if rank_expansion is None else 0.5 + 0.5 * rank_expansion["ratio"]
        value_guarded = current * value_factor
        rank_soft = current * rank_factor
        rank_cap = (
            current if rank_expansion is None
            else min(current, float(rank_expansion["ratio"]))
        )
        guarded = value_guarded * rank_factor
        rows.append({
            "task_id": task_id,
            "cell": selection["cell"],
            "correct": bool(selection["correct"]),
            "current_score": current,
            "value_score": value,
            "value_factor": value_factor,
            "rank_expansion": rank_expansion,
            "rank_factor": rank_factor,
            "value_guarded_score_diagnostic_only": value_guarded,
            "rank_soft_score_diagnostic_only": rank_soft,
            "rank_cap_score_candidate": rank_cap,
            "guarded_score_diagnostic_only": guarded,
            "severity": int(annotation["severity"]),
            "kind": annotation["kind"],
            "judgment": annotation["judgment"],
            "note": annotation["note"],
        })

    def correlation(group: list[dict[str, Any]], key: str) -> float | None:
        return _spearman(
            [float(item[key]) for item in group],
            [1.0 - int(item["severity"]) / 3 for item in group],
        )

    incorrect = [row for row in rows if not row["correct"]]
    summary = {
        "schema_version": "expanded-terminal-state-quality-human-audit-v1",
        "cases": len(rows),
        "severity_counts": dict(sorted(Counter(row["severity"] for row in rows).items())),
        "severity_by_cell": {
            cell: {
                "count": len(group),
                "mean": statistics.fmean(row["severity"] for row in group),
                "counts": dict(sorted(Counter(row["severity"] for row in group).items())),
            }
            for cell in sorted({row["cell"] for row in rows})
            if (group := [row for row in rows if row["cell"] == cell])
        },
        "failure_kinds": dict(Counter(row["kind"] for row in rows).most_common()),
        "score_vs_human_quality_spearman": {
            "all_current": correlation(rows, "current_score"),
            "all_value_guarded_diagnostic": correlation(
                rows, "value_guarded_score_diagnostic_only"
            ),
            "all_rank_soft_diagnostic": correlation(rows, "rank_soft_score_diagnostic_only"),
            "all_rank_cap_candidate": correlation(rows, "rank_cap_score_candidate"),
            "all_value_plus_rank_diagnostic": correlation(
                rows, "guarded_score_diagnostic_only"
            ),
            "incorrect_current": correlation(incorrect, "current_score"),
            "incorrect_value_guarded_diagnostic": correlation(
                incorrect, "value_guarded_score_diagnostic_only"
            ),
            "incorrect_rank_soft_diagnostic": correlation(
                incorrect, "rank_soft_score_diagnostic_only"
            ),
            "incorrect_rank_cap_candidate": correlation(
                incorrect, "rank_cap_score_candidate"
            ),
            "incorrect_value_plus_rank_diagnostic": correlation(
                incorrect, "guarded_score_diagnostic_only"
            ),
        },
        "critical_guards": {
            "value_guard_applied": sum(row["value_score"] is not None for row in rows),
            "rank_expansion_detected": sum(row["rank_expansion"] is not None for row in rows),
            "value_formula_rejected": (
                "current * (0.5+0.5*value_score if value exists else 1)"
            ),
            "rank_candidate": (
                "min(current, top_k/terminal_rows) on a directly observed unresolved "
                "post-rank join expansion; current otherwise"
            ),
            "status": "diagnostic_only_not_reward",
        },
        "notes": [
            "severity 0=no substantive error or Gold/question conflict; 1=minor; 2=moderate; 3=severe",
            "the cohort is stratified for failure discovery and is not an unbiased prevalence sample",
            "human labels are used only to validate diagnostics, never as a training reward",
        ],
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    casebook = json.loads(args.casebook.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    summary, rows = analyze(selection, casebook, annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expanded_case_manual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "expanded_case_manual_audit_scored.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
