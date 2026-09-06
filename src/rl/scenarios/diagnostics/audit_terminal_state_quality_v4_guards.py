#!/usr/bin/env python3
"""Ablate value overlap and a local post-rank-cardinality guard over frozen v3 scores.

This is an offline diagnostic only.  It does not execute SQL, compare answer rows, call a model,
or update an optimizer.  Correctness labels are joined after the candidate score for evaluation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "rl")]

from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402
from rl.scenarios.diagnostics.analyze_expanded_quality_case_audit import (  # noqa: E402
    rank_expansion_from_steps,
)


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        rank_sum += ((index + 1 + stop) / 2) * sum(label for _, label in ordered[index:stop])
        index = stop
    return (rank_sum - len(positive) * (len(positive) + 1) / 2) / (
        len(positive) * len(negative)
    )


def _value_score(score: dict[str, Any]) -> float | None:
    details = ((score.get("terminal_overlap") or {}).get("per_category") or {}).get("value")
    return float(details["score"]) if isinstance(details, dict) and details.get("score") is not None else None


def audit(input_path: Path, scores_path: Path, output_dir: Path) -> dict[str, Any]:
    scores = {}
    with scores_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("score") is not None:
                scores[str(row["task_id"])] = row

    rows = []
    with input_path.open(encoding="utf-8") as source:
        for position, line in enumerate(source):
            raw = json.loads(line)
            task = dict(raw.get("environment") or {})
            task_id = str(task.get("task_id") or task.get("example_id"))
            prior = scores.get(task_id)
            if prior is None:
                continue
            record = dict(((raw.get("sample") or {}).get("audit_record")) or {})
            trajectory, exclusion = normalize_failure_record(record, task)
            if trajectory is None:
                raise ValueError(f"scored task {task_id} no longer normalizes: {exclusion}")
            steps = list(trajectory.get("steps") or [])
            current = float(prior["score"])
            value = _value_score(prior)
            value_factor = 1.0 if value is None else 0.5 + 0.5 * value
            rank_expansion = rank_expansion_from_steps(
                steps, gold_sql=str(task.get("gold_sql") or task.get("query") or "")
            )
            rank_factor = (
                1.0 if rank_expansion is None
                else 0.5 + 0.5 * float(rank_expansion["ratio"])
            )
            rank_cap = (
                current if rank_expansion is None
                else min(current, float(rank_expansion["ratio"]))
            )
            rows.append({
                "position": position,
                "task_id": task_id,
                "db_id": task.get("db_id"),
                "correct_validation_only": bool(prior.get("correct")),
                "current_score": current,
                "value_score": value,
                "value_factor": value_factor,
                "value_guarded_score": current * value_factor,
                "rank_expansion": rank_expansion,
                "rank_factor": rank_factor,
                "rank_soft_score": current * rank_factor,
                "rank_cap_score": rank_cap,
                "guarded_score": current * value_factor * rank_factor,
            })

    correct = [row for row in rows if row["correct_validation_only"]]
    incorrect = [row for row in rows if not row["correct_validation_only"]]

    def stats(group: list[dict[str, Any]], key: str) -> dict[str, Any]:
        values = [float(row[key]) for row in group]
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "at_or_above_0_75": sum(value >= 0.75 for value in values),
            "at_or_below_0_25": sum(value <= 0.25 for value in values),
        }

    summary = {
        "schema_version": "terminal-state-quality-v4-guard-ablation-v2",
        "records": len(rows),
        "correct_validation_only": len(correct),
        "incorrect_validation_only": len(incorrect),
        "score_inputs": [
            "v3 terminal semantic overlap", "v3 value overlap",
            "gold SQL LIMIT", "deterministic terminal artifact row_count",
        ],
        "explicitly_excluded_from_score": [
            "correctness", "answer rows", "answer similarity", "model reasoning",
            "tool type mean", "backward-slice coverage",
        ],
        "candidate_formula": (
            "Q_candidate = min(Q_v3, gold_limit/terminal_rows) only when an unresolved direct "
            "top-k -> base join expansion is observed; Q_v3 otherwise"
        ),
        "rejected_value_ablation": "Q_value = Q_v3 * (0.5 + 0.5*C_value)",
        "local_rank_scope_definition": (
            "Gold has literal LIMIT k; trajectory produces top_k=k; that exact handle is used "
            "as the base of a later join; the forward-produced terminal evidence still has n>k rows"
        ),
        "current": {
            "correct": stats(correct, "current_score"),
            "incorrect": stats(incorrect, "current_score"),
            "auc_correctness_validation_only": _auc(
                [row["current_score"] for row in correct],
                [row["current_score"] for row in incorrect],
            ),
        },
        "value_guarded_rejected": {
            "correct": stats(correct, "value_guarded_score"),
            "incorrect": stats(incorrect, "value_guarded_score"),
            "auc_correctness_validation_only": _auc(
                [row["value_guarded_score"] for row in correct],
                [row["value_guarded_score"] for row in incorrect],
            ),
        },
        "rank_soft_diagnostic": {
            "correct": stats(correct, "rank_soft_score"),
            "incorrect": stats(incorrect, "rank_soft_score"),
            "auc_correctness_validation_only": _auc(
                [row["rank_soft_score"] for row in correct],
                [row["rank_soft_score"] for row in incorrect],
            ),
        },
        "rank_cap_candidate": {
            "correct": stats(correct, "rank_cap_score"),
            "incorrect": stats(incorrect, "rank_cap_score"),
            "auc_correctness_validation_only": _auc(
                [row["rank_cap_score"] for row in correct],
                [row["rank_cap_score"] for row in incorrect],
            ),
        },
        "value_plus_rank_diagnostic": {
            "correct": stats(correct, "guarded_score"),
            "incorrect": stats(incorrect, "guarded_score"),
            "auc_correctness_validation_only": _auc(
                [row["guarded_score"] for row in correct],
                [row["guarded_score"] for row in incorrect],
            ),
        },
        "guard_activation": {
            "value_present": sum(row["value_score"] is not None for row in rows),
            "value_below_one": sum(
                row["value_score"] is not None and row["value_score"] < 1 for row in rows
            ),
            "rank_expansion": sum(row["rank_expansion"] is not None for row in rows),
            "rank_expansion_correct": sum(
                row["rank_expansion"] is not None and row["correct_validation_only"] for row in rows
            ),
            "rank_expansion_incorrect": sum(
                row["rank_expansion"] is not None and not row["correct_validation_only"] for row in rows
            ),
        },
        "decision": {
            "value_guard": "rejected_from_candidate_reward",
            "local_rank_cap": "retain_as_diagnostic_candidate_pending_same-task_rollouts",
        },
        "status": "diagnostic_only_not_reward",
        "model_calls": 0,
        "optimizer_updates": 0,
        "evaluation_set_records_used": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "guarded_trajectory_scores.jsonl").open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "guarded_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.scores, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
