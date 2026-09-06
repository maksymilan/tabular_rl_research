#!/usr/bin/env python3
"""Offline audit of v2 tool-state obligation quality on existing BIRD-train trajectories.

The score itself sees only Gold SQL, SQLite schema, and deterministic relation-derivation state.
Correctness and prior denotation-error measurements are joined only after scoring for validation.
No model, optimizer, or SQL-result comparison is invoked here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "rl")]

from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402
from tool_state_obligation_quality import score_tool_state_trajectory  # noqa: E402
from trajectory_semantic_ranker import load_sqlite_columns  # noqa: E402


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _describe(values: Iterable[float]) -> dict[str, Any]:
    retained = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not retained:
        return {"count": 0}
    return {
        "count": len(retained),
        "mean": statistics.fmean(retained),
        "std": statistics.pstdev(retained) if len(retained) > 1 else 0.0,
        "min": min(retained),
        "p10": _quantile(retained, 0.10),
        "p25": _quantile(retained, 0.25),
        "median": _quantile(retained, 0.50),
        "p75": _quantile(retained, 0.75),
        "p90": _quantile(retained, 0.90),
        "max": max(retained),
    }


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:stop])
        index = stop
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _rankdata(values: list[float]) -> list[float]:
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
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _prior_answer_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    result: dict[str, float] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            score = ((row.get("quality") or {}).get("answer_score"))
            if row.get("scored") and score is not None:
                result[str(row.get("task_id"))] = float(score)
    return result


def audit(input_path: Path, output_dir: Path, answer_audit: Path | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_answers = _prior_answer_scores(answer_audit)
    schema_cache: dict[str, dict[str, tuple[str, ...]]] = {}
    results: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as source:
        for position, line in enumerate(source):
            row = json.loads(line)
            task = dict(row.get("environment") or {})
            record = dict(((row.get("sample") or {}).get("audit_record")) or {})
            task_id = str(task.get("task_id") or task.get("example_id"))
            trajectory, exclusion = normalize_failure_record(record, task)
            identity = {
                "position": position,
                "task_id": task_id,
                "example_index": task.get("example_index"),
                "db_id": task.get("db_id"),
                "correct": bool(record.get("correct")),
                "legal": bool(record.get("legal")),
                "failure_type": record.get("failure_type"),
                "sft_admitted": bool((task.get("metadata") or {}).get("sft_admitted")),
            }
            if task.get("split") != "train" or trajectory is None:
                exclusions.append({**identity, "reason": exclusion or f"split:{task.get('split')}"})
                continue
            db_path = str(task["db_path"])
            if db_path not in schema_cache:
                schema_cache[db_path] = load_sqlite_columns(db_path)
            quality = score_tool_state_trajectory(
                gold_sql=str(task.get("gold_sql") or task.get("query") or ""),
                table_columns=schema_cache[db_path],
                steps=trajectory.get("steps") or [],
            )
            tools = [
                str((step.get("tool_call") or {}).get("tool"))
                for step in trajectory.get("steps") or []
                if isinstance((step.get("tool_call") or {}).get("tool"), str)
            ]
            results.append({
                **identity,
                "score": quality.score,
                "semantic_eligible": quality.semantic_eligible,
                "c_terminal": quality.c_terminal,
                "c_max": quality.c_max,
                "terminal_evidence": quality.terminal_evidence,
                "max_handle": quality.max_handle,
                "scoring_mode": quality.scoring_mode,
                "reasons": list(quality.reasons),
                "tools": tools,
                "terminal_overlap": quality.terminal_overlap,
                "answer_score_validation_only": prior_answers.get(task_id),
            })

    scored = [row for row in results if row["score"] is not None]
    correct = [row for row in scored if row["correct"]]
    incorrect = [row for row in scored if not row["correct"]]
    labels = [bool(row["correct"]) for row in scored]
    scores = [float(row["score"]) for row in scored]

    wrong_with_answer = [row for row in incorrect if row["answer_score_validation_only"] is not None]
    wrong_scores = [float(row["score"]) for row in wrong_with_answer]
    wrong_errors = [1.0 - float(row["answer_score_validation_only"]) for row in wrong_with_answer]
    wrong_quartiles = []
    ordered_wrong = sorted(wrong_with_answer, key=lambda row: float(row["score"]))
    for index in range(4):
        start = index * len(ordered_wrong) // 4
        stop = (index + 1) * len(ordered_wrong) // 4
        bucket = ordered_wrong[start:stop]
        wrong_quartiles.append({
            "quartile": index + 1,
            "count": len(bucket),
            "quality": _describe(float(row["score"]) for row in bucket),
            "answer_error": _describe(1.0 - float(row["answer_score_validation_only"]) for row in bucket),
        })

    tool_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        for tool in set(row["tools"]):
            tool_groups[tool].append(row)
    tool_presence = {}
    for tool, group in sorted(tool_groups.items()):
        tool_presence[tool] = {
            "count": len(group),
            "all": _describe(float(row["score"]) for row in group),
            "correct": _describe(float(row["score"]) for row in group if row["correct"]),
            "incorrect": _describe(float(row["score"]) for row in group if not row["correct"]),
        }

    reason_counts = Counter(
        reason
        for row in results if row["score"] is None
        for reason in (row.get("reasons") or ["unspecified"])
    )
    summary = {
        "schema_version": "terminal-state-obligation-quality-audit-v3",
        "score_definition": "C_terminal if terminal exists else 0.5*C_max",
        "score_inputs": ["gold_sql", "sqlite_schema", "relation_derivation_v1", "terminal_evidence"],
        "explicitly_excluded_from_score": [
            "correctness", "answer_rows", "answer_similarity", "model_reasoning", "errors", "recovery"
        ],
        "input_records": len(results) + len(exclusions),
        "normalized_records": len(results),
        "excluded_records": len(exclusions),
        "eligible_records": len(scored),
        "eligibility_rate": len(scored) / len(results) if results else None,
        "correct_eligible": len(correct),
        "incorrect_eligible": len(incorrect),
        "quality": {
            "all": _describe(scores),
            "correct": _describe(float(row["score"]) for row in correct),
            "incorrect": _describe(float(row["score"]) for row in incorrect),
            "mean_gap_correct_minus_incorrect": (
                statistics.fmean(float(row["score"]) for row in correct)
                - statistics.fmean(float(row["score"]) for row in incorrect)
                if correct and incorrect else None
            ),
            "roc_auc_correctness": _roc_auc(labels, scores),
        },
        "components": {
            name: {
                "correct": _describe(float(row[name]) for row in correct if row[name] is not None),
                "incorrect": _describe(float(row[name]) for row in incorrect if row[name] is not None),
            }
            for name in ("c_terminal", "c_max")
        },
        "correct_low_tail": {
            "below_0_50": sum(float(row["score"]) < 0.50 for row in correct),
            "below_0_75": sum(float(row["score"]) < 0.75 for row in correct),
            "below_0_90": sum(float(row["score"]) < 0.90 for row in correct),
        },
        "incorrect_answer_error_validation": {
            "count": len(wrong_with_answer),
            "spearman_quality_vs_answer_error": _spearman(wrong_scores, wrong_errors),
            "quartiles_low_to_high_quality": wrong_quartiles,
            "note": "answer_score is joined after Q_tool computation and is never a score input",
        },
        "tool_presence": tool_presence,
        "unscorable_reasons": dict(reason_counts.most_common()),
        "same_task_equivalence": {
            "empirical_pairs": 0,
            "reason": "the frozen pool contains exactly one trajectory per task",
            "deterministic_tests": "covered separately by test_tool_state_obligation_quality.py",
        },
        "model_calls": 0,
        "optimizer_updates": 0,
    }

    with (output_dir / "trajectory_scores.jsonl").open("w", encoding="utf-8") as target:
        for row in results:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "excluded.jsonl").open("w", encoding="utf-8") as target:
        for row in exclusions:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-audit", type=Path)
    args = parser.parse_args()
    summary = audit(args.input, args.output_dir, args.answer_audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
