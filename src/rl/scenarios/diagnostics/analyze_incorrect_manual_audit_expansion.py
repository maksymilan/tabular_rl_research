#!/usr/bin/env python3
"""Aggregate the individually reviewed incorrect-trajectory expansion."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rl.scenarios.diagnostics.select_incorrect_manual_audit_expansion import BINS


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


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left, right = _ranks(left), _ranks(right)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _bin(score: float) -> str:
    for label, low, high in BINS:
        if low <= score < high:
            return label
    raise ValueError(f"score outside bins: {score}")


def _group_stats(rows: list[dict[str, Any]], *, include_gold: bool) -> dict[str, Any]:
    result = {}
    for label, _, _ in BINS:
        group = [row for row in rows if row["score_bin"] == label]
        if not group:
            continue
        item = {
            "count": len(group),
            "mean_score": statistics.fmean(row["current_score"] for row in group),
            "mean_severity": statistics.fmean(row["severity"] for row in group),
            "severity_counts": dict(sorted(Counter(row["severity"] for row in group).items())),
            "severe_2_or_3": sum(row["severity"] >= 2 for row in group),
        }
        if include_gold:
            item["gold_issue"] = sum(bool(row["gold_issue"]) for row in group)
        result[label] = item
    return result


def _correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality = [1.0 - row["severity"] / 3 for row in rows]
    return {
        "current_score_vs_human_quality_spearman": _spearman(
            [row["current_score"] for row in rows], quality
        ),
        "rank_cap_score_vs_human_quality_spearman": _spearman(
            [row["rank_cap_score"] for row in rows], quality
        ),
    }


def _pairwise(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    concordant = discordant = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            score_delta = float(left[key]) - float(right[key])
            quality_delta = int(right["severity"]) - int(left["severity"])
            if score_delta == 0 or quality_delta == 0:
                continue
            if score_delta * quality_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return {
        "comparable_pairs": total,
        "concordant": concordant,
        "discordant": discordant,
        "accuracy": concordant / total if total else None,
    }


def analyze(
    selection: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    prior_selection: list[dict[str, Any]],
    prior_annotations: list[dict[str, Any]],
    guarded_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotation_by_id = {str(row["task_id"]): row for row in annotations}
    if {str(row["task_id"]) for row in selection} != set(annotation_by_id):
        raise ValueError("new selection and annotations must contain the same task ids")
    guard_by_id = {str(row["task_id"]): row for row in guarded_rows}

    rows = []
    for selected in selection:
        task_id = str(selected["task_id"])
        annotation = annotation_by_id[task_id]
        guard = guard_by_id[task_id]
        rows.append({
            "task_id": task_id,
            "cohort": "new72",
            "score_bin": selected["cell"],
            "current_score": float(selected["score"]),
            "rank_cap_score": float(guard["rank_cap_score"]),
            "rank_scope_violation": guard.get("rank_expansion") is not None,
            "severity": int(annotation["severity"]),
            "kind": annotation["kind"],
            "gold_issue": bool(annotation["gold_issue"]),
            "score_judgment": annotation["score_judgment"],
            "note": annotation["note"],
        })

    prior_annotation_by_id = {str(row["task_id"]): row for row in prior_annotations}
    prior_rows = []
    for selected in prior_selection:
        if bool(selected.get("correct")):
            continue
        task_id = str(selected["task_id"])
        annotation = prior_annotation_by_id[task_id]
        guard = guard_by_id[task_id]
        current = float(selected["score"])
        prior_rows.append({
            "task_id": task_id,
            "cohort": "prior24",
            "score_bin": _bin(current),
            "current_score": current,
            "rank_cap_score": float(guard["rank_cap_score"]),
            "rank_scope_violation": guard.get("rank_expansion") is not None,
            "severity": int(annotation["severity"]),
            "kind": annotation["kind"],
            "gold_issue": None,
            "score_judgment": annotation["judgment"],
            "note": annotation["note"],
        })

    combined = prior_rows + rows
    non_gold = [row for row in rows if not row["gold_issue"]]
    summary = {
        "schema_version": "incorrect-manual-audit-expansion-analysis-v1",
        "new_cases": len(rows),
        "prior_compatible_incorrect_cases": len(prior_rows),
        "combined_incorrect_cases": len(combined),
        "new_severity_counts": dict(sorted(Counter(row["severity"] for row in rows).items())),
        "new_gold_issue": {
            "count": sum(row["gold_issue"] for row in rows),
            "rate": statistics.fmean(float(row["gold_issue"]) for row in rows),
        },
        "new_score_judgment": dict(Counter(row["score_judgment"] for row in rows).most_common()),
        "new_by_score_bin": _group_stats(rows, include_gold=True),
        "combined_by_score_bin": _group_stats(combined, include_gold=False),
        "correlations": {
            "new72_all": _correlations(rows),
            "new72_without_gold_issue": _correlations(non_gold),
            "combined96": _correlations(combined),
        },
        "pairwise_ordering": {
            "new72_current": _pairwise(rows, "current_score"),
            "new72_rank_cap": _pairwise(rows, "rank_cap_score"),
            "new72_without_gold_issue_current": _pairwise(non_gold, "current_score"),
            "combined96_current": _pairwise(combined, "current_score"),
            "combined96_rank_cap": _pairwise(combined, "rank_cap_score"),
        },
        "rank_scope_activation": {
            "new72": sum(row["rank_scope_violation"] for row in rows),
            "combined96": sum(row["rank_scope_violation"] for row in combined),
        },
        "notes": [
            "Every new case was individually reviewed from question, Gold SQL, answer samples, and tool trajectory.",
            "Severity: 0=no substantive trajectory error or Gold/question conflict; 1=minor; 2=moderate; 3=severe.",
            "The stratified sample supports failure discovery and ordering checks, not population prevalence.",
            "Correctness and human labels are validation-only and are never candidate reward inputs.",
        ],
        "model_calls": 0,
        "optimizer_updates": 0,
        "evaluation_set_records_used": 0,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--prior-annotations", type=Path, required=True)
    parser.add_argument("--guarded-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    prior_selection = json.loads(args.prior_selection.read_text(encoding="utf-8"))
    prior_annotations = json.loads(args.prior_annotations.read_text(encoding="utf-8"))
    with args.guarded_scores.open(encoding="utf-8") as source:
        guarded = [json.loads(line) for line in source if line.strip()]
    summary, rows = analyze(
        selection, annotations, prior_selection, prior_annotations, guarded
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "incorrect_expansion72_manual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "incorrect_expansion72_manual_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
