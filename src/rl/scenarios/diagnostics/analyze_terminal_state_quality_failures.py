#!/usr/bin/env python3
"""Large-sample failure analysis for terminal-state semantic quality scores.

This diagnostic never changes the reward and never reads model-authored reasoning.  It uses the
frozen score JSONL to quantify cross-tail failures, output-position artifacts, per-category false
negatives, and tool-presence bias.  It also selects a deterministic, database-diverse case cohort
for a separate human audit against the original trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = ("source", "join", "predicate", "grain", "value", "set", "rank", "output")
RELATION_CATEGORIES = tuple(category for category in CATEGORIES if category != "output")
PRODUCER_TOOLS = (
    "condition_filter", "project", "join_tables", "group_aggregate",
    "scalar_compute", "extreme_value_select", "set_op",
)
PRIOR_CASE_IDS = {
    "bird_train_03566", "bird_train_03186", "bird_train_04911", "bird_train_01214",
    "bird_train_02317", "bird_train_03073", "bird_train_05388", "bird_train_05745",
    "bird_train_03061", "bird_train_02007", "bird_train_03371", "bird_train_02902",
}


def _mean(values: Iterable[float]) -> float | None:
    retained = list(values)
    return statistics.fmean(retained) if retained else None


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


def _category_details(row: dict[str, Any], category: str) -> dict[str, Any] | None:
    details = ((row.get("terminal_overlap") or {}).get("per_category") or {}).get(category)
    return details if isinstance(details, dict) else None


def _category_score(row: dict[str, Any], category: str) -> float | None:
    details = _category_details(row, category)
    return float(details["score"]) if details and details.get("score") is not None else None


def _relation_score(row: dict[str, Any]) -> float:
    scores = [
        score for category in RELATION_CATEGORIES
        if (score := _category_score(row, category)) is not None
    ]
    return statistics.fmean(scores) if scores else 1.0


def _strip_output_position(unit: str) -> str:
    payload = json.loads(unit)
    if (
        isinstance(payload, list) and len(payload) == 3 and payload[0] == "position"
        and isinstance(payload[1], int)
    ):
        return json.dumps(payload[2], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def output_content_overlap(row: dict[str, Any]) -> float | None:
    """Recover order-insensitive output overlap from stored matched/missing/extra counts.

    Already matched positional units are content matches.  Removing the position wrapper can add
    matches between the stored missing and extra units, e.g. after a leading extra output column.
    """
    details = _category_details(row, "output")
    if details is None:
        return None
    matched = int(details.get("matched") or 0)
    gold_count = int(details.get("gold") or 0)
    candidate_count = int(details.get("candidate") or 0)
    missing = Counter(_strip_output_position(unit) for unit in details.get("missing") or [])
    extra = Counter(_strip_output_position(unit) for unit in details.get("extra") or [])
    added = sum((missing & extra).values())
    content_matched = matched + added
    union = gold_count + candidate_count - content_matched
    return content_matched / union if union else 1.0


def _combined_diagnostic_score(row: dict[str, Any]) -> float:
    """A diagnostic-only 75/25 split; it is not an admitted reward."""
    output = output_content_overlap(row)
    return 0.75 * _relation_score(row) + 0.25 * (1.0 if output is None else output)


def _value_guarded_score(row: dict[str, Any]) -> float:
    """Conservatively downweight a structurally wrong required value on incorrect trajectories.

    The diagnostic is computed for both labels here, but an eventual reward would use it only in
    the already-incorrect branch; correct trajectories retain their fixed terminal reward.
    """
    value = _category_score(row, "value")
    factor = 1.0 if value is None else 0.5 + 0.5 * value
    return float(row["score"]) * factor


def _score_bucket(correct: bool, score: float) -> str:
    if score >= 0.80:
        band = "high"
    elif score >= 0.40:
        band = "mid"
    else:
        band = "low"
    return f"{'correct' if correct else 'incorrect'}_{band}"


def _stable_key(row: dict[str, Any]) -> str:
    payload = f"terminal-state-quality-expanded-v1:{row.get('task_id')}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select across score quantiles while preferring unseen databases and tool signatures."""
    if count <= 0:
        return []
    if len(rows) <= count:
        return sorted(rows, key=lambda row: (float(row["score"]), _stable_key(row)))
    ordered = sorted(rows, key=lambda row: (float(row["score"]), _stable_key(row)))
    targets = (
        [len(ordered) // 2]
        if count == 1
        else [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    )
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_dbs: set[str] = set()
    used_signatures: set[tuple[str, ...]] = set()
    for target in targets:
        candidates = sorted(
            enumerate(ordered),
            key=lambda item: (
                str(item[1].get("db_id")) in used_dbs,
                tuple(item[1].get("tools") or ()) in used_signatures,
                abs(item[0] - target),
                _stable_key(item[1]),
            ),
        )
        for _, row in candidates:
            task_id = str(row["task_id"])
            if task_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(task_id)
            used_dbs.add(str(row.get("db_id")))
            used_signatures.add(tuple(row.get("tools") or ()))
            break
    return selected


def _output_profile(row: dict[str, Any]) -> dict[str, Any] | None:
    details = _category_details(row, "output")
    if details is None:
        return None
    positional = float(details["score"])
    content = output_content_overlap(row)
    return {
        "positional_score": positional,
        "content_score": content,
        "gold": int(details.get("gold") or 0),
        "candidate": int(details.get("candidate") or 0),
        "matched": int(details.get("matched") or 0),
        "position_artifact": content is not None and content > positional + 1e-12,
        "content_exact_with_extra": (
            content is not None and content < 1.0 and int(details.get("candidate") or 0)
            > int(details.get("gold") or 0)
            and int(details.get("matched") or 0) + sum((
                Counter(_strip_output_position(unit) for unit in details.get("missing") or [])
                & Counter(_strip_output_position(unit) for unit in details.get("extra") or [])
            ).values()) == int(details.get("gold") or 0)
        ),
    }


def analyze(rows: list[dict[str, Any]], *, per_cell: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = [row for row in rows if row.get("score") is not None]
    correct = [row for row in scored if row.get("correct")]
    incorrect = [row for row in scored if not row.get("correct")]

    boundaries = [0.0, 0.125, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0000001]
    score_bins = []
    for low, high in zip(boundaries, boundaries[1:]):
        group = [row for row in scored if low <= float(row["score"]) < high]
        score_bins.append({
            "low": low,
            "high": min(high, 1.0),
            "count": len(group),
            "correct": sum(bool(row.get("correct")) for row in group),
            "correct_rate": (
                sum(bool(row.get("correct")) for row in group) / len(group) if group else None
            ),
        })

    category_stats: dict[str, Any] = {}
    for category in CATEGORIES:
        category_stats[category] = {}
        for name, group in (("correct", correct), ("incorrect", incorrect)):
            values = [
                score for row in group
                if (score := _category_score(row, category)) is not None
            ]
            category_stats[category][name] = {
                "present": len(values),
                "mean": _mean(values),
                "zero": sum(math.isclose(value, 0.0) for value in values),
                "one": sum(math.isclose(value, 1.0) for value in values),
            }

    output_rows = [row for row in scored if _output_profile(row) is not None]
    position_artifacts = [row for row in output_rows if _output_profile(row)["position_artifact"]]
    output_summary = {
        "present": len(output_rows),
        "position_artifact_count": len(position_artifacts),
        "position_artifact_correct": sum(bool(row.get("correct")) for row in position_artifacts),
        "position_artifact_incorrect": sum(not bool(row.get("correct")) for row in position_artifacts),
        "current_mean": {
            "correct": _mean(_category_score(row, "output") for row in correct
                             if _category_score(row, "output") is not None),
            "incorrect": _mean(_category_score(row, "output") for row in incorrect
                               if _category_score(row, "output") is not None),
        },
        "content_mean": {
            "correct": _mean(output_content_overlap(row) for row in correct
                             if output_content_overlap(row) is not None),
            "incorrect": _mean(output_content_overlap(row) for row in incorrect
                               if output_content_overlap(row) is not None),
        },
    }

    formulations = {
        "current": lambda row: float(row["score"]),
        "relation_only": _relation_score,
        "diagnostic_relation75_output_content25": _combined_diagnostic_score,
        "diagnostic_value_guarded": _value_guarded_score,
    }
    formulation_stats = {}
    for name, function in formulations.items():
        positive = [function(row) for row in correct]
        negative = [function(row) for row in incorrect]
        formulation_stats[name] = {
            "correct_mean": _mean(positive),
            "incorrect_mean": _mean(negative),
            "gap": (_mean(positive) or 0.0) - (_mean(negative) or 0.0),
            "auc": _auc(positive, negative),
        }

    correct_low = [row for row in correct if float(row["score"]) < 0.40]
    incorrect_high = [row for row in incorrect if float(row["score"]) >= 0.75]

    def zero_counts(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            category: sum(
                (score := _category_score(row, category)) is not None
                and math.isclose(score, 0.0)
                for row in group
            )
            for category in CATEGORIES
        }

    tool_stats = {}
    for tool in PRODUCER_TOOLS:
        ok = [row for row in correct if tool in (row.get("tools") or [])]
        bad = [row for row in incorrect if tool in (row.get("tools") or [])]
        tool_stats[tool] = {
            "correct_count": len(ok),
            "correct_current_mean": _mean(float(row["score"]) for row in ok),
            "correct_relation_mean": _mean(_relation_score(row) for row in ok),
            "incorrect_count": len(bad),
            "incorrect_current_mean": _mean(float(row["score"]) for row in bad),
            "incorrect_relation_mean": _mean(_relation_score(row) for row in bad),
        }

    cells: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        cells.setdefault(_score_bucket(bool(row.get("correct")), float(row["score"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for cell in (
        "correct_high", "correct_mid", "correct_low",
        "incorrect_high", "incorrect_mid", "incorrect_low",
    ):
        eligible = [row for row in cells.get(cell, []) if str(row.get("task_id")) not in PRIOR_CASE_IDS]
        for row in _select_diverse(eligible, per_cell):
            selected.append({
                "cell": cell,
                "task_id": row.get("task_id"),
                "db_id": row.get("db_id"),
                "position": row.get("position"),
                "correct": bool(row.get("correct")),
                "score": float(row["score"]),
                "c_terminal": row.get("c_terminal"),
                "c_max": row.get("c_max"),
                "relation_score": _relation_score(row),
                "output_profile": _output_profile(row),
                "tools": row.get("tools") or [],
                "zero_categories": [
                    category for category in CATEGORIES
                    if (value := _category_score(row, category)) is not None
                    and math.isclose(value, 0.0)
                ],
            })

    summary = {
        "schema_version": "terminal-state-quality-expanded-failure-analysis-v1",
        "records": {"scored": len(scored), "correct": len(correct), "incorrect": len(incorrect)},
        "score_bins": score_bins,
        "cross_tail": {
            "correct_below_0_40": len(correct_low),
            "correct_below_0_40_zero_categories": zero_counts(correct_low),
            "incorrect_at_or_above_0_75": len(incorrect_high),
            "incorrect_at_or_above_0_75_zero_categories": zero_counts(incorrect_high),
        },
        "category_stats": category_stats,
        "output_position_analysis": output_summary,
        "diagnostic_formulations": formulation_stats,
        "tool_presence": tool_stats,
        "selected_case_count": len(selected),
        "selected_per_cell": dict(Counter(item["cell"] for item in selected)),
        "notes": [
            "correctness is validation-only and is not an input to any candidate score",
            "the 75/25 formulation is diagnostic-only and is not an admitted RL reward",
            "tool-presence means are observational and confounded by task semantics",
            "the frozen pool still has one trajectory per task, so within-task GRPO ranking remains untested",
        ],
    }
    return summary, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=8)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    summary, selected = analyze(rows, per_cell=args.per_cell)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expanded_failure_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "expanded_case_selection.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
