#!/usr/bin/env python3
"""Analyze every incorrect trajectory in the frozen BIRD-train rollout pool.

The analysis is read-only and validation-only.  It combines deterministic trajectory events,
Gold-SQL task-shape flags, and the existing terminal-state audit.  It does not assign rewards or
use model-authored reasoning as evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TAIL_CATEGORIES = ("predicate", "grain", "value", "set", "rank", "output")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in read_jsonl(path)}


def load_manual(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row["task_id"]): row
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def _regex(sql: str, pattern: str) -> bool:
    return re.search(pattern, sql, flags=re.IGNORECASE) is not None


def gold_sql_shape(sql: str) -> dict[str, bool]:
    normalized = " ".join(str(sql or "").split())
    select_count = len(re.findall(r"\bselect\b", normalized, flags=re.IGNORECASE))
    has_limit = _regex(normalized, r"\blimit\b")
    has_order = _regex(normalized, r"\border\s+by\b")
    return {
        "join": _regex(normalized, r"\bjoin\b"),
        "aggregate": _regex(normalized, r"\b(count|sum|avg|min|max)\s*\("),
        "group_by": _regex(normalized, r"\bgroup\s+by\b"),
        "having": _regex(normalized, r"\bhaving\b"),
        "order_by": has_order,
        "limit": has_limit,
        "distinct": _regex(normalized, r"\bdistinct\b"),
        "subquery": select_count > 1,
        "set_operation": _regex(normalized, r"\b(union|intersect|except)\b"),
        "conditional": _regex(normalized, r"\bcase\b|\biif\s*\("),
        "limit_without_order": has_limit and not has_order,
    }


def question_shape(question: str) -> dict[str, bool]:
    text = str(question or "").casefold()
    return {
        "count": bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", text)),
        "percentage_ratio": bool(re.search(r"\bpercent(?:age)?\b|\bratio\b|\bproportion\b|\bshare\b", text)),
        "average": bool(re.search(r"\baverage\b|\bmean\b", text)),
        "rank_extreme": bool(re.search(r"\btop\b|\bmost\b|\bleast\b|\bhighest\b|\blowest\b|\bfirst\b|\blast\b", text)),
        "date_time": bool(re.search(r"\bdate\b|\byear\b|\bmonth\b|\bday\b|\bage\b|\bold\b|\btime\b", text)),
        "boundary": bool(re.search(r"\bat least\b|\bat most\b|\band above\b|\bmore than\b|\bless than\b|\bgreater than\b|\bno less than\b", text)),
        "arbitrary_choice": bool(re.search(r"\bany\b", text)),
    }


def _category_score(score: dict[str, Any], category: str) -> float:
    overlap = score.get("terminal_overlap") or {}
    per_category = overlap.get("per_category") or {}
    value = per_category.get(category)
    if value is None:
        return 1.0
    raw = value.get("score")
    return 0.0 if raw is None else float(raw)


def semantic_features(score: dict[str, Any] | None) -> dict[str, Any]:
    if score is None:
        return {
            "semantic_eligible": False,
            "scoring_mode": "missing_score_record",
            "semantic_score": None,
            "answer_score": None,
            "source_score": None,
            "join_score": None,
            "tail_mismatches": [],
            "tail_mismatch_count": None,
        }
    mode = str(score.get("scoring_mode") or "unknown")
    if mode == "unscorable":
        return {
            "semantic_eligible": False,
            "scoring_mode": mode,
            "semantic_score": None,
            "answer_score": score.get("answer_score_validation_only"),
            "source_score": None,
            "join_score": None,
            "tail_mismatches": [],
            "tail_mismatch_count": None,
        }
    source = _category_score(score, "source")
    join = _category_score(score, "join")
    mismatches = [
        category for category in TAIL_CATEGORIES
        if _category_score(score, category) < 1.0
    ]
    return {
        "semantic_eligible": bool(score.get("semantic_eligible")),
        "scoring_mode": mode,
        "semantic_score": score.get("score"),
        "answer_score": score.get("answer_score_validation_only"),
        "source_score": source,
        "join_score": join,
        "tail_mismatches": mismatches,
        "tail_mismatch_count": len(mismatches),
    }


def classify_incorrect(
    *,
    failure_type: str,
    legal: bool,
    semantic: dict[str, Any],
) -> str:
    if failure_type == "api_error" or semantic["scoring_mode"] == "missing_score_record":
        return "transport_or_missing_episode"
    if not legal or failure_type != "wrong_answer":
        return "no_legal_terminal"
    if semantic["scoring_mode"] == "unscorable":
        return "semantic_unscorable"
    if semantic["scoring_mode"] != "terminal":
        return "no_grounded_terminal"
    source = float(semantic["source_score"])
    join = float(semantic["join_score"])
    tail_count = int(semantic["tail_mismatch_count"])
    if source == 1.0 and join == 1.0:
        if tail_count == 0:
            return "semantic_exact_but_wrong"
        if tail_count == 1:
            return "one_tail_near_miss"
        if tail_count == 2:
            return "two_tail_near_miss"
        return "route_solved_multi_tail"
    if source == 1.0 and join >= 0.5:
        return "partial_join_route"
    return "wrong_source_or_join_route"


def _turns(audit: dict[str, Any]) -> list[dict[str, Any]]:
    value = audit.get("steps")
    if not isinstance(value, list):
        value = audit.get("turns")
    return value if isinstance(value, list) else []


def trajectory_features(audit: dict[str, Any]) -> dict[str, Any]:
    turns = _turns(audit)
    actions = []
    for turn in turns:
        parsed = turn.get("parsed")
        if isinstance(parsed, dict) and parsed.get("tool"):
            actions.append(parsed)
        elif turn.get("tool"):
            actions.append(turn)
    tools = [str(action["tool"]) for action in actions]
    events = audit.get("error_events")
    if not isinstance(events, list):
        events = []
    error_types = [str(event.get("error_type") or "unknown") for event in events]
    adjacent_repeats = 0
    prior_signature: str | None = None
    for action in actions:
        signature = json.dumps(
            {"tool": action.get("tool"), "arguments": action.get("arguments")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature == prior_signature:
            adjacent_repeats += 1
        prior_signature = signature
    return {
        "turn_count": len(turns),
        "tool_count": len(tools),
        "tools": tools,
        "tool_counts": dict(Counter(tools)),
        "error_event_count": len(events),
        "error_types": error_types,
        "adjacent_exact_action_repeats": adjacent_repeats,
        "has_answer_action": "answer_from_context" in tools,
        "final_tool": tools[-1] if tools else None,
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return statistics.fmean(retained) if retained else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cohort_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, group_iter in _grouped(rows, key=lambda row: row["analysis_cohort"]):
        group = list(group_iter)
        result[cohort] = {
            "count": len(group),
            "rate_of_all_incorrect": _rate(len(group), len(rows)),
            "mean_turns": _mean(row["turn_count"] for row in group),
            "error_episode_count": sum(row["error_event_count"] > 0 for row in group),
            "error_episode_rate": _rate(
                sum(row["error_event_count"] > 0 for row in group), len(group)
            ),
            "mean_semantic_score": _mean(row["semantic_score"] for row in group),
            "mean_answer_score_validation_only": _mean(row["answer_score"] for row in group),
        }
    return result


def _grouped(rows: list[dict[str, Any]], key: Any) -> Iterable[tuple[Any, Iterable[dict[str, Any]]]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return sorted(buckets.items(), key=lambda item: str(item[0]))


def analyze(
    pool_path: Path,
    score_path: Path,
    manual_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = load_scores(score_path)
    manual = load_manual(manual_path)
    incorrect_rows: list[dict[str, Any]] = []
    overall_shape: dict[str, Counter[str]] = defaultdict(Counter)
    overall_question: dict[str, Counter[str]] = defaultdict(Counter)
    overall_tool_presence: dict[str, Counter[str]] = defaultdict(Counter)
    population = Counter()

    for row in read_jsonl(pool_path):
        environment = row.get("environment") or {}
        sample = row.get("sample") or {}
        audit = sample.get("audit_record") or {}
        task_id = str(environment.get("task_id") or audit.get("trajectory_id") or "")
        correct = bool(sample.get("correct"))
        population["records"] += 1
        population["correct" if correct else "incorrect"] += 1
        sql_shape = gold_sql_shape(str(environment.get("gold_sql") or audit.get("gold_sql") or ""))
        q_shape = question_shape(str(environment.get("question") or audit.get("question") or ""))
        traj = trajectory_features(audit)
        outcome_key = "correct" if correct else "incorrect"
        for name, present in sql_shape.items():
            if present:
                overall_shape[name]["total"] += 1
                overall_shape[name][outcome_key] += 1
        for name, present in q_shape.items():
            if present:
                overall_question[name]["total"] += 1
                overall_question[name][outcome_key] += 1
        for tool in set(traj["tools"]):
            overall_tool_presence[tool]["total"] += 1
            overall_tool_presence[tool][outcome_key] += 1
        if correct:
            continue

        score = scores.get(task_id)
        semantic = semantic_features(score)
        failure_type = str(audit.get("failure_type") or (score or {}).get("failure_type") or "unknown")
        legal = bool(audit.get("legal", (score or {}).get("legal", False)))
        annotation = manual.get(task_id)
        compact = {
            "task_id": task_id,
            "example_index": environment.get("example_index"),
            "db_id": environment.get("db_id"),
            "failure_type": failure_type,
            "legal": legal,
            "analysis_cohort": classify_incorrect(
                failure_type=failure_type,
                legal=legal,
                semantic=semantic,
            ),
            **semantic,
            **traj,
            "gold_sql_shape": sql_shape,
            "question_shape": q_shape,
            "score_reasons": list((score or {}).get("reasons") or []),
            "manual_gold_issue": None if annotation is None else bool(annotation.get("gold_issue")),
            "manual_severity": None if annotation is None else int(annotation["severity"]),
            "manual_kind": None if annotation is None else annotation.get("kind"),
        }
        incorrect_rows.append(compact)

    if population["records"] != 2000:
        raise ValueError(f"expected frozen 2000 records, found {population['records']}")
    if population["incorrect"] != 564:
        raise ValueError(f"expected frozen 564 incorrect records, found {population['incorrect']}")

    near_rows = [
        row for row in incorrect_rows
        if row["analysis_cohort"] in {"one_tail_near_miss", "two_tail_near_miss"}
    ]
    mismatch_patterns = Counter("+".join(row["tail_mismatches"]) for row in near_rows)
    mismatch_categories = Counter(
        category for row in near_rows for category in row["tail_mismatches"]
    )
    error_types = Counter(row_type for row in incorrect_rows for row_type in row["error_types"])
    error_episode_count = sum(row["error_event_count"] > 0 for row in incorrect_rows)
    unscorable_reasons = Counter(
        reason
        for row in incorrect_rows
        if row["analysis_cohort"] == "semantic_unscorable"
        for reason in row["score_reasons"]
    )

    def outcome_table(source: dict[str, Counter[str]]) -> dict[str, Any]:
        result = {}
        for name, counts in sorted(source.items()):
            total = counts["total"]
            result[name] = {
                "tasks": total,
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "accuracy": _rate(counts["correct"], total),
                "share_of_all_incorrect": _rate(counts["incorrect"], population["incorrect"]),
            }
        return result

    manual_rows = [row for row in incorrect_rows if row["manual_severity"] is not None]
    manual_by_cohort = {}
    for cohort, group_iter in _grouped(manual_rows, key=lambda row: row["analysis_cohort"]):
        group = list(group_iter)
        manual_by_cohort[cohort] = {
            "count": len(group),
            "gold_issue": sum(bool(row["manual_gold_issue"]) for row in group),
            "gold_issue_rate": _rate(sum(bool(row["manual_gold_issue"]) for row in group), len(group)),
            "mean_severity": _mean(row["manual_severity"] for row in group),
            "severity_counts": dict(sorted(Counter(row["manual_severity"] for row in group).items())),
        }

    summary = {
        "schema_version": "all-incorrect-trajectory-pattern-analysis-v1",
        "dataset_split": "train",
        "population": dict(population),
        "incorrect_failure_types": dict(sorted(Counter(row["failure_type"] for row in incorrect_rows).items())),
        "incorrect_legal": {
            "legal": sum(row["legal"] for row in incorrect_rows),
            "not_legal": sum(not row["legal"] for row in incorrect_rows),
        },
        "cohorts": _cohort_stats(incorrect_rows),
        "near_miss": {
            "definition": "legal wrong answer; grounded terminal; exact source and join obligations; one or two mismatching tail semantic families",
            "count": len(near_rows),
            "rate_of_all_incorrect": _rate(len(near_rows), len(incorrect_rows)),
            "mismatch_patterns": dict(mismatch_patterns.most_common()),
            "mismatch_category_occurrences": dict(mismatch_categories.most_common()),
        },
        "process_errors": {
            "episodes_with_error": error_episode_count,
            "episode_rate": _rate(error_episode_count, len(incorrect_rows)),
            "events": sum(error_types.values()),
            "types": dict(error_types.most_common()),
            "adjacent_exact_action_repeats": sum(
                row["adjacent_exact_action_repeats"] for row in incorrect_rows
            ),
        },
        "gold_sql_shape_outcomes": outcome_table(overall_shape),
        "question_shape_outcomes": outcome_table(overall_question),
        "tool_presence_outcomes_observational": outcome_table(overall_tool_presence),
        "unscorable_reason_families": dict(unscorable_reasons.most_common()),
        "manual_new72_by_automatic_cohort": manual_by_cohort,
        "notes": [
            "Correctness, Gold SQL, answer similarity, and manual labels are validation-only.",
            "Tool-presence and SQL-shape accuracies are observational and are not causal tool rewards.",
            "Near-miss is a deterministic tool-state cohort, not a reward admission decision.",
            "Manual new72 is score-stratified among benchmark-wrong cases and cannot estimate population Gold-error prevalence.",
        ],
        "model_calls": 0,
        "optimizer_updates": 0,
        "evaluation_set_records_used": 0,
    }
    return summary, incorrect_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manual", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.pool, args.scores, args.manual)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "incorrect_rows.jsonl").open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
