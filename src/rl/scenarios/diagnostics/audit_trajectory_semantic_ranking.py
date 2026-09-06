#!/usr/bin/env python3
"""Audit Gold-state trajectory ranking on a frozen BIRD-train K-sample pool.

The command is read-only with respect to the policy.  It replays recorded legal actions on their
source SQLite database, computes complete predicted/gold denotations, compiles Gold SQL and the
terminal relation-derivation lineage into the shared semantic IR, and reports:

* correct-vs-incorrect raw-quality separation and ROC AUC;
* whether high-quality failures have smaller denotation errors;
* same-question correct trajectories with distinct tool paths but unexpectedly different raw
  quality;
* fail-closed semantic compilation coverage and reasons.

No optimizer or model server is used by this audit.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from rl.diagnostics.io import read_jsonl as _diagnostic_read_jsonl
from rl.diagnostics.io import sha256_file as _diagnostic_sha256_file

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from executor import Harness  # noqa: E402
from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402
from rollout import execute_tool, new_ctx, overview  # noqa: E402
from trajectory_semantic_ranker import (  # noqa: E402
    GoldSemanticCompiler,
    TrajectorySemanticCompiler,
    answer_similarity,
    load_sqlite_columns,
    semantic_overlap,
    trajectory_quality,
)


def remap_replay_handles(value: Any, handle_map: dict[str, str]) -> Any:
    """Translate recorded relation handles to handles allocated by the fresh replay."""
    if isinstance(value, list):
        return [remap_replay_handles(item, handle_map) for item in value]
    if isinstance(value, dict):
        return {key: remap_replay_handles(item, handle_map) for key, item in value.items()}
    if not isinstance(value, str) or not handle_map:
        return value
    if value in handle_map:
        return handle_map[value]
    rendered = value
    for recorded, replayed in sorted(handle_map.items(), key=lambda item: -len(item[0])):
        if rendered.startswith(f"{recorded}."):
            return f"{replayed}.{rendered[len(recorded) + 1:]}"
    return rendered


def sha256_file(path: Path) -> str:
    """Compatibility export backed by :mod:`rl.diagnostics.io`."""
    return _diagnostic_sha256_file(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _diagnostic_read_jsonl(path)]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, Any]:
    retained = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not retained:
        return {"count": 0}
    return {
        "count": len(retained),
        "mean": statistics.fmean(retained),
        "std": statistics.pstdev(retained) if len(retained) > 1 else 0.0,
        "min": min(retained),
        "p10": quantile(retained, 0.10),
        "p25": quantile(retained, 0.25),
        "median": quantile(retained, 0.50),
        "p75": quantile(retained, 0.75),
        "p90": quantile(retained, 0.90),
        "max": max(retained),
    }


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:stop])
        index = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        rank = ((index + 1) + stop) / 2.0
        for original, _ in ordered[index:stop]:
            result[original] = rank
        index = stop
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def canonical_path(steps: Iterable[dict[str, Any]]) -> str:
    actions = []
    handle_aliases: dict[str, str] = {}
    next_handle = 1

    def normalize(value: Any) -> Any:
        nonlocal next_handle
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in sorted(value.items()) if key != "reason"}
        if isinstance(value, str):
            if value in handle_aliases:
                return handle_aliases[value]
            rendered = value
            for recorded, canonical in sorted(handle_aliases.items(), key=lambda item: -len(item[0])):
                if rendered == recorded:
                    return canonical
                rendered = rendered.replace(f"{recorded}.", f"{canonical}.")
            return rendered
        return value

    for step in steps:
        call = step.get("tool_call") or {}
        tool = call.get("tool")
        if not isinstance(tool, str):
            continue
        actions.append({"tool": tool, "arguments": normalize(call.get("arguments") or {})})
        output = step.get("tool_output") or {}
        handle = output.get("table")
        if isinstance(handle, str) and handle not in handle_aliases:
            handle_aliases[handle] = f"derived_{next_handle}"
            next_handle += 1
    return hashlib.sha256(
        json.dumps(actions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    environment = dict(row.get("environment") or {})
    environment.setdefault("split", "train")
    environment.setdefault(
        "example_id",
        environment.get("task_id")
        or f"bird_train_{int(environment.get('example_index', -1)):05d}",
    )
    return environment


def _record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    sample = row.get("sample") or {}
    record = deepcopy(sample.get("audit_record") or {})
    record.setdefault("correct", bool(sample.get("correct")))
    return record


def replay_all_artifacts(
    trajectory: dict[str, Any],
    *,
    wanted_handles: set[str],
    timeout_seconds: float,
) -> tuple[list[tuple[Any, ...]], dict[str, list[tuple[Any, ...]]], dict[str, Any]]:
    source = trajectory["source"]
    harness = Harness(str(source["db_path"]))
    deadline = time.monotonic() + timeout_seconds
    harness.conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1_000)
    try:
        gold_cursor = harness.conn.execute(str(source["gold_sql"]))
        gold_rows_raw = gold_cursor.fetchmany(10_001)
        gold_complete = len(gold_rows_raw) <= 10_000
        gold_rows = [tuple(row) for row in gold_rows_raw[:10_000]]
        ctx = new_ctx(overview(harness))
        handle_map: dict[str, str] = {}
        terminal_recorded: str | None = None
        for step in trajectory.get("steps") or []:
            call = step.get("tool_call") or {}
            tool = call.get("tool")
            authored_arguments = call.get("arguments")
            if not isinstance(tool, str) or not isinstance(authored_arguments, dict):
                continue
            arguments = remap_replay_handles(deepcopy(authored_arguments), handle_map)
            if tool == "answer_from_context":
                evidence = authored_arguments.get("evidence")
                if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
                    terminal_recorded = evidence["table"]
                break
            _, replayed_handle = execute_tool(
                harness,
                tool,
                arguments,
                ctx,
                str(step.get("step_id") or ""),
                table_output_rows=0,
            )
            if replayed_handle:
                recorded = (step.get("tool_output") or {}).get("table")
                if isinstance(recorded, str):
                    handle_map[recorded] = replayed_handle
        artifact_rows: dict[str, list[tuple[Any, ...]]] = {}
        artifact_complete: dict[str, bool] = {}
        for recorded in wanted_handles:
            replayed = handle_map.get(recorded)
            if replayed is None:
                continue
            cursor = harness.conn.execute(f"SELECT * FROM {harness._src(replayed)} LIMIT 10001")
            raw = cursor.fetchmany(10_001)
            artifact_complete[recorded] = len(raw) <= 10_000
            artifact_rows[recorded] = [tuple(row) for row in raw[:10_000]]
        return gold_rows, artifact_rows, {
            "terminal_evidence": terminal_recorded,
            "handle_map": handle_map,
            "gold_rows_complete": gold_complete,
            "artifact_rows_complete": artifact_complete,
        }
    finally:
        harness.conn.set_progress_handler(None, 0)
        harness.conn.close()


def score_one(row: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    task = _task_from_row(row)
    if task.get("split") != "train":
        raise ValueError(f"non-train task entered audit: {task.get('split')!r}")
    record = _record_from_row(row)
    trajectory, exclusion = normalize_failure_record(record, task)
    sample_index = int((record.get("sample_index") if record.get("sample_index") is not None else (record.get("trajectory_id") or "0").rsplit("_", 1)[-1]))
    identity = {
        "task_id": task.get("task_id") or task.get("example_id"),
        "example_index": int(task["example_index"]),
        "sample_index": sample_index,
        "db_id": task.get("db_id"),
        "split": task.get("split"),
        "recorded_correct": bool(record.get("correct")),
        "recorded_legal": bool(record.get("legal")),
        "recorded_failure_type": record.get("failure_type"),
        "sft_admitted": bool((task.get("metadata") or {}).get("sft_admitted")),
    }
    if trajectory is None:
        return {**identity, "scored": False, "exclusion": exclusion}

    schema = load_sqlite_columns(task["db_path"])
    gold = GoldSemanticCompiler(schema).compile(str(task.get("gold_sql") or task.get("query")))
    candidate_compiler = TrajectorySemanticCompiler(schema)
    candidate, selected, selection = candidate_compiler.compile(trajectory["steps"])
    if selected in candidate_compiler.artifacts and selection == "terminal_evidence":
        candidate_handles: list[str | None] = [selected]
    else:
        # Without a legal terminal, do not scan every resident relation (some joins contain
        # millions of rows).  Keep at most the three relations whose semantic lineage best
        # overlaps Gold; answer similarity then breaks ties among this bounded candidate set.
        ranked_handles = []
        for handle, artifact in candidate_compiler.artifacts.items():
            if not artifact.input_refs:
                continue
            compiled = candidate_compiler.compiled_for(handle)
            overlap = semantic_overlap(gold, compiled)
            table_overlap = len(set(gold.tables) & set(compiled.tables)) / max(1, len(set(gold.tables) | set(compiled.tables)))
            column_overlap = len(set(gold.columns) & set(compiled.columns)) / max(1, len(set(gold.columns) | set(compiled.columns)))
            proxy = 0.45 * float(overlap.score or 0.0) + 0.04 * table_overlap + 0.06 * column_overlap
            ranked_handles.append((proxy, handle))
        ranked_handles.sort(reverse=True)
        candidate_handles = [handle for _, handle in ranked_handles[:3]] or [None]
        selection = "best_live_artifact_top3"
    try:
        gold_rows, artifact_rows, replay = replay_all_artifacts(
            trajectory,
            wanted_handles={handle for handle in candidate_handles if handle is not None},
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return {
            **identity,
            "scored": False,
            "exclusion": f"replay_error:{type(exc).__name__}:{exc}",
            "gold_semantic_eligible": gold.eligible,
            "gold_semantic_reasons": list(gold.reasons),
        }

    scored_candidates = []
    for handle in candidate_handles:
        compiled = candidate_compiler.compiled_for(handle)
        predicted = artifact_rows.get(handle, []) if handle is not None else []
        quality = trajectory_quality(
            correct=bool(record.get("correct")),
            gold=gold,
            candidate=compiled,
            predicted_rows=predicted,
            gold_rows=gold_rows,
            evidence_handle=handle,
            artifact_selection=selection,
        )
        scored_candidates.append((quality.raw_quality, handle, compiled, predicted, quality))
    _, selected, candidate, predicted_rows, quality = max(scored_candidates, key=lambda item: item[0])

    selected_complete = bool((replay.get("artifact_rows_complete") or {}).get(selected, True))
    exact_rows = (
        set(predicted_rows) == set(gold_rows)
        if replay.get("gold_rows_complete") and selected_complete
        else None
    )
    path_signature = canonical_path(trajectory["steps"])
    semantic_signature = hashlib.sha256(
        json.dumps(sorted(candidate.units.elements()), ensure_ascii=False).encode()
    ).hexdigest()[:16]
    return {
        **identity,
        "scored": True,
        "replay_correct": exact_rows,
        "correctness_agrees": (
            exact_rows == bool(record.get("correct")) if exact_rows is not None else None
        ),
        "path_signature": path_signature,
        "semantic_signature": semantic_signature,
        "tool_sequence": [
            (step.get("tool_call") or {}).get("tool")
            for step in trajectory["steps"]
            if (step.get("tool_call") or {}).get("tool")
        ],
        "quality": quality.to_dict(),
        "gold_units": sorted(gold.units.elements()),
        "candidate_units": sorted(candidate.units.elements()),
        "gold_tables": sorted(gold.tables),
        "candidate_tables": sorted(candidate.tables),
        "gold_columns": sorted(gold.columns),
        "candidate_columns": sorted(candidate.columns),
        "selected_artifact": selected,
        "candidate_artifacts_considered": len(candidate_handles),
        "row_support_complete": bool(replay.get("gold_rows_complete") and selected_complete),
    }


def _failure_quality_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = sorted(
        (record for record in records if not record["recorded_correct"]),
        key=lambda record: record["quality"]["raw_quality"],
    )
    if not failures:
        return []
    result = []
    for index in range(4):
        start = len(failures) * index // 4
        stop = len(failures) * (index + 1) // 4
        bucket = failures[start:stop]
        result.append({
            "quartile": index + 1,
            "count": len(bucket),
            "quality": describe(record["quality"]["raw_quality"] for record in bucket),
            "answer_score": describe(record["quality"]["answer_score"] for record in bucket),
            "answer_error": describe(1.0 - record["quality"]["answer_score"] for record in bucket),
            "legal_rate": statistics.fmean(float(record["recorded_legal"]) for record in bucket) if bucket else None,
        })
    return result


def summarize(records: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [record for record in records if record["recorded_correct"]]
    incorrect = [record for record in records if not record["recorded_correct"]]
    quality_scores = [record["quality"]["raw_quality"] for record in records]
    labels = [record["recorded_correct"] for record in records]
    correct_quality = [record["quality"]["raw_quality"] for record in correct]
    incorrect_quality = [record["quality"]["raw_quality"] for record in incorrect]
    admitted_correct = [record for record in correct if record.get("sft_admitted")]
    admitted_correct_quality = [record["quality"]["raw_quality"] for record in admitted_correct]
    failure_answer_scores = [record["quality"]["answer_score"] for record in incorrect]

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in correct:
        grouped[record["task_id"]].append(record)
    equivalent_groups = []
    for task_id, group in grouped.items():
        paths = {record["path_signature"] for record in group}
        if len(group) < 2 or len(paths) < 2:
            continue
        values = [record["quality"]["raw_quality"] for record in group]
        equivalent_groups.append({
            "task_id": task_id,
            "example_index": group[0]["example_index"],
            "correct_trajectories": len(group),
            "distinct_paths": len(paths),
            "distinct_semantics": len({record["semantic_signature"] for record in group}),
            "quality_min": min(values),
            "quality_max": max(values),
            "quality_range": max(values) - min(values),
            "records": [
                {
                    "sample_index": record["sample_index"],
                    "raw_quality": record["quality"]["raw_quality"],
                    "semantic_score": record["quality"]["semantic_score"],
                    "schema_score": record["quality"]["schema_score"],
                    "answer_score": record["quality"]["answer_score"],
                    "semantic_eligible": record["quality"]["semantic_eligible"],
                    "path_signature": record["path_signature"],
                    "tool_sequence": record["tool_sequence"],
                }
                for record in sorted(group, key=lambda item: item["sample_index"])
            ],
        })
    equivalent_groups.sort(key=lambda item: item["quality_range"], reverse=True)

    correct_p10 = quantile(correct_quality, 0.10)
    incorrect_p90 = quantile(incorrect_quality, 0.90)
    semantic_reasons = Counter(
        reason
        for record in records
        for key in ("gold_reasons", "candidate_reasons")
        for reason in record["quality"]["diagnostics"].get(key, [])
    )
    exclusions = Counter(record.get("exclusion") for record in excluded)
    return {
        "records": len(records),
        "excluded": len(excluded),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "correctness_disagreements": sum(record["correctness_agrees"] is False for record in records),
        "correctness_unchecked_due_row_cap": sum(record["correctness_agrees"] is None for record in records),
        "semantic_eligible": sum(record["quality"]["semantic_eligible"] for record in records),
        "semantic_eligibility_rate": (
            sum(record["quality"]["semantic_eligible"] for record in records) / len(records)
            if records else 0.0
        ),
        "separation": {
            "raw_quality_auc": roc_auc(labels, quality_scores),
            "correct_raw_quality": describe(correct_quality),
            "incorrect_raw_quality": describe(incorrect_quality),
            "mean_gap": (
                statistics.fmean(correct_quality) - statistics.fmean(incorrect_quality)
                if correct_quality and incorrect_quality else None
            ),
            "correct_p10_minus_incorrect_p90": (
                correct_p10 - incorrect_p90
                if correct_p10 is not None and incorrect_p90 is not None else None
            ),
            "correct_below_incorrect_p90": (
                sum(value < incorrect_p90 for value in correct_quality) / len(correct_quality)
                if correct_quality and incorrect_p90 is not None else None
            ),
            "sft_admitted_correct_vs_all_incorrect": {
                "admitted_correct": len(admitted_correct),
                "admitted_correct_raw_quality": describe(admitted_correct_quality),
                "incorrect": len(incorrect),
                "incorrect_raw_quality": describe(incorrect_quality),
                "raw_quality_auc": roc_auc(
                    [True] * len(admitted_correct) + [False] * len(incorrect),
                    admitted_correct_quality + incorrect_quality,
                ),
                "mean_gap": (
                    statistics.fmean(admitted_correct_quality)
                    - statistics.fmean(incorrect_quality)
                    if admitted_correct_quality and incorrect_quality else None
                ),
            },
        },
        "failure_severity": {
            "spearman_quality_vs_answer_score": spearman(incorrect_quality, failure_answer_scores),
            "spearman_quality_vs_answer_error": spearman(
                incorrect_quality,
                [1.0 - score for score in failure_answer_scores],
            ),
            "quality_quartiles": _failure_quality_bins(records),
            "highest_quality_failures": [
                {
                    "task_id": record["task_id"],
                    "example_index": record["example_index"],
                    "sample_index": record["sample_index"],
                    "raw_quality": record["quality"]["raw_quality"],
                    "semantic_score": record["quality"]["semantic_score"],
                    "answer_score": record["quality"]["answer_score"],
                    "recorded_legal": record["recorded_legal"],
                    "failure_type": record["recorded_failure_type"],
                    "tool_sequence": record["tool_sequence"],
                }
                for record in sorted(incorrect, key=lambda item: item["quality"]["raw_quality"], reverse=True)[:20]
            ],
        },
        "equivalent_correct_paths": {
            "groups_with_multiple_distinct_correct_paths": len(equivalent_groups),
            "groups_quality_range_over_0_10": sum(group["quality_range"] > 0.10 for group in equivalent_groups),
            "groups_quality_range_over_0_20": sum(group["quality_range"] > 0.20 for group in equivalent_groups),
            "worst_ranges": equivalent_groups[:30],
            "lowest_quality_correct": [
                {
                    "task_id": record["task_id"],
                    "example_index": record["example_index"],
                    "sample_index": record["sample_index"],
                    "raw_quality": record["quality"]["raw_quality"],
                    "semantic_score": record["quality"]["semantic_score"],
                    "schema_score": record["quality"]["schema_score"],
                    "answer_score": record["quality"]["answer_score"],
                    "semantic_eligible": record["quality"]["semantic_eligible"],
                    "tool_sequence": record["tool_sequence"],
                }
                for record in sorted(correct, key=lambda item: item["quality"]["raw_quality"])[:20]
            ],
            "optimization_reward_for_all_correct": 1.0,
            "pairwise_test_available": bool(equivalent_groups),
            "pairwise_test_limitation": (
                None if equivalent_groups else
                "No task has two recorded correct trajectories; false-low correct trajectories "
                "are still reported, but same-task path invariance cannot be estimated."
            ),
        },
        "semantic_reason_counts": dict(sorted(semantic_reasons.items())),
        "exclusion_counts": dict(sorted(exclusions.items(), key=lambda item: str(item[0]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episode-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.episode_timeout_seconds <= 0 or args.limit < 0:
        parser.error("workers and timeout must be positive; limit must be non-negative")
    pool_rows = rows(args.fixed_pool)
    if args.limit:
        pool_rows = pool_rows[: args.limit]
    if any((_task_from_row(row).get("split") != "train") for row in pool_rows):
        raise SystemExit("audit accepts BIRD-train fixed-pool rows only")

    scored: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(score_one, row, args.episode_timeout_seconds): index
            for index, row in enumerate(pool_rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            result["input_position"] = futures[future]
            (scored if result.get("scored") else excluded).append(result)
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures), "scored": len(scored), "excluded": len(excluded)}), flush=True)
    scored.sort(key=lambda item: item["input_position"])
    excluded.sort(key=lambda item: item["input_position"])
    summary = summarize(scored, excluded)
    summary["identity"] = {
        "fixed_pool": str(args.fixed_pool.resolve()),
        "fixed_pool_sha256": sha256_file(args.fixed_pool),
        "input_rows": len(pool_rows),
        "dataset_split": "train",
        "scorer": "gold-state-trajectory-ranker-v1",
        "semantic_overlap": "weighted-jaccard",
        "quality_weights": {"schema": 0.10, "semantic": 0.45, "answer": 0.45},
        "failure_reward": "-1 + 0.4 * raw_quality",
        "correct_reward": 1.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "trajectory_scores.jsonl"
    with records_path.open("w", encoding="utf-8") as target:
        for record in scored:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    excluded_path = args.output_dir / "excluded.jsonl"
    with excluded_path.open("w", encoding="utf-8") as target:
        for record in excluded:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["artifacts"] = {
        "trajectory_scores": str(records_path.resolve()),
        "trajectory_scores_sha256": sha256_file(records_path),
        "excluded": str(excluded_path.resolve()),
        "excluded_sha256": sha256_file(excluded_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
