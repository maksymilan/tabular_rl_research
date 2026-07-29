#!/usr/bin/env python3
"""Analyze repeated fixed-200 runs together with the final fixed-1000 corpus.

This script is intentionally artifact-only: it does not execute gold SQL or modify
trajectories.  It verifies denotation-metric alignment, applies the recorded
provider-retry overrides, and emits reproducible aggregate and per-task summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_METRIC = "bird-set"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def task_id(record: dict[str, Any]) -> str:
    for key in ("trajectory_id", "instance_id", "example_id"):
        value = record.get(key)
        if value:
            return str(value)
    index = record.get("example_index", record.get("index"))
    if index is None:
        raise ValueError(f"record has no task identifier: {sorted(record)}")
    return f"bird_train_{int(index):05d}"


def by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = task_id(record)
        if identifier in indexed:
            raise ValueError(f"duplicate task id: {identifier}")
        indexed[identifier] = record
    return indexed


def merge_with_overrides(
    primary: Iterable[dict[str, Any]],
    *overrides: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = by_id(primary)
    for override_records in overrides:
        for identifier, record in by_id(override_records).items():
            if identifier not in merged:
                raise ValueError(f"override task absent from primary set: {identifier}")
            merged[identifier] = record
    return merged


def require_metric(name: str, records: dict[str, dict[str, Any]]) -> None:
    metrics = {record.get("denotation_comparison") for record in records.values()}
    if metrics != {EXPECTED_METRIC}:
        raise ValueError(f"{name} metric mismatch: expected {EXPECTED_METRIC}, got {metrics}")


def summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record.get("correct")) for record in records.values())
    legal = sum(bool(record.get("legal")) for record in records.values())
    failures = Counter(
        str(record.get("failure_type") or "unspecified")
        for record in records.values()
        if not record.get("correct")
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "legal": legal,
        "legal_rate": legal / total if total else 0.0,
        "failure_types": dict(sorted(failures.items())),
    }


def error_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("error_events")
    return events if isinstance(events, list) else []


def step_count(record: dict[str, Any]) -> int:
    for key in ("atomic_actions", "steps", "submitted_calls", "model_turns"):
        value = record.get(key)
        if isinstance(value, int):
            return value
    turns = record.get("turns")
    return len(turns) if isinstance(turns, list) else 0


def tools_used(record: dict[str, Any]) -> set[str]:
    tools: set[str] = set()
    turns = record.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            parsed = turn.get("parsed") if isinstance(turn, dict) else None
            if isinstance(parsed, dict) and parsed.get("tool"):
                tools.add(str(parsed["tool"]))
    for field in ("atomic_events", "action_blocks"):
        events = record.get(field)
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            tool = event.get("tool") or event.get("tool_name")
            if tool:
                tools.add(str(tool))
    return tools


def sample_shape(sample: Any) -> tuple[int, int] | None:
    if not isinstance(sample, list):
        return None
    if not sample:
        return (0, 0)
    if not all(isinstance(row, list) for row in sample):
        return None
    widths = {len(row) for row in sample}
    return (len(sample), next(iter(widths))) if len(widths) == 1 else None


def scalar_family(value: Any) -> str:
    if isinstance(value, bool):
        return "mixed"
    if isinstance(value, (int, float)):
        return "numeric"
    if isinstance(value, str):
        try:
            float(value.replace(",", "").rstrip("%"))
            return "numeric"
        except ValueError:
            return "text"
    return "mixed"


def visible_mismatch(record: dict[str, Any]) -> str:
    pred = record.get("pred_sample")
    gold = record.get("gold_sample")
    pred_shape = sample_shape(pred)
    gold_shape = sample_shape(gold)
    if pred_shape is None or gold_shape is None:
        return "unclassifiable"
    if pred_shape == (0, 0):
        return "empty_prediction"
    if pred_shape == gold_shape and pred == gold:
        return "visible_sample_equal_denotation_differs"
    if pred_shape[1] > gold_shape[1]:
        return "extra_columns"
    if pred_shape[1] < gold_shape[1]:
        return "too_few_columns"
    if pred_shape[0] != gold_shape[0]:
        return "same_width_row_count_differs"
    pred_types = {scalar_family(value) for row in pred for value in row}
    gold_types = {scalar_family(value) for row in gold for value in row}
    families = pred_types | gold_types
    if families == {"numeric"}:
        return "same_shape_numeric_mismatch"
    if families == {"text"}:
        return "same_shape_text_mismatch"
    return "same_shape_mixed_mismatch"


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return float(ordered[index])


def describe_steps(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = [step_count(record) for record in records]
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p90_nearest_rank": percentile(values, 0.9),
    }


def load_taxonomy(path: Path) -> tuple[dict[str, str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for category, entry in data["wrong_answer_primary_hypotheses"].items():
        for suffix in entry["ids"]:
            labels[f"bird_train_{suffix}"] = category
    nonlegal = {
        f"bird_train_{suffix}"
        for entry in data["nonlegal_failures"].values()
        for suffix in entry["ids"]
    }
    return labels, nonlegal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the trajectory and input artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ARTIFACT_ROOT/data/results/failure_corpus_scale_20260729).",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "data/results/failure_corpus_scale_20260729"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    core = by_id(
        read_jsonl(
            root
            / "data/trajectories/tool_usability_20260724/"
            "version24_fixed200_first50_bird_set.all.jsonl"
        )
        + read_jsonl(
            root
            / "data/trajectories/tool_usability_20260724/"
            "version24_fixed200_remaining150_bird_set.all.jsonl"
        )
    )
    additional = merge_with_overrides(
        read_jsonl(
            root
            / "data/trajectories/external_teacher_fixed1000_20260724/"
            "additional800_bird_set.all.jsonl"
        ),
        read_jsonl(
            root
            / "data/trajectories/external_teacher_fixed1000_20260724/"
            "provider_retry_bird_set.all.jsonl"
        ),
        read_jsonl(
            root
            / "data/trajectories/external_teacher_fixed1000_20260724/"
            "provider_retry_second_bird_set.all.jsonl"
        ),
    )
    fixed_1000 = {**core, **additional}
    if len(core) != 200 or len(additional) != 800 or len(fixed_1000) != 1000:
        raise ValueError(
            f"unexpected fixed corpus sizes: core={len(core)}, "
            f"additional={len(additional)}, total={len(fixed_1000)}"
        )

    runs = {
        "atomic_version24": core,
        "action_block_v10": by_id(
            read_jsonl(
                root
                / "data/trajectories/batch_plan_20260726/"
                "action_block_v10_low_friction_fixed200_r1.all.jsonl"
            )
        ),
        "action_block_v21": by_id(
            read_jsonl(
                root
                / "data/trajectories/batch_plan_20260726/"
                "action_block_v21_fixed200_r1.all.jsonl"
            )
        ),
        "action_block_v32": by_id(
            read_jsonl(
                root
                / "data/trajectories/batch_plan_20260726/"
                "action_block_v32_exact_terminal_join_shape_fixed40_r1.all.jsonl"
            )
            + read_jsonl(
                root
                / "data/trajectories/batch_plan_20260726/"
                "action_block_v32_exact_terminal_join_shape_remaining160_r1.all.jsonl"
            )
        ),
        "atomic_version39_full_context": by_id(
            read_jsonl(
                root
                / "data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/"
                "verified_success.all.jsonl"
            )
        ),
    }
    core_ids = set(core)
    for name, records in runs.items():
        require_metric(name, records)
        if set(records) != core_ids:
            raise ValueError(
                f"{name} does not match frozen fixed-200 ids: "
                f"missing={len(core_ids - set(records))}, extra={len(set(records) - core_ids)}"
            )
    require_metric("fixed_1000", fixed_1000)

    inputs = by_id(
        read_jsonl(root / "data/eval_inputs/bird_train_external_teacher_fixed1000.jsonl")
    )
    if set(inputs) != set(fixed_1000):
        raise ValueError("fixed-1000 inputs do not match final trajectory ids")

    taxonomy, taxonomy_nonlegal = load_taxonomy(
        root
        / "data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/"
        "failure_taxonomy.json"
    )

    stability_records: list[dict[str, Any]] = []
    stability_hist = Counter()
    for identifier in sorted(core_ids):
        outcomes = {
            name: bool(records[identifier].get("correct"))
            for name, records in runs.items()
        }
        correct_runs = sum(outcomes.values())
        stability_hist[correct_runs] += 1
        full_context_record = runs["atomic_version39_full_context"][identifier]
        stability_records.append(
            {
                "task_id": identifier,
                "correct_runs": correct_runs,
                "run_outcomes": outcomes,
                "universal_failure": correct_runs == 0,
                "universal_success": correct_runs == len(runs),
                "full_context_failure_type": (
                    full_context_record.get("failure_type")
                    if not full_context_record.get("correct")
                    else None
                ),
                "manual_primary_hypothesis": taxonomy.get(identifier),
                "manual_nonlegal_in_full_context": identifier in taxonomy_nonlegal,
                "external_knowledge_present": bool(
                    str(inputs[identifier].get("external_knowledge") or "").strip()
                ),
                "difficulty": full_context_record.get("difficulty"),
                "db_id": full_context_record.get("db_id"),
            }
        )

    failures = [record for record in fixed_1000.values() if not record.get("correct")]
    wrong_answers = [
        record for record in failures if record.get("failure_type") == "wrong_answer"
    ]
    successes = [record for record in fixed_1000.values() if record.get("correct")]
    failure_zero_errors = sum(not error_events(record) for record in failures)
    wrong_zero_errors = sum(not error_events(record) for record in wrong_answers)
    process_error_tasks = {
        "success": sum(bool(error_events(record)) for record in successes),
        "failure": sum(bool(error_events(record)) for record in failures),
        "wrong_answer": sum(bool(error_events(record)) for record in wrong_answers),
    }

    difficulty = defaultdict(lambda: {"total": 0, "correct": 0, "legal": 0})
    knowledge = defaultdict(lambda: {"total": 0, "correct": 0})
    error_event_types = {"success": Counter(), "failure": Counter(), "wrong_answer": Counter()}
    tool_presence = {"success": Counter(), "failure": Counter()}
    for identifier, record in fixed_1000.items():
        bucket = difficulty[str(record.get("difficulty") or "unknown")]
        bucket["total"] += 1
        bucket["correct"] += int(bool(record.get("correct")))
        bucket["legal"] += int(bool(record.get("legal")))
        knowledge_key = (
            "present"
            if str(inputs[identifier].get("external_knowledge") or "").strip()
            else "absent"
        )
        knowledge[knowledge_key]["total"] += 1
        knowledge[knowledge_key]["correct"] += int(bool(record.get("correct")))
        group = "success" if record.get("correct") else "failure"
        for tool in tools_used(record):
            tool_presence[group][tool] += 1
        for event in error_events(record):
            event_type = str(
                event.get("error_type")
                or event.get("type")
                or event.get("code")
                or "unknown"
            )
            error_event_types[group][event_type] += 1
            if record.get("failure_type") == "wrong_answer":
                error_event_types["wrong_answer"][event_type] += 1

    mismatch = Counter(visible_mismatch(record) for record in wrong_answers)
    universal = [record for record in stability_records if record["universal_failure"]]
    universal_taxonomy = Counter(
        record["manual_primary_hypothesis"] or "full_context_nonlegal_presentation"
        for record in universal
    )

    fixed_failure_records = []
    for record in sorted(failures, key=task_id):
        identifier = task_id(record)
        fixed_failure_records.append(
            {
                "task_id": identifier,
                "db_id": record.get("db_id"),
                "difficulty": record.get("difficulty"),
                "failure_type": record.get("failure_type"),
                "legal": bool(record.get("legal")),
                "process_error_count": len(error_events(record)),
                "zero_process_errors": not error_events(record),
                "steps": step_count(record),
                "tools_used": sorted(tools_used(record)),
                "external_knowledge_present": bool(
                    str(inputs[identifier].get("external_knowledge") or "").strip()
                ),
                "visible_mismatch": (
                    visible_mismatch(record)
                    if record.get("failure_type") == "wrong_answer"
                    else None
                ),
                "in_frozen_200": identifier in core_ids,
                "correct_runs_in_modern_fixed200": (
                    next(
                        (
                            item["correct_runs"]
                            for item in stability_records
                            if item["task_id"] == identifier
                        ),
                        None,
                    )
                    if identifier in core_ids
                    else None
                ),
            }
        )

    result = {
        "schema_version": "large-failure-corpus-analysis-v1",
        "denotation_comparison": EXPECTED_METRIC,
        "artifact_root": str(root),
        "fixed_1000": {
            **summary(fixed_1000),
            "failure_total": len(failures),
            "failure_zero_process_errors": failure_zero_errors,
            "wrong_answer_total": len(wrong_answers),
            "wrong_answer_zero_process_errors": wrong_zero_errors,
            "tasks_with_process_errors": process_error_tasks,
            "difficulty": {
                key: {
                    **value,
                    "accuracy": value["correct"] / value["total"],
                    "legal_rate": value["legal"] / value["total"],
                }
                for key, value in sorted(difficulty.items())
            },
            "external_knowledge": {
                key: {
                    **value,
                    "accuracy": value["correct"] / value["total"],
                }
                for key, value in sorted(knowledge.items())
            },
            "steps": {
                "success": describe_steps(successes),
                "failure": describe_steps(failures),
            },
            "error_event_types": {
                key: dict(sorted(counter.items()))
                for key, counter in error_event_types.items()
            },
            "tool_presence": {
                key: dict(sorted(counter.items()))
                for key, counter in tool_presence.items()
            },
            "wrong_answer_visible_mismatch": dict(sorted(mismatch.items())),
        },
        "repeated_fixed_200": {
            "run_summaries": {
                name: summary(records) for name, records in runs.items()
            },
            "correct_run_histogram": {
                str(key): stability_hist[key] for key in range(len(runs) + 1)
            },
            "universal_failures": len(universal),
            "universal_successes": stability_hist[len(runs)],
            "variant_sensitive": 200 - len(universal) - stability_hist[len(runs)],
            "oracle_union_correct": 200 - len(universal),
            "universal_failure_manual_taxonomy": dict(sorted(universal_taxonomy.items())),
            "universal_failure_external_knowledge_present": sum(
                record["external_knowledge_present"] for record in universal
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "fixed1000_failures.jsonl", fixed_failure_records)
    write_jsonl(output_dir / "modern_fixed200_stability.jsonl", stability_records)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
