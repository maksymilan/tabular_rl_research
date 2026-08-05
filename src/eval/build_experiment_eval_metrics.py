#!/usr/bin/env python3
"""Build compact, comparable experiment metrics from a tool pass@k artifact."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    questions = len(records)
    samples = [
        (record, sample)
        for record in records
        for sample in record.get("samples") or []
    ]
    first_samples = [
        (record, record["samples"][0])
        for record in records
        if record.get("samples")
    ]
    pass_keys = sorted(
        {
            int(key)
            for record in records
            for key in (record.get("pass_at") or {})
        }
    )
    accuracy = {
        f"pass@{k}": {
            "correct": sum(
                bool((record.get("pass_at") or {}).get(str(k)))
                for record in records
            ),
            "total": questions,
        }
        for k in pass_keys
    }
    for value in accuracy.values():
        value["rate"] = value["correct"] / value["total"] if value["total"] else 0.0

    sample0_at_1 = {
        "correct": sum(bool(sample.get("correct")) for _, sample in first_samples),
        "total": len(first_samples),
    }
    sample0_at_1["rate"] = (
        sample0_at_1["correct"] / sample0_at_1["total"]
        if sample0_at_1["total"]
        else 0.0
    )
    is_greedy = bool(records) and all(
        int(record.get("n_samples") or len(record.get("samples") or [])) == 1
        and float(record.get("temperature") or 0.0) == 0.0
        and float(record.get("top_p") or 1.0) == 1.0
        for record in records
    )
    greedy_at_1 = (
        {**sample0_at_1, "available": True}
        if is_greedy
        else {
            "available": False,
            "correct": None,
            "total": questions,
            "rate": None,
            "reason": "requires n_samples=1, temperature=0, top_p=1 for every record",
        }
    )
    trajectory_accuracy = {
        "correct": sum(bool(sample.get("correct")) for _, sample in samples),
        "total": len(samples),
    }
    trajectory_accuracy["rate"] = (
        trajectory_accuracy["correct"] / trajectory_accuracy["total"]
        if trajectory_accuracy["total"]
        else 0.0
    )
    max_observed_samples = max(
        (len(record.get("samples") or []) for record in records),
        default=0,
    )
    correct_count_counter = Counter(
        sum(bool(sample.get("correct")) for sample in record.get("samples") or [])
        for record in records
    )
    correct_count_distribution = {
        str(count): {
            "questions": correct_count_counter.get(count, 0),
            "total": questions,
            "rate": correct_count_counter.get(count, 0) / questions if questions else 0.0,
        }
        for count in range(max_observed_samples + 1)
    }

    valid_rate = {
        "first_sample": {
            "valid": sum(bool(sample.get("legal")) for _, sample in first_samples),
            "total": len(first_samples),
        },
        "all_samples": {
            "valid": sum(bool(sample.get("legal")) for _, sample in samples),
            "total": len(samples),
        },
        "any_valid_per_question": {
            "valid": sum(
                any(bool(sample.get("legal")) for sample in record.get("samples") or [])
                for record in records
            ),
            "total": questions,
        },
    }
    for value in valid_rate.values():
        value["rate"] = value["valid"] / value["total"] if value["total"] else 0.0

    all_failure_types: Counter[str] = Counter()
    first_failure_types: Counter[str] = Counter()
    for _, sample in samples:
        label = "correct" if sample.get("correct") else str(
            sample.get("failure_type") or "unspecified_failure"
        )
        all_failure_types[label] += 1
    for _, sample in first_samples:
        label = "correct" if sample.get("correct") else str(
            sample.get("failure_type") or "unspecified_failure"
        )
        first_failure_types[label] += 1
    failure_types = {
        "all_samples": dict(sorted(all_failure_types.items())),
        "first_sample": dict(sorted(first_failure_types.items())),
    }

    avg_steps = {
        "first_sample": mean_or_none(
            float(sample.get("steps") or 0) for _, sample in first_samples
        ),
        "all_samples": mean_or_none(
            float(sample.get("steps") or 0) for _, sample in samples
        ),
        "correct_samples": mean_or_none(
            float(sample.get("steps") or 0)
            for _, sample in samples
            if sample.get("correct")
        ),
        "incorrect_samples": mean_or_none(
            float(sample.get("steps") or 0)
            for _, sample in samples
            if not sample.get("correct")
        ),
    }

    action_counts: Counter[str] = Counter()
    outcome_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    compact_trajectories = []
    logprob_rows = []
    for record, sample in samples:
        outcome = "correct" if sample.get("correct") else "incorrect"
        actions = []
        for turn in sample.get("turns") or []:
            parsed = turn.get("parsed") or {}
            tool = parsed.get("tool")
            if tool:
                action_counts[str(tool)] += 1
                outcome_action_counts[outcome][str(tool)] += 1
            actions.append(
                {
                    "turn_index": turn.get("turn_index"),
                    "tool": tool,
                    "arguments": parsed.get("arguments"),
                    "execution_error_type": turn.get("execution_error_type"),
                    "generation_stats": turn.get("generation_stats"),
                }
            )
        generation = sample.get("generation_stats")
        if generation:
            logprob_rows.append(
                {
                    "correct": bool(sample.get("correct")),
                    **generation,
                }
            )
        compact_trajectories.append(
            {
                "example_index": record.get("example_index"),
                "sample_index": sample.get("sample_index"),
                "correct": bool(sample.get("correct")),
                "legal": bool(sample.get("legal")),
                "steps": sample.get("steps"),
                "errors": sample.get("errors"),
                "failure_type": sample.get("failure_type"),
                "generation_stats": generation,
                "actions": actions,
            }
        )

    total_actions = sum(action_counts.values())
    action_distribution = {
        "total_actions": total_actions,
        "counts": dict(sorted(action_counts.items())),
        "rates": {
            tool: count / total_actions if total_actions else 0.0
            for tool, count in sorted(action_counts.items())
        },
        "by_trajectory_outcome": {
            outcome: {
                "total_actions": sum(counts.values()),
                "counts": dict(sorted(counts.items())),
            }
            for outcome, counts in sorted(outcome_action_counts.items())
        },
    }

    def logprob_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trajectories": len(rows),
            "mean_trajectory_action_logprob": mean_or_none(
                row["trajectory_action_logprob"] for row in rows
            ),
            "mean_token_surprisal": mean_or_none(
                row["mean_token_surprisal"] for row in rows
            ),
            "mean_token_entropy_lower_bound": mean_or_none(
                row["mean_token_entropy_lower_bound"] for row in rows
            ),
        }

    trajectory_entropy = {
        "metric_note": (
            "mean_token_entropy_lower_bound treats probability mass outside the "
            "requested top-logprobs as one bucket; mean_token_surprisal is exact "
            "for sampled tokens"
        ),
        "coverage": {
            "with_logprobs": len(logprob_rows),
            "total_trajectories": len(samples),
            "rate": len(logprob_rows) / len(samples) if samples else 0.0,
        },
        "all": logprob_summary(logprob_rows),
        "correct": logprob_summary(
            [row for row in logprob_rows if row["correct"]]
        ),
        "incorrect": logprob_summary(
            [row for row in logprob_rows if not row["correct"]]
        ),
    }

    per_question = [
        {
            "example_index": record.get("example_index"),
            "db_id": record.get("db_id"),
            "question": record.get("question"),
            "pass_at": record.get("pass_at"),
            "sample_correct_count": record.get("sample_correct_count"),
            "sample_legal_count": record.get("sample_legal_count"),
            "samples": [
                {
                    "sample_index": sample.get("sample_index"),
                    "correct": bool(sample.get("correct")),
                    "legal": bool(sample.get("legal")),
                    "steps": sample.get("steps"),
                    "errors": sample.get("errors"),
                    "failure_type": sample.get("failure_type"),
                    "generation_stats": sample.get("generation_stats"),
                }
                for sample in record.get("samples") or []
            ],
        }
        for record in records
    ]
    return {
        "evaluation_protocol": {
            "mode": "greedy" if is_greedy else "stochastic",
            "questions": questions,
            "trajectories": len(samples),
            "n_samples": sorted(
                {
                    int(record.get("n_samples") or len(record.get("samples") or []))
                    for record in records
                }
            ),
            "temperature": sorted(
                {float(record.get("temperature") or 0.0) for record in records}
            ),
            "top_p": sorted(
                {float(record.get("top_p") or 1.0) for record in records}
            ),
            "max_steps": sorted(
                {int(record.get("max_steps") or 0) for record in records}
            ),
            "protocol_version": sorted(
                {str(record.get("protocol_version")) for record in records}
            ),
            "denotation_comparison": sorted(
                {str(record.get("denotation_comparison")) for record in records}
            ),
        },
        "accuracy": accuracy,
        "sample0_at_1": sample0_at_1,
        "greedy_at_1": greedy_at_1,
        "trajectory_accuracy": trajectory_accuracy,
        "correct_count_distribution": correct_count_distribution,
        "valid_rate": valid_rate,
        "failure_types": failure_types,
        "avg_steps": avg_steps,
        "action_distribution": action_distribution,
        "trajectory_entropy": trajectory_entropy,
        "per_question_result": per_question,
        "trajectory": compact_trajectories,
    }


def write_metrics(result_dir: Path, metrics: dict[str, Any]) -> None:
    for name, payload in metrics.items():
        (result_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    all_path = args.result_dir / "all.jsonl"
    if not all_path.is_file():
        raise SystemExit(f"missing evaluation artifact: {all_path}")
    records = load_records(all_path)
    metrics = build_metrics(records)
    write_metrics(args.result_dir, metrics)
    print(
        json.dumps(
            {
                "result_dir": str(args.result_dir),
                "questions": len(records),
                "trajectories": len(metrics["trajectory"]),
                "accuracy": metrics["accuracy"],
                "valid_rate": metrics["valid_rate"],
                "avg_steps": metrics["avg_steps"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
