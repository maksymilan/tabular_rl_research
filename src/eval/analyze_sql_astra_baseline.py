#!/usr/bin/env python3
"""Pair a SQL-ASTRA-style direct-SQL run with a canonical direct-SQL control."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record["instance_id"])
            if identifier in records:
                raise ValueError(f"duplicate instance_id in {path}: {identifier}")
            records[identifier] = record
    return records


def exact_two_sided_binomial_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, numerator / (2**discordant))


def sample_outcome(record: dict[str, Any]) -> str:
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        return str(record.get("failure_type") or "missing_sample")
    return str(samples[0].get("failure_type") or "correct")


def difficulty(record: dict[str, Any]) -> str:
    value = record.get("difficulty")
    if value:
        return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("difficulty"):
        return str(metadata["difficulty"])
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    astra = read_jsonl(args.astra)
    canonical = read_jsonl(args.canonical)
    tasks = read_jsonl(args.tasks)
    if set(astra) != set(canonical) or set(astra) != set(tasks):
        raise ValueError(
            "paired inputs have different ids: "
            f"astra={len(astra)}, canonical={len(canonical)}, tasks={len(tasks)}"
        )

    paired = {
        "both_correct": [],
        "astra_only": [],
        "canonical_only": [],
        "both_wrong": [],
    }
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for identifier in sorted(astra):
        astra_correct = bool(astra[identifier].get("correct"))
        canonical_correct = bool(canonical[identifier].get("correct"))
        if astra_correct and canonical_correct:
            bucket = "both_correct"
        elif astra_correct:
            bucket = "astra_only"
        elif canonical_correct:
            bucket = "canonical_only"
        else:
            bucket = "both_wrong"
        paired[bucket].append(identifier)
        diff = difficulty(tasks[identifier])
        difficulty_counts[diff]["total"] += 1
        difficulty_counts[diff]["astra_correct"] += int(astra_correct)
        difficulty_counts[diff]["canonical_correct"] += int(canonical_correct)

    astra_correct = sum(bool(record.get("correct")) for record in astra.values())
    canonical_correct = sum(
        bool(record.get("correct")) for record in canonical.values()
    )
    total = len(astra)
    astra_only = len(paired["astra_only"])
    canonical_only = len(paired["canonical_only"])
    result = {
        "schema_version": "sql-astra-direct-sql-paired-analysis-v1",
        "denotation_comparison": "bird-set",
        "total": total,
        "astra": {
            "correct": astra_correct,
            "accuracy": astra_correct / total,
            "sample_outcomes": dict(
                sorted(Counter(map(sample_outcome, astra.values())).items())
            ),
        },
        "canonical": {
            "correct": canonical_correct,
            "accuracy": canonical_correct / total,
            "sample_outcomes": dict(
                sorted(Counter(map(sample_outcome, canonical.values())).items())
            ),
        },
        "difference": {
            "correct": astra_correct - canonical_correct,
            "percentage_points": 100 * (astra_correct - canonical_correct) / total,
            "paper_sql_astra_qwen25_7b_accuracy": 0.475,
            "astra_to_paper_percentage_points": 100 * (astra_correct / total - 0.475),
        },
        "paired": {
            key: {"count": len(identifiers), "ids": identifiers}
            for key, identifiers in paired.items()
        },
        "exact_two_sided_mcnemar_p": exact_two_sided_binomial_p(
            astra_only, canonical_only
        ),
        "difficulty": {
            key: {
                **counts,
                "astra_accuracy": counts["astra_correct"] / counts["total"],
                "canonical_accuracy": counts["canonical_correct"] / counts["total"],
                "difference_percentage_points": 100
                * (counts["astra_correct"] - counts["canonical_correct"])
                / counts["total"],
            }
            for key, counts in sorted(difficulty_counts.items())
        },
        "completion_recoveries": [
            {
                "instance_id": identifier,
                "correct": bool(record.get("correct")),
                "failure_type": record.get("failure_type"),
                "completion_recovery": record["completion_recovery"],
            }
            for identifier, record in sorted(astra.items())
            if record.get("completion_recovery")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
