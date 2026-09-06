#!/usr/bin/env python3
"""CPU-only pass1-versus-pass2 training-cohort behavior diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl.diagnostics.metrics import exact_sign_p
from rl.diagnostics.records import group_by_example_index, load_jsonl
from rl.diagnostics.reporting import write_report


# Historical callers imported ``atomic_json`` from this scenario.  Keep the
# spelling as a compatibility alias while routing writes through the public API.
atomic_json = write_report


def summarize(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != 2880:
        raise ValueError(f"expected 2880 rollout rows, got {len(rows)}")
    by_pass = [
        group_by_example_index(block)
        for block in (rows[:1440], rows[1440:])
    ]
    if set(by_pass[0]) != set(by_pass[1]) or len(by_pass[0]) != 180:
        raise ValueError("two training passes do not contain the same 180 tasks")
    per_task = []
    for index in sorted(by_pass[0]):
        first = by_pass[0][index]
        second = by_pass[1][index]
        if len(first) != 8 or len(second) != 8:
            raise ValueError(f"task {index} is not K8 in both passes")
        first_correct = sum(bool(row.get("correct")) for row in first)
        second_correct = sum(bool(row.get("correct")) for row in second)
        first_legal = sum(bool(row.get("legal")) for row in first)
        second_legal = sum(bool(row.get("legal")) for row in second)
        per_task.append(
            {
                "example_index": index,
                "pass1_correct": first_correct,
                "pass2_correct": second_correct,
                "correct_delta": second_correct - first_correct,
                "pass1_legal": first_legal,
                "pass2_legal": second_legal,
                "legal_delta": second_legal - first_legal,
            }
        )

    def paired(field: str) -> dict[str, Any]:
        deltas = [int(row[field]) for row in per_task]
        improved = sum(value > 0 for value in deltas)
        regressed = sum(value < 0 for value in deltas)
        return {
            "improved_tasks": improved,
            "regressed_tasks": regressed,
            "tied_tasks": sum(value == 0 for value in deltas),
            "task_delta_sum": sum(deltas),
            "mean_task_delta": sum(deltas) / len(deltas),
            "exact_two_sided_sign_p": exact_sign_p(improved, regressed),
        }

    return {
        "schema_version": "earlystop-mixed180-pass-pairing-diagnostic-v1",
        "interpretation": (
            "training-cohort fitting signal only; selected tasks were used by both "
            "passes, so this is not a generalization or formal-evaluation gate"
        ),
        "tasks": 180,
        "samples_per_task_per_pass": 8,
        "correct": paired("correct_delta"),
        "legal": paired("legal_delta"),
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.rollouts)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("tasks", "correct", "legal")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
