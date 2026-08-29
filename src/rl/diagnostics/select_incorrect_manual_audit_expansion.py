#!/usr/bin/env python3
"""Select a reproducible score-stratified expansion of incorrect frozen trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


BINS = (
    ("wrong_00_0125", 0.0, 0.125),
    ("wrong_0125_025", 0.125, 0.25),
    ("wrong_025_040", 0.25, 0.40),
    ("wrong_040_060", 0.40, 0.60),
    ("wrong_060_075", 0.60, 0.75),
    ("wrong_075_100", 0.75, 1.0000000001),
)


def _task_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        task_id = value.get("task_id")
        if isinstance(task_id, str):
            yield task_id
        for child in value.values():
            yield from _task_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _task_ids(child)


def _stable_key(task_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select(
    score_rows: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    per_bin: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    availability: dict[str, int] = {}
    for label, low, high in BINS:
        candidates = [
            row
            for row in score_rows
            if not bool(row.get("correct"))
            and row.get("score") is not None
            and low <= float(row["score"]) < high
            and str(row.get("task_id")) not in excluded_ids
        ]
        availability[label] = len(candidates)
        candidates.sort(key=lambda row: _stable_key(str(row["task_id"]), seed))
        if len(candidates) < per_bin:
            raise ValueError(f"{label}: need {per_bin}, have {len(candidates)}")
        for row in candidates[:per_bin]:
            selected.append({
                "task_id": str(row["task_id"]),
                "db_id": row.get("db_id"),
                "score": float(row["score"]),
                "cell": label,
                "bin_low": low,
                "bin_high_exclusive": high,
                "selection_key": _stable_key(str(row["task_id"]), seed),
            })
    summary = {
        "schema_version": "incorrect-manual-audit-expansion-selection-v1",
        "seed": seed,
        "per_bin": per_bin,
        "selected": len(selected),
        "excluded_previously_audited": len(excluded_ids),
        "availability_after_exclusion": availability,
        "bins": [
            {"cell": label, "low": low, "high_exclusive": high}
            for label, low, high in BINS
        ],
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--per-bin", type=int, default=12)
    parser.add_argument("--seed", default="incorrect-manual-expansion-20260828")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.scores.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    excluded: set[str] = set()
    for path in args.exclude:
        excluded.update(_task_ids(json.loads(path.read_text(encoding="utf-8"))))
    selected, summary = select(
        rows,
        excluded_ids=excluded,
        per_bin=args.per_bin,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
