#!/usr/bin/env python3
"""Freeze a teacher-union cohort with exact category and difficulty marginals."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))

from src.rl.distillation.teacher_eligibility import load_jsonl, sha256_file


CATEGORIES = (
    "both_dense",
    "branch_only",
    "exp15_only_dense",
    "sft2_only_dense",
)
LEVELS = ("simple", "moderate", "challenging")


def allocate_cells(
    availability: Mapping[tuple[str, str], int],
    *,
    per_category: int,
    per_difficulty: int,
) -> dict[tuple[str, str], int]:
    """Find the most even feasible 4x3 integer table under exact marginals."""
    if len(CATEGORIES) * per_category != len(LEVELS) * per_difficulty:
        raise ValueError("category and difficulty totals do not match")
    ideal = per_category / len(LEVELS)
    options: dict[str, list[tuple[int, int, int]]] = {}
    for category in CATEGORIES:
        values = []
        for simple in range(per_category + 1):
            for moderate in range(per_category - simple + 1):
                challenging = per_category - simple - moderate
                triple = (simple, moderate, challenging)
                if all(
                    triple[index] <= int(availability.get((category, level), 0))
                    for index, level in enumerate(LEVELS)
                ):
                    values.append(triple)
        options[category] = values
    states: dict[tuple[int, int, int], tuple[float, tuple[tuple[int, int, int], ...]]] = {
        (0, 0, 0): (0.0, ())
    }
    for category in CATEGORIES:
        next_states = {}
        for totals, (cost, choices) in states.items():
            for triple in options[category]:
                updated = tuple(totals[index] + triple[index] for index in range(3))
                if any(value > per_difficulty for value in updated):
                    continue
                candidate = (
                    cost + sum((value - ideal) ** 2 for value in triple),
                    choices + (triple,),
                )
                previous = next_states.get(updated)
                if previous is None or candidate < previous:
                    next_states[updated] = candidate
        states = next_states
    target = (per_difficulty,) * len(LEVELS)
    if target not in states:
        raise ValueError(
            "teacher eligibility pool cannot satisfy strict category/difficulty marginals"
        )
    choices = states[target][1]
    return {
        (category, level): choices[category_index][level_index]
        for category_index, category in enumerate(CATEGORIES)
        for level_index, level in enumerate(LEVELS)
    }


def interleaved_cell_order(
    selected: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[int]:
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in selected:
        buckets[(str(row["category"]), str(row["difficulty"]))].append(
            int(row["example_index"])
        )
    queues = {}
    for cell in sorted(buckets):
        values = buckets[cell]
        digest = hashlib.sha256(f"{seed}\0{cell[0]}\0{cell[1]}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
        queues[cell] = deque(values)
    order = []
    while any(queues.values()):
        for cell in sorted(queues):
            if queues[cell]:
                order.append(queues[cell].popleft())
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligibility", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--per-difficulty", type=int, default=40)
    parser.add_argument("--seed", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite strict teacher-union selection")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen":
        raise ValueError("source teacher eligibility is not frozen")
    if source_manifest.get("eligibility_sha256") != sha256_file(args.eligibility):
        raise ValueError("source teacher eligibility hash mismatch")
    tasks = {int(row["example_index"]): row for row in load_jsonl(args.tasks)}
    rows = []
    availability: Counter[tuple[str, str]] = Counter()
    for row in load_jsonl(args.eligibility):
        category = str(row["category"])
        if category not in CATEGORIES:
            continue
        example_index = int(row["example_index"])
        task = tasks.get(example_index)
        if task is None:
            raise ValueError(f"missing difficulty metadata for example {example_index}")
        level = str((task.get("metadata") or {}).get("fixed_pool_difficulty"))
        if level not in LEVELS:
            raise ValueError(f"unsupported difficulty for example {example_index}: {level!r}")
        retained = dict(row)
        retained["difficulty"] = level
        rows.append(retained)
        availability[(category, level)] += 1
    allocation = allocate_cells(
        availability,
        per_category=args.per_category,
        per_difficulty=args.per_difficulty,
    )
    selected = []
    for category in CATEGORIES:
        for level in LEVELS:
            candidates = [
                row
                for row in rows
                if row["category"] == category and row["difficulty"] == level
            ]
            digest = hashlib.sha256(
                f"{args.seed}\0select\0{category}\0{level}".encode()
            ).digest()
            random.Random(int.from_bytes(digest[:8], "big")).shuffle(candidates)
            selected.extend(candidates[: allocation[(category, level)]])
    order = interleaved_cell_order(selected, seed=args.seed)
    rank = {example_index: index for index, example_index in enumerate(order)}
    for row in selected:
        row["training_order"] = rank[int(row["example_index"])]
    selected.sort(key=lambda row: int(row["example_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.next.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    manifest = {
        **source_manifest,
        "schema_version": "strict-teacher-eligibility-manifest-v1",
        "status": "frozen",
        "source_eligibility": str(args.eligibility.resolve()),
        "source_eligibility_sha256": sha256_file(args.eligibility),
        "eligibility": str(args.output.resolve()),
        "eligibility_sha256": sha256_file(args.output),
        "tasks": len(selected),
        "selection_seed": args.seed,
        "selection_rule": "exact-category-and-difficulty-marginals-minimum-cell-imbalance",
        "category_counts": dict(sorted(Counter(row["category"] for row in selected).items())),
        "difficulty_counts": dict(
            sorted(Counter(row["difficulty"] for row in selected).items())
        ),
        "category_difficulty_counts": {
            category: {
                level: allocation[(category, level)] for level in LEVELS
            }
            for category in CATEGORIES
        },
        "source_category_difficulty_availability": {
            category: {
                level: availability[(category, level)] for level in LEVELS
            }
            for category in CATEGORIES
        },
        "training_order": order,
        "training_order_sha256": hashlib.sha256(
            json.dumps(order, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
