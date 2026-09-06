#!/usr/bin/env python3
"""Select a deterministic reserve window after an existing stratified cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rl.scenarios.fixed_pool.select_stratified_bird_train import level_of, load


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_reserve_window(
    rows: list[dict[str, Any]],
    *,
    selected_per_level: int,
    reserve_per_level: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("example_id") or row.get("instance_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        if not row.get("gold_sql") and not row.get("query"):
            continue
        if not Path(str(row.get("db_path") or "")).is_file():
            continue
        grouped[level_of(row)].append(row)

    rng = random.Random(seed)
    reserves: list[dict[str, Any]] = []
    stop = selected_per_level + reserve_per_level
    for level in ("simple", "moderate", "challenging"):
        candidates = sorted(
            grouped[level],
            key=lambda row: str(row.get("example_id") or row.get("instance_id")),
        )
        if len(candidates) < stop:
            raise ValueError(f"only {len(candidates)} usable {level} tasks; need {stop}")
        rng.shuffle(candidates)
        for source_rank, row in enumerate(
            candidates[selected_per_level:stop],
            start=selected_per_level,
        ):
            retained = json.loads(json.dumps(row))
            retained.setdefault("metadata", {})["fixed_pool_difficulty"] = level
            retained["metadata"]["fixed_pool_reserve_rank"] = source_rank
            reserves.append(retained)
    rng.shuffle(reserves)
    return reserves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected-per-level", type=int, default=20)
    parser.add_argument("--reserve-per-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    rows = load(args.input)
    selected = select_reserve_window(
        rows,
        selected_per_level=args.selected_per_level,
        reserve_per_level=args.reserve_per_level,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "fixed-pool-admission-reserve-selection-v1",
        "source": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.input
        ],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "selected_per_level": args.selected_per_level,
        "reserve_per_level": args.reserve_per_level,
        "tasks": len(selected),
        "difficulty_counts": dict(
            sorted(Counter(level_of(row) for row in selected).items())
        ),
        "task_ids": [
            str(row.get("example_id") or row.get("instance_id")) for row in selected
        ],
        "selection_rule": (
            "same seed-101 per-level shuffle as the original cohort; take the next "
            "reserve_per_level tasks after the original selected_per_level window"
        ),
        "admission_rule": (
            "every correct SFT2 rollout must pass the unchanged counterfactual-completeness "
            "gate; zero-correct tasks satisfy the universal gate but contribute no positive pair"
        ),
        "gold_visibility": "hidden harness metadata only; never rendered to the actor",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
