#!/usr/bin/env python3
"""Select a deterministic 20/20/20 BIRD-train cohort without exposing gold to actors."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVEL_MAP = {"easy": "simple", "medium": "moderate", "hard": "challenging"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def level_of(row: dict[str, Any]) -> str:
    raw = str((row.get("metadata") or {}).get("difficulty_proxy") or "").lower()
    try:
        return LEVEL_MAP[raw]
    except KeyError as exc:
        raise ValueError(f"missing/unsupported difficulty proxy for {row.get('example_id')}: {raw}") from exc


def select(rows: list[dict[str, Any]], per_level: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("example_id") or row.get("instance_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        if not row.get("gold_sql") and not row.get("query"):
            continue
        db_path = Path(str(row.get("db_path") or ""))
        if not db_path.is_file():
            continue
        grouped[level_of(row)].append(row)
    rng = random.Random(seed)
    selected = []
    for level in ("simple", "moderate", "challenging"):
        candidates = sorted(
            grouped[level],
            key=lambda row: str(row.get("example_id") or row.get("instance_id")),
        )
        if len(candidates) < per_level:
            raise ValueError(f"only {len(candidates)} usable {level} tasks; need {per_level}")
        rng.shuffle(candidates)
        for row in candidates[:per_level]:
            retained = dict(row)
            retained.setdefault("metadata", {})["fixed_pool_difficulty"] = level
            selected.append(retained)
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    rows = load(args.input)
    selected = select(rows, args.per_level, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "fixed-pool-task-selection-v1",
        "source": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.input
        ],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "per_level": args.per_level,
        "tasks": len(selected),
        "difficulty_counts": dict(
            sorted(Counter(level_of(row) for row in selected).items())
        ),
        "task_ids": [
            str(row.get("example_id") or row.get("instance_id")) for row in selected
        ],
        "gold_visibility": "hidden harness metadata only; never rendered to the actor",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
