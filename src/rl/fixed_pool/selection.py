"""Deterministic, eligibility checked fixed-pool task selection."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LEVEL_MAP = {"easy": "simple", "medium": "moderate", "hard": "challenging"}
LEVELS = ("simple", "moderate", "challenging")


def level_of(row: dict[str, Any]) -> str:
    raw = str((row.get("metadata") or {}).get("difficulty_proxy") or "").lower()
    try:
        return LEVEL_MAP[raw]
    except KeyError as exc:
        raise ValueError(f"missing/unsupported difficulty proxy for {row.get('example_id')}: {raw}") from exc


def usable_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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
    return grouped


def select(rows: list[dict[str, Any]], per_level: int, seed: int) -> list[dict[str, Any]]:
    grouped = usable_rows(rows)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for level in LEVELS:
        candidates = sorted(grouped[level], key=lambda r: str(r.get("example_id") or r.get("instance_id")))
        if len(candidates) < per_level:
            raise ValueError(f"only {len(candidates)} usable {level} tasks; need {per_level}")
        rng.shuffle(candidates)
        for row in candidates[:per_level]:
            retained = dict(row)
            retained["metadata"] = dict(retained.get("metadata") or {})
            retained["metadata"]["fixed_pool_difficulty"] = level
            selected.append(retained)
    rng.shuffle(selected)
    return selected
