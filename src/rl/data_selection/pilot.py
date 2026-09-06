"""Reusable legality/correctness based pilot selection policies."""
from __future__ import annotations

from typing import Any

from .passk import sample_attempt_count, sample_correct_count


def legal_rate(record: dict[str, Any]) -> float:
    samples = record.get("samples") or []
    if not samples:
        return 1.0 if record.get("legal") else 0.0
    return sum(bool(sample.get("legal")) for sample in samples) / len(samples)


def pilot_bucket(record: dict[str, Any]) -> str:
    """Classify a rollout group for deterministic pilot selection."""
    attempts = sample_attempt_count(record)
    correct = sample_correct_count(record)
    rate = legal_rate(record)
    if rate < 0.8:
        return "format_or_execution_unstable"
    if correct == 0:
        return "legal_but_all_wrong"
    if correct < attempts:
        return "mixed_success"
    return "already_easy"


def select_pilot(records: list[dict[str, Any]], *, limit: int = 200,
                 include_hard: bool = False) -> list[dict[str, Any]]:
    order = {"mixed_success": 0, "legal_but_all_wrong": 1,
             "already_easy": 2, "format_or_execution_unstable": 3}
    allowed = {"mixed_success"}
    if include_hard:
        allowed.add("legal_but_all_wrong")
    enriched = [{**r, "rl_bucket": pilot_bucket(r), "rl_legal_rate": legal_rate(r)}
                for r in records]
    return [r for r in sorted(enriched, key=lambda x: (order[x["rl_bucket"]],
                                                        x.get("example_index", 10**9)))
            if r["rl_bucket"] in allowed][:limit]
