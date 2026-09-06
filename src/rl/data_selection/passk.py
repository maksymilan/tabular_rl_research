"""Shared pass@k group predicates and selection policy for RL data."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

try:
    from rl.shared.io import read_jsonl
except ModuleNotFoundError:
    from shared.io import read_jsonl


INFRASTRUCTURE_FAILURE_TYPES = {
    "api_error", "context_overflow", "generation_oom", "incomplete_api_response",
    "provider_carrier_error", "task_timeout", "transport_error",
}


def sample_attempt_count(record: dict[str, Any]) -> int:
    samples = record.get("samples") or []
    if samples:
        return len(samples)
    return int(record.get("n_samples") or 1)


def sample_correct_count(record: dict[str, Any]) -> int:
    if record.get("sample_correct_count") is not None:
        return int(record["sample_correct_count"])
    samples = record.get("samples") or []
    return sum(bool(sample.get("correct") if sample.get("correct") is not None else sample.get("is_correct")) for sample in samples)


def sample_legal_count(record: dict[str, Any]) -> int:
    if record.get("sample_legal_count") is not None:
        return int(record["sample_legal_count"])
    return sum(bool(sample.get("legal")) for sample in record.get("samples") or [])


def has_infrastructure_failure(record: dict[str, Any]) -> bool:
    if record.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES:
        return True
    return any(
        sample.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES
        or any(bool(turn.get("api_error")) for turn in sample.get("turns") or [])
        for sample in record.get("samples") or []
    )


def has_mixed_attempt_outcomes(record: dict[str, Any]) -> bool:
    attempts = sample_attempt_count(record)
    correct = sample_correct_count(record)
    return attempts > 1 and 0 < correct < attempts and not has_infrastructure_failure(record)


def pass_keys(record: dict[str, Any]) -> list[int]:
    keys: list[int] = []
    for key in (record.get("pass_at") or {}):
        try:
            keys.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(keys)


def target_pass_k(records: list[dict[str, Any]], requested: int | None = None) -> int:
    if requested is not None:
        return requested
    keys = sorted({key for record in records for key in pass_keys(record)})
    if not keys:
        raise ValueError("pass@k artifact does not contain pass_at fields")
    return keys[-1]


def classify(record: dict[str, Any], *, target_k: int) -> str:
    pass_at = record.get("pass_at") or {}
    if bool(pass_at.get(str(target_k))) and not bool(pass_at.get("1")):
        return "pass1_fail_passk_success"
    if bool(pass_at.get("1")):
        return "pass1_success"
    return "all_attempted_failed"


def select_records(
    records: list[dict[str, Any]], *, mode: str, target_k: int | None = None,
    limit: int = 0, seed: int = 20260712, shuffle: bool = True,
) -> tuple[list[tuple[dict[str, Any], str]], int]:
    target = target_pass_k(records, target_k)
    buckets = [(record, classify(record, target_k=target)) for record in records]
    if mode == "all":
        selected = buckets
    elif mode == "mixed_attempt_outcomes":
        selected = [(record, mode) for record, _ in buckets if has_mixed_attempt_outcomes(record)]
    else:
        selected = [(record, bucket) for record, bucket in buckets if bucket == mode]
    if shuffle:
        random.Random(seed).shuffle(selected)
    return (selected[:limit] if limit else selected), target
