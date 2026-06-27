#!/usr/bin/env python3
"""Shared pass@k helpers for evaluation runners."""
from __future__ import annotations

import json

from artifacts import ArtifactWriter, utc_now


def parse_pass_k(value: str, *, n_samples: int) -> tuple[int, ...]:
    pass_k = tuple(int(item) for item in value.split(",") if item.strip())
    if not pass_k:
        raise ValueError("pass@k list cannot be empty")
    if any(k <= 0 for k in pass_k):
        raise ValueError("pass@k values must be positive")
    if max(pass_k) > n_samples:
        raise ValueError("pass@k contains a value larger than n_samples")
    return pass_k


def pass_at_from_flags(correct_flags: list[bool], pass_k: tuple[int, ...]) -> dict[str, bool]:
    return {str(k): any(correct_flags[:k]) for k in pass_k}


def attach_passk_fields(record: dict, samples: list[dict], pass_k: tuple[int, ...]) -> None:
    correct_flags = [bool(sample.get("correct")) for sample in samples]
    record["samples"] = samples
    record["sample_correct_count"] = sum(correct_flags)
    record["sample_legal_count"] = sum(bool(sample.get("legal")) for sample in samples)
    record["pass_at"] = pass_at_from_flags(correct_flags, pass_k)
    record["correct"] = bool(record["pass_at"].get(str(max(pass_k)), False))
    if not record["correct"]:
        record["failure_type"] = "all_samples_failed"


def write_passk_summary(
    writer: ArtifactWriter,
    pass_k: tuple[int, ...],
    *,
    include_legal: bool = False,
) -> dict:
    summary = writer.summarize()
    records = []
    if writer.all_path.exists():
        with writer.all_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    total = len(records)
    pass_counts = {
        str(k): sum(bool(record.get("pass_at", {}).get(str(k))) for record in records)
        for k in pass_k
    }
    summary["pass_at"] = {
        str(k): {
            "correct": pass_counts[str(k)],
            "total": total,
            "rate": pass_counts[str(k)] / total if total else 0.0,
        }
        for k in pass_k
    }
    summary["average_correct_samples"] = (
        sum(record.get("sample_correct_count", 0) for record in records) / total if total else 0.0
    )
    if include_legal:
        summary["average_legal_samples"] = (
            sum(record.get("sample_legal_count", 0) for record in records) / total if total else 0.0
        )
    summary["updated_at_utc"] = utc_now()
    writer.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
