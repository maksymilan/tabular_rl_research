#!/usr/bin/env python3
"""Fail-closed exact-example comparison of a SAAM candidate and SFT1."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_RECORDS = 1534
IDENTITY_FIELDS = (
    "dataset_split",
    "db_id",
    "question",
    "protocol_version",
    "protocol_hash",
    "assistant_carrier",
    "denotation_comparison",
)


def load(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("example_index")
            if not isinstance(index, int):
                raise ValueError(f"{path}:{line_number} has no integer example_index")
            if index in records:
                raise ValueError(f"{path} duplicates example_index={index}")
            records[index] = record
    return records


def exact_two_sided_binomial(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(gains, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sft1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_RECORDS)
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"refusing to overwrite comparison output: {args.output}")
    candidate = load(args.candidate)
    sft1 = load(args.sft1)
    expected_ids = set(range(args.expected_records))
    if set(candidate) != expected_ids:
        missing = sorted(expected_ids - set(candidate))[:10]
        extra = sorted(set(candidate) - expected_ids)[:10]
        raise ValueError(f"candidate identity set mismatch missing={missing} extra={extra}")
    if set(sft1) != expected_ids:
        missing = sorted(expected_ids - set(sft1))[:10]
        extra = sorted(set(sft1) - expected_ids)[:10]
        raise ValueError(f"SFT1 identity set mismatch missing={missing} extra={extra}")

    identity_mismatches = []
    for index in sorted(expected_ids):
        left, right = candidate[index], sft1[index]
        differences = [field for field in IDENTITY_FIELDS if left.get(field) != right.get(field)]
        if differences:
            identity_mismatches.append({"example_index": index, "fields": differences})
    if identity_mismatches:
        raise ValueError(f"candidate/SFT1 task identities differ: {identity_mismatches[:10]}")

    correct_gain: list[int] = []
    correct_regression: list[int] = []
    legal_gain: list[int] = []
    legal_regression: list[int] = []
    candidate_correct = sft1_correct = candidate_legal = sft1_legal = 0
    for index in sorted(expected_ids):
        cand = candidate[index]
        base = sft1[index]
        cand_correct = bool(cand.get("correct"))
        base_correct = bool(base.get("correct"))
        cand_legal = int(cand.get("sample_legal_count", 0)) > 0
        base_legal = int(base.get("sample_legal_count", 0)) > 0
        candidate_correct += int(cand_correct)
        sft1_correct += int(base_correct)
        candidate_legal += int(cand_legal)
        sft1_legal += int(base_legal)
        if cand_correct and not base_correct:
            correct_gain.append(index)
        elif base_correct and not cand_correct:
            correct_regression.append(index)
        if cand_legal and not base_legal:
            legal_gain.append(index)
        elif base_legal and not cand_legal:
            legal_regression.append(index)

    total = args.expected_records
    payload = {
        "schema_version": "qwen3-v26-saam-candidate-sft1-paired-comparison-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": str(args.candidate.resolve()),
        "sft1": str(args.sft1.resolve()),
        "records": total,
        "accuracy": {
            "candidate_correct": candidate_correct,
            "candidate_rate": candidate_correct / total,
            "sft1_correct": sft1_correct,
            "sft1_rate": sft1_correct / total,
            "gain_ids": correct_gain,
            "regression_ids": correct_regression,
            "gains": len(correct_gain),
            "regressions": len(correct_regression),
            "net_gain": len(correct_gain) - len(correct_regression),
            "percentage_point_change": 100.0 * (candidate_correct - sft1_correct) / total,
            "exact_two_sided_mcnemar_p": exact_two_sided_binomial(
                len(correct_gain), len(correct_regression)
            ),
        },
        "legal": {
            "candidate_legal": candidate_legal,
            "candidate_rate": candidate_legal / total,
            "sft1_legal": sft1_legal,
            "sft1_rate": sft1_legal / total,
            "gain_ids": legal_gain,
            "regression_ids": legal_regression,
            "gains": len(legal_gain),
            "regressions": len(legal_regression),
            "net_gain": len(legal_gain) - len(legal_regression),
            "percentage_point_change": 100.0 * (candidate_legal - sft1_legal) / total,
            "exact_two_sided_mcnemar_p": exact_two_sided_binomial(
                len(legal_gain), len(legal_regression)
            ),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
