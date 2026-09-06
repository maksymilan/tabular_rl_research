"""Freeze training-only SFT2/Exp15 teacher eligibility from exact K rollouts."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


FORBIDDEN_OUTPUT_KEYS = {"gold_sql", "gold_query", "reference_sql"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _flatten_samples(row: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    environment = row.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("teacher rollout row lacks environment metadata")
    if isinstance(row.get("sample"), dict):
        yield environment, dict(row["sample"])
        return
    samples = row.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("teacher rollout samples must be objects")
            yield environment, dict(sample)
        return
    raise ValueError("teacher rollout row has neither sample nor samples")


def _identity(environment: Mapping[str, Any]) -> tuple[int, str, str]:
    split = environment.get("dataset_split")
    if split != "train":
        raise ValueError(f"teacher eligibility is train-only, observed split={split!r}")
    return (
        int(environment["example_index"]),
        str(environment["db_id"]),
        str(environment["question"]),
    )


def _sample_index(sample: Mapping[str, Any]) -> int:
    audit = sample.get("audit_record") or {}
    if "sample_index" not in audit:
        raise ValueError("teacher rollout sample lacks audit_record.sample_index")
    return int(audit["sample_index"])


def _is_correct(sample: Mapping[str, Any]) -> bool:
    audit = sample.get("audit_record") or {}
    return bool(sample.get("correct") and audit.get("correct") and audit.get("legal"))


def _is_legal(sample: Mapping[str, Any]) -> bool:
    audit = sample.get("audit_record") or {}
    return bool(audit.get("legal"))


@dataclass(frozen=True)
class TeacherCounts:
    trials: int
    correct: int
    legal: int

    def as_dict(self, *, minimum_dense_correct: int) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "correct": self.correct,
            "legal": self.legal,
            "dense_eligible": self.correct >= minimum_dense_correct,
            "branch_eligible": self.correct >= 1,
        }


def collect_teacher_rollouts(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_k: int,
    expected_protocol_version: str = "version26",
) -> dict[tuple[int, str, str], TeacherCounts]:
    if expected_k < 1:
        raise ValueError("expected_k must be positive")
    grouped: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        for environment, sample in _flatten_samples(row):
            identity = _identity(environment)
            audit = sample.get("audit_record") or {}
            if audit.get("protocol_version") != expected_protocol_version:
                raise ValueError(
                    f"example {identity[0]} protocol_version={audit.get('protocol_version')!r} "
                    f"!= {expected_protocol_version!r}"
                )
            index = _sample_index(sample)
            if index in grouped[identity]:
                raise ValueError(f"duplicate sample_index={index} for example {identity[0]}")
            grouped[identity][index] = sample
    result = {}
    expected = set(range(expected_k))
    for identity, samples in grouped.items():
        if set(samples) != expected:
            raise ValueError(
                f"example {identity[0]} sample indices {sorted(samples)} != {sorted(expected)}"
            )
        ordered = [samples[index] for index in range(expected_k)]
        result[identity] = TeacherCounts(
            trials=expected_k,
            correct=sum(_is_correct(sample) for sample in ordered),
            legal=sum(_is_legal(sample) for sample in ordered),
        )
    return result


def build_eligibility_rows(
    sft2_rows: Iterable[Mapping[str, Any]],
    exp15_rows: Iterable[Mapping[str, Any]],
    *,
    expected_k: int = 4,
    minimum_dense_correct: int = 2,
    expected_protocol_version: str = "version26",
) -> list[dict[str, Any]]:
    if minimum_dense_correct < 2 or minimum_dense_correct > expected_k:
        raise ValueError("minimum_dense_correct must be in [2, expected_k]")
    sft2 = collect_teacher_rollouts(
        sft2_rows,
        expected_k=expected_k,
        expected_protocol_version=expected_protocol_version,
    )
    exp15 = collect_teacher_rollouts(
        exp15_rows,
        expected_k=expected_k,
        expected_protocol_version=expected_protocol_version,
    )
    if set(sft2) != set(exp15):
        missing_sft2 = sorted(identity[0] for identity in set(exp15) - set(sft2))
        missing_exp15 = sorted(identity[0] for identity in set(sft2) - set(exp15))
        raise ValueError(
            f"teacher task mismatch: missing_sft2={missing_sft2[:8]}, "
            f"missing_exp15={missing_exp15[:8]}"
        )

    output = []
    for identity in sorted(sft2):
        example_index, db_id, question = identity
        a = sft2[identity]
        b = exp15[identity]
        a_dense = a.correct >= minimum_dense_correct
        b_dense = b.correct >= minimum_dense_correct
        if a_dense and b_dense:
            category = "both_dense"
        elif a_dense:
            category = "sft2_only_dense"
        elif b_dense:
            category = "exp15_only_dense"
        elif a.correct or b.correct:
            category = "branch_only"
        else:
            category = "both_wrong"
        row = {
            "schema_version": "teacher-eligibility-v1",
            "dataset_split": "train",
            "example_index": example_index,
            "db_id": db_id,
            "question": question,
            "category": category,
            "teachers": {
                "sft2": a.as_dict(minimum_dense_correct=minimum_dense_correct),
                "exp15": b.as_dict(minimum_dense_correct=minimum_dense_correct),
            },
        }
        if FORBIDDEN_OUTPUT_KEYS & row.keys():
            raise AssertionError("eligibility row leaked verifier-only SQL")
        output.append(row)
    return output


def interleaved_training_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    include_branch_only: bool = True,
) -> list[int]:
    """Pre-randomize within categories, then round-robin across categories."""
    allowed = {"sft2_only_dense", "exp15_only_dense", "both_dense"}
    if include_branch_only:
        allowed.add("branch_only")
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        category = str(row["category"])
        if category in allowed:
            buckets[category].append(int(row["example_index"]))
    rng = random.Random(seed)
    queues = {}
    for category in sorted(buckets):
        values = list(buckets[category])
        rng.shuffle(values)
        queues[category] = deque(values)
    output: list[int] = []
    categories = sorted(queues)
    while any(queues[category] for category in categories):
        for category in categories:
            if queues[category]:
                output.append(queues[category].popleft())
    return output
