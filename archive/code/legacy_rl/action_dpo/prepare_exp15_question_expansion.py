#!/usr/bin/env python3
"""Freeze clean source questions not covered by existing Exp15 pair datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clean_correct_source_rows(pool: Path) -> dict[int, dict[str, Any]]:
    """Match generate_exact_prefix_branches.source_rows eligibility exactly."""
    chosen: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(pool):
        environment = row.get("environment") or {}
        sample = row.get("sample") or {}
        audit = sample.get("audit_record") or {}
        if str(environment.get("dataset_split") or "train") != "train":
            continue
        if not sample.get("correct") or not audit.get("correct") or not audit.get("legal"):
            continue
        if int(audit.get("errors") or 0) != 0:
            continue
        if audit.get("protocol_version") != "version26":
            continue
        turns = audit.get("turns") or []
        if not turns or any(not isinstance(turn.get("model_output"), str) for turn in turns):
            continue
        example_index = int(environment["example_index"])
        sample_index = int(audit.get("sample_index") or 0)
        previous = chosen.get(example_index)
        if previous is None or sample_index < int(
            previous["sample"]["audit_record"].get("sample_index") or 0
        ):
            chosen[example_index] = row
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument(
        "--existing-dataset",
        required=True,
        action="append",
        type=Path,
        help="repeat for every already selected Exp15 pair dataset",
    )
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--expected-existing-questions", type=int)
    parser.add_argument("--expected-eligible-questions", type=int)
    parser.add_argument("--expected-new-questions", type=int)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite frozen question-expansion artifacts")

    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"frozen_passed", "frozen_rank_ready"}:
        raise SystemExit("source pool is not frozen and verifier-passed/rank-ready")
    if manifest.get("protocol_version") != "version26":
        raise SystemExit("source pool is not version26")
    expected_pool_sha = (
        (manifest.get("files") or {}).get("validated_trajectories") or {}
    ).get("sha256")
    actual_pool_sha = sha256_file(args.source_pool)
    if expected_pool_sha != actual_pool_sha:
        raise SystemExit("source pool hash differs from its frozen manifest")

    existing_questions: set[int] = set()
    input_summaries = []
    for path in args.existing_dataset:
        pairs = 0
        questions: set[int] = set()
        for row in load_jsonl(path):
            if "example_index" not in row or not row.get("pair_sha256"):
                raise SystemExit(f"existing dataset is not an Exp15 pair dataset: {path}")
            example_index = int(row["example_index"])
            questions.add(example_index)
            existing_questions.add(example_index)
            pairs += 1
        input_summaries.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "pairs": pairs,
                "questions": len(questions),
            }
        )

    chosen = clean_correct_source_rows(args.source_pool)
    new_indices = sorted(set(chosen) - existing_questions)
    overlap = sorted(set(chosen) & existing_questions)
    rows = []
    for example_index in new_indices:
        source = chosen[example_index]
        environment = source["environment"]
        source_audit = source["sample"]["audit_record"]
        rows.append(
            {
                "schema_version": "exp15-question-expansion-filter-v1",
                "question_id": str(example_index),
                "example_index": example_index,
                "db_id": environment["db_id"],
                "question": environment["question"],
                "source_sample_index": int(source_audit.get("sample_index") or 0),
                "source_turns": len(source_audit["turns"]),
            }
        )

    checks = (
        ("existing questions", len(existing_questions), args.expected_existing_questions),
        ("eligible questions", len(chosen), args.expected_eligible_questions),
        ("new questions", len(rows), args.expected_new_questions),
    )
    for label, actual, expected in checks:
        if expected is not None and actual != expected:
            raise SystemExit(f"expected {expected} {label}, found {actual}")

    write_jsonl_atomic(args.output_dataset, rows)
    audit = {
        "schema_version": "exp15-question-expansion-selection-audit-v1",
        "status": "passed",
        "selection": "clean_correct_version26_zero_error_first_sample_not_in_existing_exp15_questions",
        "source_pool": str(args.source_pool.resolve()),
        "source_pool_sha256": actual_pool_sha,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "eligible_clean_questions": len(chosen),
        "existing_questions": len(existing_questions),
        "eligible_existing_overlap_questions": len(overlap),
        "new_questions": len(rows),
        "new_question_ids": new_indices,
        "existing_inputs": input_summaries,
        "output": str(args.output_dataset.resolve()),
        "output_sha256": sha256_file(args.output_dataset),
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
