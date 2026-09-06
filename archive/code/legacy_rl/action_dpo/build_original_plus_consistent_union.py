#!/usr/bin/env python3
"""Freeze the exact union of original Exp15 and verified non-overlap branch pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_expanded_exact_prefix_union import (
    sha256_file,
    validate_rows,
    write_json_atomic,
    write_jsonl_atomic,
)
from train_fixed_prefix_action_dpo import load_jsonl


def require_audited_dataset(
    dataset: Path,
    audit_path: Path,
    *,
    expected_pairs: int,
    label: str,
) -> tuple[list[dict], dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise SystemExit(f"{label} audit is not passed")
    if audit.get("output_sha256") != sha256_file(dataset):
        raise SystemExit(f"{label} dataset hash differs from its audit")
    rows = load_jsonl(dataset)
    validate_rows(rows, label=label)
    if len(rows) != expected_pairs:
        raise SystemExit(
            f"{label} expected {expected_pairs} pairs, found {len(rows)}"
        )
    if int(audit.get("verified_pairs") or 0) != len(rows):
        raise SystemExit(f"{label} audit pair count mismatch")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset", required=True, type=Path)
    parser.add_argument("--original-audit", required=True, type=Path)
    parser.add_argument("--consistent-dataset", required=True, type=Path)
    parser.add_argument("--consistent-audit", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--expected-original-pairs", type=int, default=36)
    parser.add_argument("--expected-consistent-pairs", type=int, default=28)
    parser.add_argument("--expected-union-pairs", type=int, default=64)
    parser.add_argument("--expected-union-questions", type=int, default=43)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite frozen union64 artifacts")

    original_rows, original_audit = require_audited_dataset(
        args.original_dataset,
        args.original_audit,
        expected_pairs=args.expected_original_pairs,
        label="original Exp15",
    )
    consistent_rows, consistent_audit = require_audited_dataset(
        args.consistent_dataset,
        args.consistent_audit,
        expected_pairs=args.expected_consistent_pairs,
        label="non-overlap online-consistent branches",
    )
    if consistent_audit.get("new_model_generation_performed") is not False:
        raise SystemExit("consistent28 audit does not declare frozen-data materialization")

    original_hashes = {str(row["pair_sha256"]) for row in original_rows}
    consistent_hashes = {str(row["pair_sha256"]) for row in consistent_rows}
    overlap = original_hashes & consistent_hashes
    if overlap:
        raise SystemExit(f"union inputs unexpectedly overlap by {len(overlap)} pairs")

    rows = sorted(
        original_rows + consistent_rows,
        key=lambda row: (
            int(row["example_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        ),
    )
    validate_rows(rows, label="original Exp15 plus consistent28 union")
    questions = {str(row["question_id"]) for row in rows}
    if len(rows) != args.expected_union_pairs:
        raise SystemExit(
            f"expected {args.expected_union_pairs} union pairs, found {len(rows)}"
        )
    if len(questions) != args.expected_union_questions:
        raise SystemExit(
            f"expected {args.expected_union_questions} union questions, found {len(questions)}"
        )

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dataset, rows)
    audit = {
        "schema_version": "original-plus-consistent-action-replay-audit-v1",
        "status": "passed",
        "training_admission": "user_authorized_original_exp15_plus_existing_consistent28",
        "split": "bird-train",
        "protocol_version": "version26",
        "verified_pairs": len(rows),
        "questions": len(questions),
        "optimizer_steps_expected": len(questions),
        "original_pairs": len(original_rows),
        "consistent_pairs": len(consistent_rows),
        "overlap_pairs": 0,
        "real_harness_execution_required": True,
        "positive_suffix_replay_required": True,
        "source_trajectory_replayed_required": True,
        "exact_visible_prefix_required": True,
        "state_normalization": "none",
        "denotation_comparison": "bird-set",
        "gold_sql_model_visible": False,
        "new_model_generation_performed": False,
        "original_dataset": str(args.original_dataset.resolve()),
        "original_dataset_sha256": sha256_file(args.original_dataset),
        "original_audit_sha256": sha256_file(args.original_audit),
        "consistent_dataset": str(args.consistent_dataset.resolve()),
        "consistent_dataset_sha256": sha256_file(args.consistent_dataset),
        "consistent_audit_sha256": sha256_file(args.consistent_audit),
        "output": str(args.output_dataset.resolve()),
        "output_sha256": sha256_file(args.output_dataset),
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
