#!/usr/bin/env python3
"""Freeze the non-overlapping stable branch pairs as a standalone DPO set.

This is a materialization step over already executed and verified branches.  It does not
generate model outputs or execute new trajectories.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_expanded_exact_prefix_union import (
    sha256_file,
    validate_rows,
    write_json_atomic,
    write_jsonl_atomic,
)
from build_strict_exact_prefix_training_set import (
    NONSEMANTIC_FAILURES,
    describe_table_reorder_only,
)
from train_fixed_prefix_action_dpo import load_jsonl


def rejection_reason(row: dict[str, Any], *, min_trials: int) -> str | None:
    verification = row.get("positive_verified_by") or {}
    positive_trials = int(verification.get("branching_online_trials") or 0)
    positive_successes = int(verification.get("branching_online_successes") or 0)
    if positive_trials < min_trials:
        return "insufficient_positive_trials"
    if positive_successes != positive_trials:
        return "positive_not_all_successful"

    negative = row.get("negative_observed_from") or {}
    negative_trials = int(negative.get("online_trials") or 0)
    negative_successes = int(negative.get("online_successes") or 0)
    if negative_trials < min_trials:
        return "insufficient_negative_trials"
    if negative_successes != 0:
        return "negative_not_all_failed"
    failure_counts = {
        str(key): int(value)
        for key, value in (negative.get("failure_types") or {}).items()
        if int(value) > 0
    }
    if set(failure_counts) & NONSEMANTIC_FAILURES:
        return "negative_has_nonsemantic_failure"
    if sum(failure_counts.values()) != negative_trials:
        return "negative_failure_count_mismatch"
    if describe_table_reorder_only(row):
        return "describe_table_reorder_only"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset", required=True, type=Path)
    parser.add_argument("--original-audit", required=True, type=Path)
    parser.add_argument("--consistent-dataset", required=True, type=Path)
    parser.add_argument("--consistent-manifest", required=True, type=Path)
    parser.add_argument("--source-replay-audit", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--min-online-trials", type=int, default=2)
    parser.add_argument("--expected-pairs", type=int, default=28)
    parser.add_argument("--expected-questions", type=int, default=19)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite frozen non-overlap artifacts")
    if args.min_online_trials < 2:
        raise SystemExit("online-consistent pairs require at least two trials")

    original_audit = json.loads(args.original_audit.read_text(encoding="utf-8"))
    if original_audit.get("status") != "passed":
        raise SystemExit("original Exp15 verification audit is not passed")
    if original_audit.get("output_sha256") != sha256_file(args.original_dataset):
        raise SystemExit("original Exp15 dataset hash differs from its audit")

    manifest = json.loads(args.consistent_manifest.read_text(encoding="utf-8"))
    required_manifest = {
        "status": "generated_verified_diagnostic",
        "split": "bird-train",
        "protocol_version": "version26",
        "denotation_comparison": "bird-set",
        "state_normalization": "none",
        "same_visible_prefix_required": True,
        "gold_sql_model_visible": False,
        "model_parameter_updates_during_generation": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"consistent manifest gate failed for {key}")
    recorded = (
        (manifest.get("files") or {}).get("pairs_online_consistent") or {}
    ).get("sha256")
    if recorded != sha256_file(args.consistent_dataset):
        raise SystemExit("online-consistent dataset hash differs from its manifest")
    continuation_count = int((manifest.get("settings") or {}).get("continuation_count") or 0)
    if continuation_count < args.min_online_trials:
        raise SystemExit("generation continuation count is below the requested gate")

    replay = json.loads(args.source_replay_audit.read_text(encoding="utf-8"))
    attempted_anchors = int(manifest.get("attempted_anchors") or 0)
    if replay.get("status") != "passed":
        raise SystemExit("source replay audit is not passed")
    if replay.get("anchors") != attempted_anchors:
        raise SystemExit("source replay anchor count differs from the manifest")
    if replay.get("correct_legal_suffixes") != attempted_anchors:
        raise SystemExit("not every source suffix replayed correctly and legally")

    original_rows = load_jsonl(args.original_dataset)
    consistent_rows = load_jsonl(args.consistent_dataset)
    validate_rows(original_rows, label="original Exp15")
    validate_rows(consistent_rows, label="online-consistent branches")
    original_hashes = {str(row["pair_sha256"]) for row in original_rows}

    rejection_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in consistent_rows:
        if str(row["pair_sha256"]) in original_hashes:
            rejection_counts["overlap_with_original_exp15"] += 1
            continue
        reason = rejection_reason(row, min_trials=args.min_online_trials)
        if reason:
            rejection_counts[reason] += 1
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            int(row["example_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        )
    )
    validate_rows(selected, label="non-overlap online-consistent branches")
    questions = {str(row["question_id"]) for row in selected}
    if len(selected) != args.expected_pairs:
        raise SystemExit(
            f"expected {args.expected_pairs} pairs, found {len(selected)}"
        )
    if len(questions) != args.expected_questions:
        raise SystemExit(
            f"expected {args.expected_questions} questions, found {len(questions)}"
        )

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dataset, selected)
    audit = {
        "schema_version": "nonoverlap-online-consistent-action-replay-audit-v1",
        "status": "passed",
        "training_admission": "user_authorized_existing_28_pair_fast_validation",
        "split": "bird-train",
        "protocol_version": "version26",
        "verified_pairs": len(selected),
        "questions": len(questions),
        "optimizer_steps_expected": len(questions),
        "consistent_input_pairs": len(consistent_rows),
        "original_exp15_pairs": len(original_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "minimum_online_trials": args.min_online_trials,
        "positive_online_successes": "all",
        "negative_online_successes": "zero",
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
        "consistent_dataset": str(args.consistent_dataset.resolve()),
        "consistent_dataset_sha256": sha256_file(args.consistent_dataset),
        "consistent_manifest_sha256": sha256_file(args.consistent_manifest),
        "source_replay_audit_sha256": sha256_file(args.source_replay_audit),
        "output": str(args.output_dataset.resolve()),
        "output_sha256": sha256_file(args.output_dataset),
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
