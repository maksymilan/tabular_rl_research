#!/usr/bin/env python3
"""Build a controlled Exp15 union from stable online exact-prefix branches.

The original verified Exp15 rows remain the control backbone.  Every admitted new
row must use an original training question, start after turn zero, make the recorded
source action succeed in every online continuation, and make the alternative action
fail in every continuation.  Deterministic per-question/stage caps keep the number of
optimizer steps and the training-question distribution identical to original Exp15.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_expanded_exact_prefix_union import (
    canonical_json,
    sha256_file,
    validate_rows,
    write_json_atomic,
    write_jsonl_atomic,
)
from train_fixed_prefix_action_dpo import load_jsonl


NONSEMANTIC_FAILURES = {
    "api_error",
    "context_overflow",
    "generation_oom",
    "provider_error",
    "transport_error",
}


def stage_band(turn_index: int) -> str:
    if turn_index == 0:
        return "t0"
    if turn_index <= 3:
        return "t1_3"
    return "t4_plus"


def describe_table_reorder_only(row: dict[str, Any]) -> bool:
    positive = row.get("positive_action") or {}
    negative = row.get("negative_action") or {}
    if positive.get("tool") != "describe_table" or negative.get("tool") != "describe_table":
        return False
    positive_tables = (positive.get("arguments") or {}).get("tables")
    negative_tables = (negative.get("arguments") or {}).get("tables")
    return (
        isinstance(positive_tables, list)
        and isinstance(negative_tables, list)
        and positive_tables != negative_tables
        and sorted(map(str, positive_tables)) == sorted(map(str, negative_tables))
    )


def strict_gate_reason(
    row: dict[str, Any],
    *,
    original_questions: set[str],
    original_hashes: set[str],
    min_online_trials: int,
    min_turn_index: int,
) -> str | None:
    if str(row.get("pair_sha256")) in original_hashes:
        return "overlap_with_original"
    if str(row.get("question_id")) not in original_questions:
        return "question_not_in_original_exp15"
    verification = row.get("positive_verified_by") or {}
    source = verification.get("source") or {}
    turn_index = int(source.get("turn_index", -1))
    if turn_index < min_turn_index:
        return "before_minimum_stage"
    positive_trials = int(verification.get("branching_online_trials") or 0)
    positive_successes = int(verification.get("branching_online_successes") or 0)
    if positive_trials < min_online_trials:
        return "insufficient_positive_trials"
    if positive_successes != positive_trials:
        return "positive_not_all_successful"
    negative = row.get("negative_observed_from") or {}
    negative_trials = int(negative.get("online_trials") or 0)
    negative_successes = int(negative.get("online_successes") or 0)
    if negative_trials < min_online_trials:
        return "insufficient_negative_trials"
    if negative_successes != 0:
        return "negative_not_all_failed"
    failure_types = {
        str(key) for key, value in (negative.get("failure_types") or {}).items() if int(value) > 0
    }
    if failure_types & NONSEMANTIC_FAILURES:
        return "negative_has_nonsemantic_failure"
    if sum(int(value) for value in (negative.get("failure_types") or {}).values()) != negative_trials:
        return "negative_failure_count_mismatch"
    if describe_table_reorder_only(row):
        return "describe_table_reorder_only"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset", required=True, type=Path)
    parser.add_argument("--original-audit", required=True, type=Path)
    parser.add_argument("--expanded-dataset", required=True, type=Path)
    parser.add_argument("--expanded-manifest", required=True, type=Path)
    parser.add_argument("--expanded-file-key", default="pairs_online_consistent")
    parser.add_argument("--source-replay-audit", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--min-online-trials", type=int, default=4)
    parser.add_argument("--min-turn-index", type=int, default=1)
    parser.add_argument("--max-new-pairs-per-question-stage", type=int, default=1)
    parser.add_argument("--max-new-pairs-per-question", type=int, default=2)
    parser.add_argument("--min-new-pairs", type=int, default=8)
    parser.add_argument("--min-new-questions", type=int, default=6)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite a frozen strict Exp15 artifact")
    if args.min_online_trials < 2 or args.min_turn_index < 1:
        raise SystemExit("strict Exp15 requires repeated online evidence after turn zero")
    if (
        args.max_new_pairs_per_question_stage < 1
        or args.max_new_pairs_per_question < 1
    ):
        raise SystemExit("strict Exp15 caps must be positive")

    original_audit = json.loads(args.original_audit.read_text(encoding="utf-8"))
    if original_audit.get("status") != "passed":
        raise SystemExit("original Exp15 verification audit is not passed")
    if original_audit.get("output_sha256") != sha256_file(args.original_dataset):
        raise SystemExit("original Exp15 dataset hash differs from its audit")

    expanded_manifest = json.loads(args.expanded_manifest.read_text(encoding="utf-8"))
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
        if expanded_manifest.get(key) != expected:
            raise SystemExit(f"expanded manifest gate failed for {key}")
    settings = expanded_manifest.get("settings") or {}
    if int(settings.get("continuation_count") or 0) < args.min_online_trials:
        raise SystemExit("generation used fewer continuations than the strict gate")
    expected_expanded_sha = (
        (expanded_manifest.get("files") or {}).get(args.expanded_file_key) or {}
    ).get("sha256")
    if expected_expanded_sha != sha256_file(args.expanded_dataset):
        raise SystemExit("strict branch dataset hash differs from its manifest")

    source_replay = json.loads(args.source_replay_audit.read_text(encoding="utf-8"))
    expected_anchors = int(expanded_manifest.get("attempted_anchors") or 0)
    if source_replay.get("status") != "passed":
        raise SystemExit("source-prefix replay audit is not passed")
    if source_replay.get("anchors") != expected_anchors:
        raise SystemExit("source-prefix replay anchor count mismatch")
    if source_replay.get("correct_legal_suffixes") != expected_anchors:
        raise SystemExit("not every selected source suffix replayed correctly and legally")

    original_rows = load_jsonl(args.original_dataset)
    expanded_rows = load_jsonl(args.expanded_dataset)
    validate_rows(original_rows, label="original Exp15")
    validate_rows(expanded_rows, label="strict exact-prefix candidates")
    original_questions = {str(row["question_id"]) for row in original_rows}
    original_hashes = {str(row["pair_sha256"]) for row in original_rows}
    question_filter = expanded_manifest.get("question_filter") or {}
    if question_filter.get("dataset_sha256") != sha256_file(args.original_dataset):
        raise SystemExit("generation question filter is not the frozen original Exp15 dataset")
    if int(question_filter.get("questions") or 0) != len(original_questions):
        raise SystemExit("generation question-filter coverage differs from original Exp15")

    rejection_counts: Counter[str] = Counter()
    eligible = []
    for row in expanded_rows:
        reason = strict_gate_reason(
            row,
            original_questions=original_questions,
            original_hashes=original_hashes,
            min_online_trials=args.min_online_trials,
            min_turn_index=args.min_turn_index,
        )
        if reason:
            rejection_counts[reason] += 1
            continue
        eligible.append(row)

    eligible.sort(
        key=lambda row: (
            int(row["example_index"]),
            int((row["positive_verified_by"].get("source") or {})["turn_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        )
    )
    selected = []
    per_question: Counter[str] = Counter()
    per_question_stage: Counter[tuple[str, str]] = Counter()
    for row in eligible:
        question_id = str(row["question_id"])
        turn_index = int(row["positive_verified_by"]["source"]["turn_index"])
        band = stage_band(turn_index)
        if per_question[question_id] >= args.max_new_pairs_per_question:
            rejection_counts["per_question_cap"] += 1
            continue
        if (
            per_question_stage[(question_id, band)]
            >= args.max_new_pairs_per_question_stage
        ):
            rejection_counts["per_question_stage_cap"] += 1
            continue
        selected.append(row)
        per_question[question_id] += 1
        per_question_stage[(question_id, band)] += 1

    selected_questions = {str(row["question_id"]) for row in selected}
    if len(selected) < args.min_new_pairs:
        raise SystemExit(
            f"strict scale gate failed: selected new pairs {len(selected)} < {args.min_new_pairs}"
        )
    if len(selected_questions) < args.min_new_questions:
        raise SystemExit(
            "strict scale gate failed: selected new questions "
            f"{len(selected_questions)} < {args.min_new_questions}"
        )

    union_by_hash = {str(row["pair_sha256"]): row for row in original_rows}
    for row in selected:
        union_by_hash[str(row["pair_sha256"])] = row
    rows = sorted(
        union_by_hash.values(),
        key=lambda row: (
            int(row["example_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        ),
    )
    validate_rows(rows, label="strict controlled Exp15 union")
    if {str(row["question_id"]) for row in rows} != original_questions:
        raise SystemExit("strict union changed the original Exp15 question set")
    write_jsonl_atomic(args.output_dataset, rows)

    stage_counts = Counter(
        stage_band(int(row["positive_verified_by"]["source"]["turn_index"]))
        for row in selected
    )
    tool_counts = Counter(str(row["positive_action"]["tool"]) for row in selected)
    audit = {
        "schema_version": "strict-controlled-exact-prefix-action-replay-audit-v1",
        "status": "passed",
        "training_admission": "user_authorized_strict_online_consistency_gate",
        "split": "bird-train",
        "protocol_version": "version26",
        "verified_pairs": len(rows),
        "questions": len(original_questions),
        "optimizer_steps_expected": len(original_questions),
        "original_pairs": len(original_rows),
        "expanded_input_pairs": len(expanded_rows),
        "eligible_strict_pairs_before_caps": len(eligible),
        "selected_new_pairs": len(selected),
        "selected_new_questions": len(selected_questions),
        "selected_stage_counts": dict(sorted(stage_counts.items())),
        "selected_positive_tools": dict(sorted(tool_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "strict_requirements": {
            "same_original_question_set": True,
            "minimum_turn_index": args.min_turn_index,
            "positive_online_successes": f"all/{args.min_online_trials}+",
            "negative_online_successes": f"0/{args.min_online_trials}+",
            "reject_nonsemantic_negative_failures": True,
            "reject_describe_table_reorder_only": True,
            "max_new_pairs_per_question_stage": args.max_new_pairs_per_question_stage,
            "max_new_pairs_per_question": args.max_new_pairs_per_question,
        },
        "real_harness_execution_required": True,
        "positive_suffix_replay_required": True,
        "source_trajectory_replayed_required": True,
        "exact_visible_prefix_required": True,
        "state_normalization": "none",
        "denotation_comparison": "bird-set",
        "gold_sql_model_visible": False,
        "model_parameter_updates_during_generation": False,
        "original_dataset": str(args.original_dataset.resolve()),
        "original_dataset_sha256": sha256_file(args.original_dataset),
        "original_audit_sha256": sha256_file(args.original_audit),
        "expanded_dataset": str(args.expanded_dataset.resolve()),
        "expanded_dataset_sha256": sha256_file(args.expanded_dataset),
        "expanded_manifest_sha256": sha256_file(args.expanded_manifest),
        "source_replay_audit_sha256": sha256_file(args.source_replay_audit),
        "output": str(args.output_dataset.resolve()),
        "output_sha256": sha256_file(args.output_dataset),
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
