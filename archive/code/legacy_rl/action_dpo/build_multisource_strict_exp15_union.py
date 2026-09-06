#!/usr/bin/env python3
"""Freeze a verified, pair-deduplicated Exp15 union from audited sources."""
from __future__ import annotations

import argparse
import json
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


def pair_identity(row: dict[str, Any]) -> str:
    return canonical_json(
        {
            "question_id": row.get("question_id"),
            "example_index": row.get("example_index"),
            "db_id": row.get("db_id"),
            "state_sha256": row.get("state_sha256"),
            "positive_action": row.get("positive_action"),
            "negative_action": row.get("negative_action"),
        }
    )


def validate_audited_source(label: str, dataset: Path, attestation: Path) -> tuple[list[dict], dict]:
    audit = json.loads(attestation.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise SystemExit(f"{label}: verification audit is not passed")
    if audit.get("output_sha256") != sha256_file(dataset):
        raise SystemExit(f"{label}: dataset hash differs from verification audit")
    rows = load_jsonl(dataset)
    validate_rows(rows, label=label)
    if int(audit.get("verified_pairs") or 0) != len(rows):
        raise SystemExit(f"{label}: verified pair count mismatch")
    return rows, audit


def validate_branch_source(label: str, dataset: Path, attestation: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(attestation.read_text(encoding="utf-8"))
    required = {
        "status": "generated_verified_diagnostic",
        "split": "bird-train",
        "protocol_version": "version26",
        "denotation_comparison": "bird-set",
        "state_normalization": "none",
        "same_visible_prefix_required": True,
        "gold_sql_model_visible": False,
        "model_parameter_updates_during_generation": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"{label}: branch manifest gate failed for {key}")
    expected_sha = (
        (manifest.get("files") or {}).get("pairs_online_consistent") or {}
    ).get("sha256")
    if expected_sha != sha256_file(dataset):
        raise SystemExit(f"{label}: strict branch dataset hash differs from manifest")
    rows = load_jsonl(dataset)
    validate_rows(rows, label=label)
    for row in rows:
        positive = row.get("positive_verified_by") or {}
        negative = row.get("negative_observed_from") or {}
        trials = int(positive.get("branching_online_trials") or 0)
        positive_success_value = positive.get("branching_online_successes")
        successes = -1 if positive_success_value is None else int(positive_success_value)
        negative_trials = int(negative.get("online_trials") or 0)
        negative_success_value = negative.get("online_successes")
        negative_successes = -1 if negative_success_value is None else int(negative_success_value)
        source_replay = positive.get("source_replay") or {}
        if trials < 2 or successes != trials:
            raise SystemExit(f"{label}: positive is not successful on every online continuation")
        if negative_trials < 2 or negative_successes != 0:
            raise SystemExit(f"{label}: negative is not consistently unsuccessful online")
        if not positive.get("real_harness_execution") or not positive.get("exact_prefix_replay"):
            raise SystemExit(f"{label}: missing real exact-prefix execution evidence")
        if not source_replay.get("correct") or not source_replay.get("legal"):
            raise SystemExit(f"{label}: source suffix replay is not correct and legal")
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audited-source",
        nargs=3,
        action="append",
        metavar=("LABEL", "DATASET", "AUDIT"),
        default=[],
    )
    parser.add_argument(
        "--branch-source",
        nargs=3,
        action="append",
        metavar=("LABEL", "DATASET", "MANIFEST"),
        default=[],
    )
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--minimum-total-pairs", type=int, default=1)
    parser.add_argument("--minimum-total-questions", type=int, default=1)
    parser.add_argument("--minimum-last-source-questions", type=int, default=1)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite frozen multi-source Exp15 union")
    if not args.audited_source and not args.branch_source:
        raise SystemExit("at least one source is required")

    loaded: list[tuple[str, Path, Path, str, list[dict], dict]] = []
    for label, dataset_text, audit_text in args.audited_source:
        dataset, attestation = Path(dataset_text), Path(audit_text)
        rows, metadata = validate_audited_source(label, dataset, attestation)
        loaded.append((label, dataset, attestation, "verified_audit", rows, metadata))
    for label, dataset_text, manifest_text in args.branch_source:
        dataset, attestation = Path(dataset_text), Path(manifest_text)
        rows, metadata = validate_branch_source(label, dataset, attestation)
        loaded.append((label, dataset, attestation, "strict_branch_manifest", rows, metadata))

    union: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    source_summaries = []
    last_source_new_questions: set[str] = set()
    for source_index, (label, dataset, attestation, kind, rows, _metadata) in enumerate(loaded):
        source_questions = {str(row["question_id"]) for row in rows}
        new_pairs = 0
        overlaps = 0
        new_questions: set[str] = set()
        prior_questions = {str(row["question_id"]) for row in union.values()}
        for row in rows:
            pair_hash = str(row["pair_sha256"])
            identity = pair_identity(row)
            if pair_hash in union:
                overlaps += 1
                if identities[pair_hash] != identity:
                    raise SystemExit(f"{label}: pair hash collision with different pair content")
                continue
            union[pair_hash] = row
            identities[pair_hash] = identity
            new_pairs += 1
            if str(row["question_id"]) not in prior_questions:
                new_questions.add(str(row["question_id"]))
        if source_index == len(loaded) - 1:
            last_source_new_questions = new_questions
        source_summaries.append(
            {
                "label": label,
                "kind": kind,
                "dataset": str(dataset.resolve()),
                "dataset_sha256": sha256_file(dataset),
                "attestation": str(attestation.resolve()),
                "attestation_sha256": sha256_file(attestation),
                "input_pairs": len(rows),
                "input_questions": len(source_questions),
                "new_unique_pairs": new_pairs,
                "overlap_pairs": overlaps,
                "new_questions_at_admission": len(new_questions),
            }
        )

    rows = sorted(
        union.values(),
        key=lambda row: (
            int(row["example_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        ),
    )
    validate_rows(rows, label="multi-source strict Exp15 union")
    questions = {str(row["question_id"]) for row in rows}
    if len(rows) < args.minimum_total_pairs:
        raise SystemExit(f"union has only {len(rows)} pairs")
    if len(questions) < args.minimum_total_questions:
        raise SystemExit(f"union has only {len(questions)} questions")
    if len(last_source_new_questions) < args.minimum_last_source_questions:
        raise SystemExit(
            "last source adds only "
            f"{len(last_source_new_questions)} questions; expected at least "
            f"{args.minimum_last_source_questions}"
        )

    write_jsonl_atomic(args.output_dataset, rows)
    audit = {
        "schema_version": "multisource-strict-exp15-action-replay-audit-v1",
        "status": "passed",
        "training_admission": "user_authorized_strict_exp15_question_scale_expansion",
        "split": "bird-train",
        "protocol_version": "version26",
        "verified_pairs": len(rows),
        "questions": len(questions),
        "optimizer_steps_expected": len(questions),
        "input_pairs": sum(len(item[4]) for item in loaded),
        "deduplicated_pairs": sum(len(item[4]) for item in loaded) - len(rows),
        "last_source_new_questions": len(last_source_new_questions),
        "real_harness_execution_required": True,
        "positive_suffix_replay_required": True,
        "source_trajectory_replayed_required": True,
        "exact_visible_prefix_required": True,
        "strict_branch_positive_success": "all_online_trials",
        "strict_branch_negative_success": "zero_online_trials",
        "state_normalization": "none",
        "denotation_comparison": "bird-set",
        "gold_sql_model_visible": False,
        "model_parameter_updates_during_generation": False,
        "sources": source_summaries,
        "output": str(args.output_dataset.resolve()),
        "output_sha256": sha256_file(args.output_dataset),
        "minimums": {
            "total_pairs": args.minimum_total_pairs,
            "total_questions": args.minimum_total_questions,
            "last_source_questions": args.minimum_last_source_questions,
        },
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
