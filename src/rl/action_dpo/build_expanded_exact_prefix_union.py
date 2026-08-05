#!/usr/bin/env python3
"""Freeze a verified union of the original Exp15 pairs and exact-prefix branches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from train_fixed_prefix_action_dpo import load_jsonl, sha256_file, validate_dataset


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_sha256(state: Any) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def forbidden_visible_field(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"gold_sql", "gold_query", "reference_sql"}:
                return str(key)
            nested = forbidden_visible_field(child)
            if nested:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = forbidden_visible_field(child)
            if nested:
                return nested
    return None


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


def validate_rows(rows: list[dict[str, Any]], *, label: str) -> None:
    validate_dataset(rows)
    for row in rows:
        if row.get("state_normalization", "none") != "none":
            raise ValueError(f"{label} contains a normalized rather than exact state")
        if row.get("state_sha256") != state_sha256(row.get("state")):
            raise ValueError(f"{label} state hash mismatch: {row.get('pair_sha256')}")
        forbidden = forbidden_visible_field(row.get("state"))
        if forbidden:
            raise ValueError(f"{label} model-visible state contains {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset", required=True, type=Path)
    parser.add_argument("--original-audit", required=True, type=Path)
    parser.add_argument("--expanded-dataset", required=True, type=Path)
    parser.add_argument("--expanded-manifest", required=True, type=Path)
    parser.add_argument("--source-replay-audit", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument("--min-expanded-pairs", type=int, default=36)
    parser.add_argument("--min-expanded-questions", type=int, default=20)
    parser.add_argument("--min-new-unique-pairs", type=int, default=24)
    args = parser.parse_args()

    if args.output_dataset.exists() or args.output_audit.exists():
        raise SystemExit("refusing to overwrite a frozen expanded Exp15 artifact")

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
    expected_expanded_sha = (
        (expanded_manifest.get("files") or {}).get("pairs") or {}
    ).get("sha256")
    if expected_expanded_sha != sha256_file(args.expanded_dataset):
        raise SystemExit("expanded branch dataset hash differs from its manifest")

    source_replay = json.loads(args.source_replay_audit.read_text(encoding="utf-8"))
    expected_anchors = int(expanded_manifest.get("attempted_anchors") or 0)
    if source_replay.get("status") != "passed":
        raise SystemExit("full source-prefix replay audit is not passed")
    if source_replay.get("protocol_version") != "version26":
        raise SystemExit("full source-prefix replay audit is not version26")
    if source_replay.get("denotation_comparison") != "bird-set":
        raise SystemExit("full source-prefix replay audit does not use bird-set")
    if source_replay.get("state_normalization") != "none":
        raise SystemExit("full source-prefix replay audit normalized visible states")
    if source_replay.get("anchors") != expected_anchors:
        raise SystemExit("full source-prefix replay audit anchor count mismatch")
    if source_replay.get("correct_legal_suffixes") != expected_anchors:
        raise SystemExit("not every full-scale source suffix replayed correctly and legally")

    original_rows = load_jsonl(args.original_dataset)
    expanded_rows = load_jsonl(args.expanded_dataset)
    validate_rows(original_rows, label="original Exp15")
    validate_rows(expanded_rows, label="expanded exact-prefix branch")

    expanded_questions = {str(row["question_id"]) for row in expanded_rows}
    if len(expanded_rows) < args.min_expanded_pairs:
        raise SystemExit(
            f"scale gate failed: expanded pairs {len(expanded_rows)} < {args.min_expanded_pairs}"
        )
    if len(expanded_questions) < args.min_expanded_questions:
        raise SystemExit(
            "scale gate failed: expanded questions "
            f"{len(expanded_questions)} < {args.min_expanded_questions}"
        )

    original_hashes = {str(row["pair_sha256"]) for row in original_rows}
    new_unique = [
        row for row in expanded_rows if str(row["pair_sha256"]) not in original_hashes
    ]
    if len(new_unique) < args.min_new_unique_pairs:
        raise SystemExit(
            f"scale gate failed: new unique pairs {len(new_unique)} "
            f"< {args.min_new_unique_pairs}"
        )

    union_by_hash = {str(row["pair_sha256"]): row for row in original_rows}
    for row in expanded_rows:
        union_by_hash.setdefault(str(row["pair_sha256"]), row)
    rows = sorted(
        union_by_hash.values(),
        key=lambda row: (
            int(row["example_index"]),
            str(row["state_sha256"]),
            str(row["pair_sha256"]),
        ),
    )
    validate_rows(rows, label="expanded Exp15 union")
    write_jsonl_atomic(args.output_dataset, rows)

    audit = {
        "schema_version": "expanded-exact-prefix-action-replay-audit-v1",
        "status": "passed",
        "training_admission": "user_authorized_exp15_scale_gate_passed",
        "split": "bird-train",
        "protocol_version": "version26",
        "input_pairs": len(original_rows) + len(expanded_rows),
        "verified_pairs": len(rows),
        "rejected_pairs": 0,
        "questions": len({str(row["question_id"]) for row in rows}),
        "original_pairs": len(original_rows),
        "expanded_branch_pairs": len(expanded_rows),
        "expanded_branch_questions": len(expanded_questions),
        "overlap_pairs": len(expanded_rows) - len(new_unique),
        "new_unique_pairs": len(new_unique),
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
        "scale_thresholds": {
            "min_expanded_pairs": args.min_expanded_pairs,
            "min_expanded_questions": args.min_expanded_questions,
            "min_new_unique_pairs": args.min_new_unique_pairs,
        },
    }
    write_json_atomic(args.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
