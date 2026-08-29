#!/usr/bin/env python3
"""Read-only audit wrapper for the augmented-4k Qwen3 v26 dev rollout.

The implementation is shared with the frozen SFT1 dev/train consistency audit, but the
population contract is bound to the new 1534-record epoch4 evaluation (924 correct/610
incorrect).  This file never exports dev rows or changes rewards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.rl.diagnostics import audit_dev_incorrect_train_consistency as base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-results", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--train-incorrect-rows", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Bind the expected population for this completed epoch4 run without changing the
    # historical audit's frozen 838/696 contract.
    base.EXPECTED_DEV_TOTAL = 1534
    base.EXPECTED_DEV_CORRECT = 924
    base.EXPECTED_DEV_INCORRECT = 610

    summary, cases, scores = base.audit(
        dev_results=args.dev_results,
        dev_manifest=args.dev_manifest,
        database_root=args.database_root,
        train_incorrect_rows=args.train_incorrect_rows,
        train_summary_path=args.train_summary,
    )
    summary["schema_version"] = "bird-dev-augmented-epoch4-error-audit-v1"
    summary["identity"]["evaluation_label"] = "qwen3-v26-augmented-epoch4-cp6380"
    summary["dataset_boundary"]["dev_records_used_for_training"] = 0
    summary["dataset_boundary"]["dev_records_used_for_reward_design_or_selection"] = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "dev_incorrect_casebook.jsonl").open("w", encoding="utf-8") as target:
        for row in cases:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "dev_incorrect_semantic_scores.jsonl").open("w", encoding="utf-8") as target:
        for row in scores:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    base.write_casebook_index(args.output_dir / "dev_incorrect_casebook_index.md", cases)
    generated = [
        "summary.json",
        "dev_incorrect_casebook.jsonl",
        "dev_incorrect_semantic_scores.jsonl",
        "dev_incorrect_casebook_index.md",
    ]
    manifest = {
        "schema_version": "bird-dev-augmented-epoch4-error-audit-manifest-v1",
        "analysis_only": True,
        "dataset_split": "dev",
        "dev_records_used_for_training": 0,
        "reward_admitted": False,
        "model_calls": 0,
        "optimizer_updates": 0,
        "input": {
            "dev_results": str(args.dev_results.resolve()),
            "dev_results_sha256": sha256_file(args.dev_results),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "dev_manifest_sha256": sha256_file(args.dev_manifest),
            "train_incorrect_rows": str(args.train_incorrect_rows.resolve()),
            "train_incorrect_rows_sha256": sha256_file(args.train_incorrect_rows),
            "train_summary": str(args.train_summary.resolve()),
            "train_summary_sha256": sha256_file(args.train_summary),
        },
        "outputs": {name: sha256_file(args.output_dir / name) for name in generated},
        "code": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
