#!/usr/bin/env python3
"""Freeze the unused hard stratum for schema-context diagnostics.

Selection uses only source position and model-independent difficulty metadata. Historical model
outcomes may be joined only after the selected IDs are frozen, and are reference statistics rather
than selection features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SELECTION_METHOD = "unused-hard-stratum-no-model-outcome-v1"
FORBIDDEN_SELECTION_FIELDS = (
    "gold_sql",
    "correct",
    "legal",
    "failure_type",
    "model output",
    "tool trajectory",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_hard_stratum(
    examples: list[dict],
    *,
    exclude_prefix: int,
    difficulty: str,
    minimum_complexity_score: int,
) -> list[tuple[int, dict]]:
    if exclude_prefix < 0:
        raise ValueError("exclude_prefix must be non-negative")
    selected = []
    seen_ids: set[str] = set()
    for source_index, example in enumerate(examples):
        example_id = str(example.get("example_id") or example.get("instance_id") or "")
        if not example_id:
            raise ValueError(f"source example {source_index} lacks a stable ID")
        if example_id in seen_ids:
            raise ValueError(f"duplicate source example ID: {example_id}")
        seen_ids.add(example_id)
        if source_index < exclude_prefix:
            continue
        metadata = example.get("metadata") or {}
        features = metadata.get("difficulty_proxy_features") or {}
        if metadata.get("difficulty_proxy") != difficulty:
            continue
        if int(features.get("score", -1)) < minimum_complexity_score:
            continue
        selected.append((source_index, example))
    return selected


def summarize_reference(
    selected: list[tuple[int, dict]],
    reference_paths: list[Path],
) -> dict:
    records: dict[str, dict] = {}
    for path in reference_paths:
        for record in read_jsonl(path):
            trajectory_id = str(record.get("trajectory_id") or "")
            if not trajectory_id:
                raise ValueError(f"reference record lacks trajectory_id: {path}")
            if trajectory_id in records:
                raise ValueError(f"duplicate reference trajectory_id: {trajectory_id}")
            records[trajectory_id] = record

    selected_ids = [
        str(example.get("example_id") or example.get("instance_id"))
        for _, example in selected
    ]
    missing = [example_id for example_id in selected_ids if example_id not in records]
    if missing:
        raise ValueError(f"reference baseline lacks {len(missing)} selected IDs")
    chosen = [records[example_id] for example_id in selected_ids]
    correct_ids = [
        example_id
        for example_id, record in zip(selected_ids, chosen)
        if record.get("correct")
    ]
    failure_ids = [
        example_id
        for example_id, record in zip(selected_ids, chosen)
        if not record.get("correct")
    ]
    metrics = sorted(
        {
            str(record.get("denotation_comparison"))
            for record in chosen
            if record.get("denotation_comparison")
        }
    )
    return {
        "role": "post_selection_reference_only",
        "paths": [str(path) for path in reference_paths],
        "denotation_metrics": metrics,
        "count": len(chosen),
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / max(1, len(chosen)),
        "legal": sum(bool(record.get("legal")) for record in chosen),
        "process_errors": sum(int(record.get("errors") or 0) for record in chosen),
        "failure_types": dict(
            sorted(
                Counter(
                    str(record.get("failure_type"))
                    for record in chosen
                    if record.get("failure_type")
                ).items()
            )
        ),
        "correct_control_ids": correct_ids,
        "failure_target_ids": failure_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-prefix", type=int, default=50)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--minimum-complexity-score", type=int, default=6)
    parser.add_argument("--expected-count", type=int, default=60)
    parser.add_argument("--reference-baseline", type=Path, action="append", default=[])
    args = parser.parse_args()

    examples = read_jsonl(args.source_examples)
    selected = select_hard_stratum(
        examples,
        exclude_prefix=args.exclude_prefix,
        difficulty=args.difficulty,
        minimum_complexity_score=args.minimum_complexity_score,
    )
    if len(selected) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} selected tasks, found {len(selected)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, example in selected:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    indices = [source_index for source_index, _ in selected]
    scores = [
        int(example["metadata"]["difficulty_proxy_features"]["score"])
        for _, example in selected
    ]
    manifest = {
        "selector": "src/eval/select_schema_context_hard_cohort.py",
        "selection_method": SELECTION_METHOD,
        "source_examples": str(args.source_examples),
        "source_sha256": sha256_file(args.source_examples),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "source_count": len(examples),
        "selected_count": len(selected),
        "selection_fields": [
            "source_index >= exclude_prefix",
            "metadata.difficulty_proxy",
            "metadata.difficulty_proxy_features.score",
        ],
        "forbidden_selection_fields": list(FORBIDDEN_SELECTION_FIELDS),
        "selection_conditioned_on_model_outcome": False,
        "exclude_prefix": args.exclude_prefix,
        "difficulty": args.difficulty,
        "minimum_complexity_score": args.minimum_complexity_score,
        "source_indices": indices,
        "source_index_min": min(indices),
        "source_index_max": max(indices),
        "complexity_score_histogram": dict(sorted(Counter(scores).items())),
        "distinct_databases": len(
            {str(example.get("db_id")) for _, example in selected}
        ),
        "database_histogram": dict(
            sorted(Counter(str(example.get("db_id")) for _, example in selected).items())
        ),
        "selected_ids": [
            str(example.get("example_id") or example.get("instance_id"))
            for _, example in selected
        ],
        "reporting_scope": (
            "selection-defined hard diagnostic cohort; not an unbiased BIRD-wide accuracy estimate"
        ),
    }
    if args.reference_baseline:
        # The selected IDs are frozen above before any model outcome file is opened.
        manifest["historical_reference_baseline"] = summarize_reference(
            selected,
            args.reference_baseline,
        )

    manifest_path = args.output.with_suffix(args.output.suffix + ".selection.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
