#!/usr/bin/env python3
"""Extract exact-visible-prefix action preferences from a frozen BIRD-train pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl" / "diagnostics"))
from build_fixed_prefix_dataset import (  # noqa: E402
    ASSISTANT_REASONING_NORMALIZATION,
    Occurrence,
    action_of,
    build_candidates,
    canonical_json,
    normalize_assistant_reasoning,
    turn_error,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def collect_pool_occurrences(
    path: Path,
    *,
    normalize_history_reasoning: bool = False,
) -> list[Occurrence]:
    occurrences = []
    for row in load_jsonl(path):
        sample = row["sample"]
        audit = sample["audit_record"]
        for turn in audit.get("turns") or []:
            action = action_of(turn)
            state = turn.get("model_input")
            if action is None or not isinstance(state, list) or not state:
                continue
            if normalize_history_reasoning:
                state = normalize_assistant_reasoning(state)
            occurrences.append(
                Occurrence(
                    artifact=str(path.resolve()),
                    example_index=int(row["environment"]["example_index"]),
                    sample_index=int(audit["sample_index"]),
                    turn_index=int(turn.get("turn_index") or 0),
                    db_id=str(row["environment"]["db_id"]),
                    question=str(row["environment"]["question"]),
                    model_input=state,
                    action=action,
                    trajectory_correct=bool(sample["correct"]),
                    trajectory_legal=bool(audit.get("legal")),
                    feedback_recovery=bool(turn.get("feedback_recovery")),
                    explicit_error=turn_error(turn),
                )
            )
    return occurrences


def select_question_balanced(
    candidates: list[dict[str, Any]],
    *,
    max_pairs_per_question: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_pairs_per_question < 1:
        raise ValueError("max_pairs_per_question must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["question_id"])].append(row)
    selected = []
    for question_id in sorted(grouped):
        ranked = sorted(
            grouped[question_id],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['pair_sha256']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:max_pairs_per_question])
    return sorted(selected, key=lambda row: (int(row["example_index"]), row["state_sha256"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-pairs-per-question", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--normalize-history-reasoning",
        action="store_true",
        help=(
            "replace authored <think> contents in historical assistant messages "
            "with one constant placeholder before state matching and training"
        ),
    )
    args = parser.parse_args()

    frozen = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    if frozen.get("status") not in {"frozen_passed", "frozen_rank_ready"}:
        raise SystemExit(
            "Action-DPO extraction requires an immutable rank-ready or fully passed pool"
        )
    expected_pool_sha = (frozen.get("files", {}).get("validated_trajectories") or {}).get(
        "sha256"
    )
    actual_pool_sha = hashlib.sha256(args.pool.read_bytes()).hexdigest()
    if expected_pool_sha != actual_pool_sha:
        raise SystemExit("Action-DPO source pool hash does not match frozen manifest")

    occurrences = collect_pool_occurrences(
        args.pool,
        normalize_history_reasoning=args.normalize_history_reasoning,
    )
    candidates = build_candidates(occurrences)
    if args.normalize_history_reasoning:
        for row in candidates:
            row["state_normalization"] = ASSISTANT_REASONING_NORMALIZATION
    selected = select_question_balanced(
        candidates,
        max_pairs_per_question=args.max_pairs_per_question,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("the fixed pool contains no verifier-eligible same-state pairs")
    if any("gold_sql" in canonical_json(row["state"]) for row in selected):
        raise SystemExit("model-visible Action-DPO state unexpectedly contains gold_sql")
    write_jsonl(args.output, selected)
    manifest = {
        "schema_version": "fixed-prefix-train-extraction-v1",
        "split": "bird-train",
        "source_pool": str(args.pool.resolve()),
        "source_pool_sha256": actual_pool_sha,
        "source_pool_manifest_sha256": hashlib.sha256(
            args.pool_manifest.read_bytes()
        ).hexdigest(),
        "occurrences": len(occurrences),
        "candidate_pairs": len(candidates),
        "selected_pairs": len(selected),
        "questions": len({row["question_id"] for row in selected}),
        "max_pairs_per_question": args.max_pairs_per_question,
        "seed": args.seed,
        "same_visible_prefix_required": True,
        "state_normalization": (
            ASSISTANT_REASONING_NORMALIZATION
            if args.normalize_history_reasoning
            else "none"
        ),
        "authored_reasoning_used_for_state_matching": False
        if args.normalize_history_reasoning
        else True,
        "dev300_used_for_training": False,
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
