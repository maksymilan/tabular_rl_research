#!/usr/bin/env python3
"""Replay the positive source suffix and verify every Action-DPO training pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "rl" / "diagnostics"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]
from build_fixed_prefix_dataset import (  # noqa: E402
    ASSISTANT_REASONING_NORMALIZATION,
    action_of,
    normalize_assistant_reasoning,
    sha256_json,
)
from external_failure_adapter import normalize_failure_record  # noqa: E402
from trajectory_replay import replay_terminal_evidence  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def task_record(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row["environment"]
    audit = row["sample"]["audit_record"]
    return {
        "example_id": audit["trajectory_id"],
        "dataset": "bird-sql",
        "split": "train",
        "db_id": metadata["db_id"],
        "db_path": metadata["db_path"],
        "question": metadata["question"],
        "gold_sql": metadata["gold_sql"],
        "external_knowledge": metadata.get("external_knowledge"),
        "denotation_comparison": "bird-set",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    pool_rows = load_jsonl(args.pool)
    sources = {
        (
            int(row["environment"]["example_index"]),
            int(row["sample"]["audit_record"]["sample_index"]),
        ): row
        for row in pool_rows
    }
    verified = []
    rejected = []
    replay_cache: dict[tuple[int, int], dict[str, Any]] = {}
    normalization_modes: set[str] = set()
    for pair in load_jsonl(args.pairs):
        source_info = pair["positive_verified_by"]["source"]
        key = (int(source_info["example_index"]), int(source_info["sample_index"]))
        try:
            source_row = sources[key]
            sample = source_row["sample"]
            audit = sample["audit_record"]
            if not sample["correct"] or not audit.get("legal"):
                raise ValueError("positive source trajectory is not correct and legal")
            turn_index = int(source_info["turn_index"])
            turn = next(
                turn
                for turn in audit["turns"]
                if int(turn.get("turn_index") or 0) == turn_index
            )
            normalization = str(pair.get("state_normalization") or "none")
            normalization_modes.add(normalization)
            source_state = turn["model_input"]
            if normalization == ASSISTANT_REASONING_NORMALIZATION:
                source_state = normalize_assistant_reasoning(source_state)
            elif normalization != "none":
                raise ValueError(f"unknown state normalization: {normalization}")
            if sha256_json(source_state) != pair["state_sha256"]:
                raise ValueError("positive source visible prefix hash mismatch")
            if action_of(turn) != pair["positive_action"]:
                raise ValueError("positive source action mismatch")
            if key not in replay_cache:
                normalized, exclusion = normalize_failure_record(audit, task_record(source_row))
                if normalized is None:
                    raise ValueError(f"positive source normalization failed: {exclusion}")
                result = replay_terminal_evidence(
                    normalized,
                    source_row["environment"]["db_path"],
                    denotation_comparison="bird-set",
                )
                replay_cache[key] = result.to_dict()
            replay = replay_cache[key]
            if not replay["correct"] or replay.get("execution_error"):
                raise ValueError("positive source trajectory suffix did not replay correctly")
            checked = json.loads(json.dumps(pair))
            checked["positive_verified_by"].update(
                {
                    "source_trajectory_replayed": True,
                    "source_replay": replay,
                }
            )
            verified.append(checked)
        except Exception as exc:
            rejected.append(
                {
                    "pair_sha256": pair.get("pair_sha256"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not verified:
        raise SystemExit("no Action-DPO pair passed source-suffix replay")
    write_jsonl(args.output, verified)
    audit = {
        "schema_version": "fixed-prefix-action-replay-audit-v1",
        "status": "passed",
        "input_pairs": len(verified) + len(rejected),
        "verified_pairs": len(verified),
        "rejected_pairs": len(rejected),
        "questions": len({row["question_id"] for row in verified}),
        "real_harness_execution_required": True,
        "positive_suffix_replay_required": True,
        "state_normalization_modes": sorted(normalization_modes),
        "denotation_comparison": "bird-set",
        "rejections": rejected,
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
