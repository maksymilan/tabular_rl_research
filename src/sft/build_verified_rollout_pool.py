#!/usr/bin/env python3
"""Merge verified rolling rollout files through the current SFT quality contract."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from select_verified_rollouts import quality_reason, read_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-think-words", type=int, default=300)
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.input]
    out = args.out.resolve()
    accepted: dict[str, dict[str, Any]] = {}
    accepted_source: dict[str, str] = {}
    rejected = collections.Counter()
    input_rows = collections.Counter()
    accepted_by_source = collections.Counter()
    duplicate_ids: list[str] = []

    for source in inputs:
        for episode in read_jsonl(source):
            input_rows[source.name] += 1
            reason = quality_reason(
                episode,
                max_steps=args.max_steps,
                max_think_words=args.max_think_words,
            )
            if reason:
                rejected[reason] += 1
                continue
            generation = episode.get("rollout_generation") or {}
            if generation.get("context_mode") != "rolling-legal-history":
                rejected["not_rolling_legal_history"] += 1
                continue
            if generation.get("history_turns") != 4:
                rejected["history_turns_not_4"] += 1
                continue
            episode_id = str(episode.get("trajectory_id") or "")
            if not episode_id:
                rejected["missing_trajectory_id"] += 1
                continue
            if episode_id in accepted:
                duplicate_ids.append(episode_id)
                if accepted[episode_id] != episode:
                    rejected["conflicting_duplicate"] += 1
                else:
                    rejected["exact_duplicate"] += 1
                continue
            accepted[episode_id] = episode
            accepted_source[episode_id] = source.name
            accepted_by_source[source.name] += 1

    rows = [accepted[episode_id] for episode_id in sorted(accepted)]
    write_jsonl(out, rows)
    manifest = {
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path), "rows": input_rows[path.name]}
            for path in inputs
        ],
        "output": str(out),
        "output_sha256": file_sha256(out),
        "quality_gates": {
            "label_status": "verified",
            "max_steps": args.max_steps,
            "max_think_words": args.max_think_words,
            "repeated_identical_calls": "reject",
            "current_model_action_protocol": "required",
            "context_mode": "rolling-legal-history",
            "history_turns": 4,
        },
        "accepted_episodes": len(rows),
        "accepted_step_targets": sum(len(row.get("steps") or []) for row in rows),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "accepted_by_difficulty": dict(sorted(collections.Counter(
            row.get("difficulty") or "unknown" for row in rows
        ).items())),
        "accepted_feedback_recovery_targets": sum(
            bool(step.get("feedback_recovery"))
            for row in rows
            for step in row.get("steps") or []
        ),
        "rejected": dict(sorted(rejected.items())),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "episode_source": accepted_source,
        "training_status": "candidate_only_pending_grounding_completeness_gate",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "episode_source"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
