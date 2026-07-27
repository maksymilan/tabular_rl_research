#!/usr/bin/env python3
"""Assemble one preferred, fully verified SFT-2 trajectory per BIRD task."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src" / "eval"), str(ROOT / "src" / "harness"), str(HERE)]

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from select_verified_rollouts import quality_reason  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example_id(trajectory: dict[str, Any]) -> str:
    value = (trajectory.get("source") or {}).get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("trajectory source is missing a non-empty example_id")
    return value


def trajectory_quality_reason(trajectory: dict[str, Any]) -> str | None:
    """Apply hard gates while ignoring repetition in context-only recovery prefixes."""
    if trajectory.get("label_status") != "verified":
        return "not_verified"
    steps = trajectory.get("steps") or []
    if not steps:
        return "empty_trajectory"

    # Check every legal step against the active tool/argument/reference protocol one at a time.
    # A one-step view cannot accidentally turn context-only prefix repetition into rejection.
    for step in steps:
        reason = quality_reason(
            {"label_status": "verified", "steps": [step]},
            max_steps=None,
            max_think_words=None,
        )
        if reason:
            return reason

    targets = [
        step
        for step in steps
        if step.get("sft_target_eligible", True) is not False
    ]
    if not targets:
        return "no_sft_targets"
    # Length limits are intentionally disabled. The only trajectory-shape quality rejection is an
    # exactly repeated eligible tool+arguments call.
    return quality_reason(
        {"label_status": "verified", "steps": targets},
        max_steps=None,
        max_think_words=None,
    )


def parse_lane(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--lane must be NAME=PATH")
    return name, Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", type=parse_lane, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejected-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--abandoned-infrastructure", type=Path)
    parser.add_argument(
        "--reuse-source-replay-attestations",
        action="store_true",
        help=(
            "Do not replay immutable inputs again. Use only when every lane was already admitted "
            "by source-time bird-set deterministic replay."
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if len({name for name, _ in args.lane}) != len(args.lane):
        parser.error("lane names must be unique")
    for path in (args.out, args.rejected_out, args.manifest):
        if path.exists():
            parser.error(f"{path} exists; choose a fresh output path")

    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for priority, (lane, path) in enumerate(args.lane):
        if not path.is_file():
            parser.error(f"lane input does not exist: {path}")
        rows = read_jsonl(path)
        sources.append(
            {
                "lane": lane,
                "priority": priority,
                "path": str(path),
                "sha256": sha256(path),
                "trajectories": len(rows),
            }
        )
        for ordinal, trajectory in enumerate(rows):
            candidates.append(
                {
                    "lane": lane,
                    "priority": priority,
                    "ordinal": ordinal,
                    "trajectory": trajectory,
                }
            )

    structural_rejections: list[dict[str, Any]] = []
    replay_work: list[dict[str, Any]] = []
    for candidate in candidates:
        trajectory = candidate["trajectory"]
        try:
            identity = example_id(trajectory)
            reason = trajectory_quality_reason(trajectory)
        except Exception as exc:  # noqa: BLE001
            identity = str((trajectory.get("source") or {}).get("example_id") or "")
            reason = f"gate_exception:{type(exc).__name__}:{exc}"
        if reason:
            structural_rejections.append(
                {
                    "example_id": identity,
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "lane": candidate["lane"],
                    "reason": reason,
                }
            )
        else:
            candidate["example_id"] = identity
            replay_work.append(candidate)

    replay_rejections: list[dict[str, Any]] = []
    if args.reuse_source_replay_attestations:
        replay_verified = replay_work
    else:
        replay_results: dict[int, tuple[bool, str | None]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    replay_success_trajectory,
                    candidate["trajectory"],
                    denotation_comparison="bird-set",
                ): index
                for index, candidate in enumerate(replay_work)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    replay_results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    replay_results[index] = (False, f"{type(exc).__name__}: {exc}")

        replay_verified = []
        for index, candidate in enumerate(replay_work):
            ok, message = replay_results[index]
            if not ok:
                replay_rejections.append(
                    {
                        "example_id": candidate["example_id"],
                        "trajectory_id": candidate["trajectory"].get("trajectory_id"),
                        "lane": candidate["lane"],
                        "reason": "deterministic_replay_failed",
                        "detail": message,
                    }
                )
            else:
                replay_verified.append(candidate)

    chosen: dict[str, dict[str, Any]] = {}
    lower_priority: list[dict[str, Any]] = []
    for candidate in sorted(
        replay_verified,
        key=lambda item: (item["priority"], item["ordinal"]),
    ):
        identity = candidate["example_id"]
        if identity in chosen:
            lower_priority.append(
                {
                    "example_id": identity,
                    "trajectory_id": candidate["trajectory"].get("trajectory_id"),
                    "lane": candidate["lane"],
                    "reason": "lower_priority_same_task",
                    "preferred_lane": chosen[identity]["lane"],
                    "preferred_trajectory_id": chosen[identity]["trajectory"].get(
                        "trajectory_id"
                    ),
                }
            )
            continue
        chosen[identity] = candidate

    accepted_candidates = sorted(
        chosen.values(),
        key=lambda item: (
            int((item["trajectory"].get("source") or {}).get("example_index", 10**9)),
            item["example_id"],
        ),
    )
    accepted = [item["trajectory"] for item in accepted_candidates]
    rejected = structural_rejections + replay_rejections + lower_priority
    write_jsonl_atomic(args.out.resolve(), accepted)
    write_jsonl_atomic(args.rejected_out.resolve(), rejected)

    abandoned: list[dict[str, Any]] = []
    if args.abandoned_infrastructure:
        abandoned = read_jsonl(args.abandoned_infrastructure.resolve())
    lane_counts = collections.Counter(item["lane"] for item in accepted_candidates)
    rejection_counts = collections.Counter(item["reason"] for item in rejected)
    target_count = sum(
        step.get("sft_target_eligible", True) is not False
        for trajectory in accepted
        for step in trajectory.get("steps") or []
    )
    context_only_count = sum(
        step.get("sft_target_eligible", True) is False
        for trajectory in accepted
        for step in trajectory.get("steps") or []
    )
    manifest = {
        "method": "batch2_preferred_verified_trajectory_assembly",
        "denotation_comparison": "bird-set",
        "priority": [name for name, _ in args.lane],
        "sources": sources,
        "input_candidates": len(candidates),
        "structurally_eligible": len(replay_work),
        "fresh_replay_verified": (
            None if args.reuse_source_replay_attestations else len(replay_verified)
        ),
        "source_replay_attestations_reused": (
            len(replay_verified) if args.reuse_source_replay_attestations else 0
        ),
        "accepted_trajectories": len(accepted),
        "accepted_unique_tasks": len(chosen),
        "accepted_sft_targets": target_count,
        "context_only_prefix_steps": context_only_count,
        "accepted_by_lane": dict(sorted(lane_counts.items())),
        "rejected": len(rejected),
        "rejected_by_reason": dict(sorted(rejection_counts.items())),
        "abandoned_infrastructure_attempts": {
            "count": len(abandoned),
            "path": str(args.abandoned_infrastructure.resolve())
            if args.abandoned_infrastructure
            else None,
            "policy": "excluded; not interpreted as semantic failures or training samples",
        },
        "quality_policy": {
            "deterministic_replay": (
                "reused immutable source-time attestations"
                if args.reuse_source_replay_attestations
                else "fresh assembly-time replay"
            ),
            "bird_set": True,
            "current_protocol_and_reference_safety": True,
            "length_limits": None,
            "repeated_identical_eligible_calls": "reject",
            "one_preferred_trajectory_per_task": True,
            "student_prefix_steps_are_sft_targets": False,
            "error_actions_are_sft_targets": False,
            "gold_sql_visible_in_training_prompt": False,
            "token_gate": "pending exact full-trajectory Qwen2.5-Coder audit at 6400",
        },
        "outputs": {
            "accepted": str(args.out.resolve()),
            "accepted_sha256": sha256(args.out.resolve()),
            "rejected": str(args.rejected_out.resolve()),
            "rejected_sha256": sha256(args.rejected_out.resolve()),
        },
    }
    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.manifest.resolve().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
