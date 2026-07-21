#!/usr/bin/env python3
"""Select a deterministic, quality-bounded BIRD teacher subset for small SFT runs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import ProtocolError, TOOLS, validate_model_arguments  # noqa: E402

STEP_REF = re.compile(r"step_(\d+)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_repeated_call(episode: dict[str, Any]) -> bool:
    calls: set[str] = set()
    for step in episode.get("steps") or []:
        call = step.get("tool_call") or {}
        key = json.dumps(
            {"tool": call.get("tool"), "arguments": call.get("arguments")},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in calls:
            return True
        calls.add(key)
    return False


def has_current_or_future_reference(value: Any, current_step_index: int) -> bool:
    """Check only structured provenance fields, never incidental step text in prose."""
    if isinstance(value, list):
        return any(has_current_or_future_reference(item, current_step_index) for item in value)
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in {"value_ref", "evidence", "evidence_step_id", "source_step_id"} and isinstance(item, str):
            match = STEP_REF.fullmatch(item)
            if match and int(match.group(1)) >= current_step_index:
                return True
        if has_current_or_future_reference(item, current_step_index):
            return True
    return False


def quality_reason(
    episode: dict[str, Any],
    *,
    max_steps: int,
    max_think_words: int,
) -> str | None:
    if episode.get("label_status") != "verified":
        return "not_verified"
    steps = episode.get("steps") or []
    if not steps or len(steps) > max_steps:
        return "step_limit"
    if has_repeated_call(episode):
        return "repeated_call"
    if any(len(str(step.get("think") or "").split()) > max_think_words for step in steps):
        return "think_limit"
    try:
        for step in steps:
            call = step.get("tool_call") or {}
            tool = call.get("tool")
            arguments = call.get("arguments")
            if tool not in TOOLS or not isinstance(arguments, dict):
                return "current_protocol_incompatible"
            validate_model_arguments(tool, arguments)
            try:
                current_step_index = int(str(step.get("step_id")).removeprefix("step_"))
            except (TypeError, ValueError):
                return "current_protocol_incompatible"
            if has_current_or_future_reference(arguments, current_step_index):
                return "current_or_future_reference"
    except ProtocolError:
        return "current_protocol_incompatible"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-think-words", type=int, default=300)
    args = parser.parse_args()
    if args.total <= 0 or args.total % 10:
        parser.error("--total must be a positive multiple of 10 for the 4:3:3 split")

    source = args.input.resolve()
    out = args.out.resolve()
    manifest_path = (args.manifest or out.with_suffix(".manifest.json")).resolve()
    unit = args.total // 10
    quotas = {"easy": 4 * unit, "medium": 3 * unit, "hard": 3 * unit}
    eligible: dict[str, list[dict[str, Any]]] = {difficulty: [] for difficulty in quotas}
    rejected = collections.Counter()
    for episode in read_jsonl(source):
        reason = quality_reason(
            episode,
            max_steps=args.max_steps,
            max_think_words=args.max_think_words,
        )
        if reason:
            rejected[reason] += 1
            continue
        difficulty = episode.get("difficulty")
        if difficulty in eligible:
            eligible[difficulty].append(episode)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for difficulty, quota in quotas.items():
        bucket = sorted(eligible[difficulty], key=lambda item: item["trajectory_id"])
        rng.shuffle(bucket)
        if len(bucket) < quota:
            raise ValueError(f"need {quota} {difficulty} episodes; only {len(bucket)} pass quality gates")
        selected.extend(bucket[:quota])

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for episode in selected:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    manifest = {
        "input": str(source),
        "input_sha256": file_sha256(source),
        "output": str(out),
        "seed": args.seed,
        "quotas": quotas,
        "quality_gates": {
            "label_status": "verified",
            "max_steps": args.max_steps,
            "max_think_words": args.max_think_words,
            "repeated_identical_calls": "reject",
        },
        "eligible_by_difficulty": {key: len(value) for key, value in eligible.items()},
        "rejected": dict(sorted(rejected.items())),
        "selected_by_difficulty": dict(sorted(collections.Counter(item["difficulty"] for item in selected).items())),
        "selected_trajectory_ids": [item["trajectory_id"] for item in selected],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
