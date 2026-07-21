#!/usr/bin/env python3
"""Assemble a deduplicated SFT-2 pilot mixture with explicit lane metadata."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def content_key(row: dict[str, Any]) -> str:
    payload = {"system": row.get("system"), "conversations": row.get("conversations")}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tagged(row: dict[str, Any], lane: str, transition: str) -> dict[str, Any]:
    result = json.loads(json.dumps(row))
    result.setdefault("metadata", {})["sft2_lane"] = lane
    result["metadata"]["transition_type"] = transition
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--demo-index", type=Path, required=True)
    parser.add_argument("--demo-count", type=int, default=650)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    demo = read_jsonl(args.demo)
    demo_index = {
        row["record_id"]: row for row in read_jsonl(args.demo_index)
    }
    rng = random.Random(args.seed)
    by_difficulty: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in demo:
        record_id = (row.get("metadata") or {}).get("record_id")
        difficulty = (demo_index.get(record_id) or {}).get("source_difficulty") or "unknown"
        by_difficulty[str(difficulty)].append(row)
    available = sum(len(rows) for rows in by_difficulty.values())
    if not 0 <= args.demo_count <= available:
        raise ValueError(f"demo-count must be within 0..{available}")
    quotas = {
        difficulty: int(args.demo_count * len(rows) / available)
        for difficulty, rows in by_difficulty.items()
    }
    while sum(quotas.values()) < args.demo_count:
        for difficulty in sorted(by_difficulty):
            if quotas[difficulty] < len(by_difficulty[difficulty]):
                quotas[difficulty] += 1
                if sum(quotas.values()) == args.demo_count:
                    break
    demo_selected = []
    for difficulty, rows in sorted(by_difficulty.items()):
        rows = sorted(rows, key=lambda row: row["metadata"]["record_id"])
        rng.shuffle(rows)
        demo_selected.extend(rows[: quotas[difficulty]])

    lanes: list[tuple[str, list[dict[str, Any]]]] = [
        ("decision_correction", read_jsonl(args.correction)),
        ("student_success", read_jsonl(args.student)),
        ("teacher_fallback", read_jsonl(args.teacher)),
        ("sft1_demo_replay", demo_selected),
    ]
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = collections.Counter()
    lane_counts = collections.Counter()
    transition_counts = collections.Counter()
    for lane, rows in lanes:
        for row in rows:
            key = content_key(row)
            if key in seen:
                duplicates[lane] += 1
                continue
            seen.add(key)
            if lane == "student_success" and (row.get("metadata") or {}).get("feedback_recovery"):
                transition = "feedback_recovery"
            elif lane == "decision_correction":
                transition = "decision_correction"
            elif lane == "teacher_fallback":
                transition = "teacher_fallback"
            elif lane == "sft1_demo_replay":
                transition = "teacher_demo_replay"
            else:
                transition = "student_success"
            accepted.append(tagged(row, lane, transition))
            lane_counts[lane] += 1
            transition_counts[transition] += 1

    accepted.sort(key=lambda row: (
        row["metadata"]["sft2_lane"], row["metadata"].get("record_id", "")
    ))
    write_jsonl(args.out, accepted)
    manifest = {
        "protocol": "draft/bird_sft2_onpolicy_data_protocol.md",
        "seed": args.seed,
        "output": str(args.out),
        "records": len(accepted),
        "lane_counts": dict(lane_counts),
        "transition_counts": dict(transition_counts),
        "deduplicated_lower_priority_records": dict(duplicates),
        "priority": [lane for lane, _ in lanes],
        "demo_replay": {
            "requested": args.demo_count,
            "selected": len(demo_selected),
            "difficulty_quotas": quotas,
        },
        "loss_policy": "last_assistant_turn_only",
        "required_llamafactory_flag": "mask_history: true",
        "note": (
            "Canonical records are not duplicated to force mixture ratios. Use reported lane tags "
            "for sampler weighting and ablations."
        ),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

