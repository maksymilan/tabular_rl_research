#!/usr/bin/env python3
"""Export replay/grounding gate categories without discarding canonical trajectories."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reasons(score: dict[str, Any]) -> list[str]:
    diagnostics = score.get("diagnostics") or {}
    result = []
    if not score.get("correct"):
        result.append("replay_incorrect")
    if not diagnostics.get("final_value_grounding_complete"):
        result.append("unsupported_final_value")
    if not diagnostics.get("action_literal_grounding_complete"):
        result.append("unsupported_action_literal")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    trajectory_path = args.trajectories.resolve()
    scored_path = args.scored.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = read_jsonl(trajectory_path)
    by_id = {row["trajectory_id"]: row for row in trajectories}
    scores = read_jsonl(scored_path)
    score_by_id = {row["trajectory_id"]: row for row in scores}
    if set(by_id) != set(score_by_id):
        raise ValueError("trajectory ids and scored ids differ")

    categories: dict[str, list[dict[str, Any]]] = {
        "eligible": [],
        "rejected_any": [],
        "replay_incorrect": [],
        "unsupported_final_value": [],
        "unsupported_action_literal": [],
        "multiple_issues": [],
    }
    index: list[dict[str, Any]] = []
    difficulty = collections.Counter()
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        score = score_by_id[trajectory_id]
        issue_list = reasons(score)
        eligible = not issue_list
        categories["eligible" if eligible else "rejected_any"].append(trajectory)
        for issue in issue_list:
            categories[issue].append(trajectory)
        if len(issue_list) > 1:
            categories["multiple_issues"].append(trajectory)
        if eligible:
            difficulty[trajectory.get("difficulty") or "unknown"] += 1
        diagnostics = score.get("diagnostics") or {}
        index.append({
            "trajectory_id": trajectory_id,
            "eligible": eligible,
            "rejection_reasons": issue_list,
            "difficulty": trajectory.get("difficulty") or "unknown",
            "unsupported_final_values": diagnostics.get("unsupported_final_values") or [],
            "unsupported_action_literals": diagnostics.get("unsupported_action_literals") or [],
            "grounding_method": score.get("grounding_method"),
        })

    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in categories.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        outputs[name] = {"path": str(path), "rows": len(rows), "sha256": sha256(path)}
    index_path = out_dir / "gate_index.jsonl"
    write_jsonl(index_path, index)
    manifest = {
        "trajectories": {"path": str(trajectory_path), "sha256": sha256(trajectory_path)},
        "scored": {"path": str(scored_path), "sha256": sha256(scored_path)},
        "gate": {
            "replay_correct": True,
            "final_value_grounding_complete": True,
            "action_literal_grounding_complete": True,
        },
        "outputs": outputs,
        "eligible_by_difficulty": dict(sorted(difficulty.items())),
        "index": {"path": str(index_path), "rows": len(index), "sha256": sha256(index_path)},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
