#!/usr/bin/env python3
"""Score rollout artifacts with the pilot RL reward and write an auditable report."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reward import iter_episode_records, score_episode  # noqa: E402


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                yield line_no, json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rollout/pass@k all.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_episodes.jsonl"
    summary_path = output_dir / "summary.json"

    rewards = []
    component_sums = Counter()
    failure_types = Counter()
    diagnostics = Counter()
    total = 0

    with scored_path.open("w", encoding="utf-8") as sink:
        for line_no, record in load_jsonl(input_path):
            for sample_id, episode in iter_episode_records(record):
                breakdown = score_episode(episode)
                total += 1
                rewards.append(breakdown.reward)
                component_sums.update(breakdown.components)
                failure_types.update([breakdown.diagnostics.get("failure_type") or "none"])
                diagnostics.update({
                    "correct": int(breakdown.diagnostics["correct"]),
                    "legal": int(breakdown.diagnostics["legal"]),
                    "tool_errors": breakdown.diagnostics["tool_errors"],
                    "repeat_calls": breakdown.diagnostics["repeat_calls"],
                })
                sink.write(json.dumps({
                    "sample_id": sample_id,
                    "source_line": line_no,
                    "example_index": episode.get("example_index"),
                    "db_id": episode.get("db_id"),
                    "question": episode.get("question"),
                    "reward": breakdown.reward,
                    "components": breakdown.components,
                    "diagnostics": breakdown.diagnostics,
                }, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "scored_episodes": total,
        "reward": {
            "mean": statistics.mean(rewards) if rewards else 0.0,
            "median": statistics.median(rewards) if rewards else 0.0,
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "component_sums": dict(sorted(component_sums.items())),
        "failure_types": dict(sorted(failure_types.items())),
        "diagnostics": dict(sorted(diagnostics.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
