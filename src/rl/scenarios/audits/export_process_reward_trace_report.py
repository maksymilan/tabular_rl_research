#!/usr/bin/env python3
"""Export gold-free, action-aligned Process RL reward traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rl.objectives.process_credit import ProcessRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def compact(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def classify(row: dict[str, Any]) -> str:
    reward = row["process_reward"]
    if row.get("correct"):
        return "correct_penalty" if reward["capped_penalty"] else "correct_clean"
    return "wrong_negative" if reward["total_reward"] < 0 else "wrong_zero"


def action_records(
    row: dict[str, Any],
    config: ProcessRewardConfig,
    source: str,
) -> list[dict[str, Any]]:
    reward = row["process_reward"]
    steps = reward["steps"]
    turns = row["turns"]
    if len(steps) != len(turns):
        raise ValueError(
            f"{row['trajectory_id']}: reward/action length mismatch "
            f"{len(steps)} != {len(turns)}"
        )
    penalty_scale = (
        float(reward["capped_penalty"]) / float(reward["raw_penalty_mass"])
        if reward["raw_penalty_mass"]
        else 0.0
    )
    records = []
    for step, turn in zip(steps, turns, strict=True):
        features = step["features"]
        parsed = turn.get("parsed") or {}
        positive_raw = {
            "terminal_correct": (
                config.w_terminal_correct * float(bool(features["is_terminal"]))
            ),
            "data_dependency": config.w_back_slice * float(features["back_slice"]),
            "search_reduction": (
                config.w_search_reduction * float(features["search_reduction"])
            ),
        }
        positive_applied = {
            key: (
                value / float(reward["positive_mass"])
                if row.get("correct") and reward["positive_mass"]
                else 0.0
            )
            for key, value in positive_raw.items()
        }
        penalty_raw = {
            "tool_error": config.lambda_tool_error * float(features["tool_error"]),
            "adjacent_repeat": (
                config.lambda_adjacent_repeat
                * float(bool(features["adjacent_repeat"]))
            ),
            "legal_no_state_change": (
                config.lambda_legal_no_state_change
                * float(features["legal_no_state_change"])
            ),
        }
        penalty_applied = {
            key: value * penalty_scale for key, value in penalty_raw.items()
        }
        reconstructed = sum(positive_applied.values()) - sum(
            penalty_applied.values()
        )
        if abs(reconstructed - float(step["reward"])) > 1e-8:
            raise ValueError(
                f"{row['trajectory_id']} action {step['action_index']}: "
                f"component reconstruction {reconstructed} != {step['reward']}"
            )
        records.append(
            {
                "source": source,
                "training_step": row["training_step"],
                "sample_index": row["sample_index"],
                "trajectory_id": row["trajectory_id"],
                "example_index": row["example_index"],
                "db_id": row["db_id"],
                "question": row["question"],
                "trajectory_class": classify(row),
                "trajectory_correct": bool(row.get("correct")),
                "trajectory_failure_type": row.get("failure_type"),
                "trajectory_total_reward": reward["total_reward"],
                "trajectory_positive_mass": reward["positive_mass"],
                "trajectory_raw_penalty_mass": reward["raw_penalty_mass"],
                "trajectory_capped_penalty": reward["capped_penalty"],
                "action_index": step["action_index"],
                "step_id": step["step_id"],
                "tool": parsed.get("tool", step["tool"]),
                "arguments": parsed.get("arguments"),
                "think": parsed.get("think"),
                "error_type": features["error_type"],
                "state_changed": features["state_changed"],
                "positive_raw": positive_raw,
                "positive_applied": positive_applied,
                "penalty_raw": penalty_raw,
                "penalty_applied": penalty_applied,
                "reward": step["reward"],
            }
        )
    return records


def component_text(record: dict[str, Any], key: str) -> str:
    values = record[key]
    active = [f"{name}={value:.4f}" for name, value in values.items() if value]
    return ", ".join(active) if active else "—"


def render_trajectory(
    row: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    include_reasoning: bool,
) -> list[str]:
    reward = row["process_reward"]
    lines = [
        f"### `{row['trajectory_id']}`",
        "",
        (
            f"- source: `{actions[0]['source']}`; training step/sample: "
            f"`{row['training_step']}/{row['sample_index']}`; example: "
            f"`{row['example_index']}`; db: `{row['db_id']}`"
        ),
        f"- question: {row['question']}",
        (
            f"- result: class=`{classify(row)}`, correct=`{bool(row.get('correct'))}`, "
            f"failure=`{row.get('failure_type')}`, total reward=`{reward['total_reward']}` "
            f"(positive base=`{float(bool(row.get('correct'))):.1f}`, "
            f"penalty=`{reward['capped_penalty']}`)"
        ),
        "",
        "| # | tool | arguments | positive credit | penalty | action reward | state/error |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for action in actions:
        state = (
            f"changed={action['state_changed']}; "
            f"error={action['error_type'] or 'none'}"
        )
        lines.append(
            "| {index} | `{tool}` | `{arguments}` | {positive} | {penalty} | "
            "**{reward:.4f}** | {state} |".format(
                index=action["action_index"],
                tool=action["tool"],
                arguments=compact(action["arguments"]).replace("|", "\\|"),
                positive=component_text(action, "positive_applied"),
                penalty=component_text(action, "penalty_applied"),
                reward=float(action["reward"]),
                state=state,
            )
        )
        if include_reasoning and action.get("think"):
            reasoning = " ".join(str(action["think"]).split())
            lines.append(
                f"|  | reasoning | {reasoning[:240].replace('|', '\\|')}"
                f"{'…' if len(reasoning) > 240 else ''} |  |  |  |  |"
            )
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    config = load_config(args.reward_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    actions_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    rows_by_trajectory: dict[str, dict[str, Any]] = {}
    for rollout_path in args.rollouts:
        source = rollout_path.stem
        with rollout_path.open(encoding="utf-8") as source_file:
            for line in source_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.pop("gold_sql", None)
                actions = action_records(row, config, source)
                trajectory_key = f"{source}:{row['trajectory_id']}"
                all_rows.extend(actions)
                actions_by_trajectory[trajectory_key] = actions
                rows_by_trajectory[trajectory_key] = row

    action_path = args.output_dir / "actions.jsonl"
    with action_path.open("w", encoding="utf-8") as sink:
        for record in all_rows:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    classes = Counter(
        classify(row) for row in rows_by_trajectory.values()
    )
    markdown = [
        "# Process RL action-level reward traces",
        "",
        "Gold SQL and reference result values are intentionally omitted.",
        "",
        f"- trajectories: {len(rows_by_trajectory)}",
        f"- actions: {len(all_rows)}",
        f"- classes: `{json.dumps(dict(classes), ensure_ascii=False)}`",
        "",
        "A successful trajectory receives one normalized positive unit split over terminal "
        "correctness, executed back-slice dependencies, and search reduction. Local penalties "
        "are then subtracted per action.",
        "",
    ]
    for trajectory_id, row in rows_by_trajectory.items():
        markdown.extend(
            render_trajectory(
                row,
                actions_by_trajectory[trajectory_id],
                include_reasoning=False,
            )
        )
    (args.output_dir / "all_trajectories.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    representatives: dict[str, str] = {}
    for trajectory_id, row in rows_by_trajectory.items():
        representatives.setdefault(classify(row), trajectory_id)
    representative_markdown = [
        "# Representative Process RL reward traces",
        "",
        "One real trajectory from each currently observed reward class. Gold SQL is omitted.",
        "",
    ]
    for category in (
        "correct_clean",
        "correct_penalty",
        "wrong_zero",
        "wrong_negative",
    ):
        trajectory_id = representatives.get(category)
        if trajectory_id is None:
            continue
        representative_markdown.extend([f"## {category}", ""])
        representative_markdown.extend(
            render_trajectory(
                rows_by_trajectory[trajectory_id],
                actions_by_trajectory[trajectory_id],
                include_reasoning=True,
            )
        )
    (args.output_dir / "representative_trajectories.md").write_text(
        "\n".join(representative_markdown) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "process-reward-action-trace-v1",
        "rollout_sources": [str(path) for path in args.rollouts],
        "reward_config": str(args.reward_config),
        "trajectory_count": len(rows_by_trajectory),
        "action_count": len(all_rows),
        "class_counts": dict(classes),
        "gold_sql_included": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
