#!/usr/bin/env python3
"""Export verified BIRD SFT-1 teacher episodes as one-tool-decision ShareGPT records."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bird_sft1_teacher import first_future_reference, replay_success_trajectory  # noqa: E402
from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    assistant_message,
    model_context_messages,
    protocol_hash,
    task_context_message,
)

DEFAULT_SUCCESS = ROOT / "data" / "trajectories" / "bird_sft1_teacher_pilot30_success.jsonl"
DEFAULT_OUT = ROOT / "data" / "sft" / "bird_sft1_teacher_pilot30_step_train.jsonl"
DEFAULT_INDEX = ROOT / "data" / "sft" / "bird_sft1_teacher_pilot30_step_index.jsonl"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def state_hash(state: dict) -> str:
    return hashlib.sha256(compact(state).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def numeric_step_id(step_id: str) -> int:
    try:
        return int(step_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid step id {step_id!r}") from exc


def validate_step_input(trajectory: dict, step: dict, human: str) -> None:
    current = numeric_step_id(step["step_id"])
    future = first_future_reference(step["tool_call"]["arguments"], current)
    if future:
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: future target reference {future}")
    # Every state id in the user context must precede this tool decision.  The target itself may
    # reference only prior scalar values and is checked above.
    for token in __import__("re").findall(r"\bstep_(\d+)\b", human):
        if int(token) >= current:
            raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: future id in input step_{token}")
    if trajectory["source"]["gold_sql"] in human:
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: gold SQL leaked into input")
    output_text = compact(step.get("tool_output"))
    if output_text and output_text in human:
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: current tool output leaked into input")


def build(success_path: Path, out_path: Path, index_path: Path) -> dict:
    trajectories = read_jsonl(success_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tool_hist = collections.Counter()
    difficulty_hist = collections.Counter()
    difficulty_steps = collections.Counter()
    recovery_steps = collections.Counter()
    episode_steps: dict[str, int] = {}
    record_count = 0

    with out_path.open("w", encoding="utf-8") as out, index_path.open("w", encoding="utf-8") as index:
        for trajectory in trajectories:
            if trajectory.get("label_status") != "verified":
                raise ValueError(f"{trajectory.get('trajectory_id')}: not verified")
            replay_ok, replay_error = replay_success_trajectory(trajectory)
            if not replay_ok:
                raise ValueError(f"{trajectory['trajectory_id']}: {replay_error}")
            source = trajectory["source"]
            catalog = trajectory["initial_state"]["dataset_overview"]
            difficulty = trajectory.get("difficulty") or "unknown"
            steps = trajectory["steps"]
            episode_steps[trajectory["trajectory_id"]] = len(steps)
            difficulty_hist[difficulty] += 1
            for source_step_index, step in enumerate(steps):
                state_before = step["environment_state_before"]
                last_error = step.get("last_tool_error_before")
                human = task_context_message(
                    catalog,
                    trajectory["question"],
                    state_before,
                    last_error,
                    source.get("external_knowledge"),
                )
                # Assert that the exact online renderer has identical system/user semantics.
                online = model_context_messages(
                    SYSTEM_PROMPT, catalog, trajectory["question"], state_before, last_error,
                    source.get("external_knowledge"),
                )
                if online != [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": human}]:
                    raise AssertionError("step SFT renderer drifted from online model_context_messages")
                validate_step_input(trajectory, step, human)
                tool_call = step["tool_call"]
                record_id = f"{trajectory['trajectory_id']}_step_{source_step_index:03d}"
                record = {
                    "system": SYSTEM_PROMPT,
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": assistant_message(
                            step["think"], tool_call["tool"], tool_call["arguments"],
                        )},
                    ],
                }
                if len(record["conversations"]) != 2 or any(not msg["value"] for msg in record["conversations"]):
                    raise AssertionError(f"{record_id}: invalid one-pair ShareGPT record")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                index.write(json.dumps({
                    "sft_record_id": record_id,
                    "source_episode_id": trajectory["trajectory_id"],
                    "source_difficulty": difficulty,
                    "source_step_index": source_step_index,
                    "source_step_id": step["step_id"],
                    "tool_name": tool_call["tool"],
                    "feedback_recovery": bool(step.get("feedback_recovery")),
                    "recovered_from_error_type": step.get("recovered_from_error_type"),
                    "state_before_hash": state_hash(state_before),
                    "source_trajectory_path": str(success_path),
                }, ensure_ascii=False) + "\n")
                record_count += 1
                tool_hist[tool_call["tool"]] += 1
                difficulty_steps[difficulty] += 1
                if step.get("feedback_recovery"):
                    recovery_steps[step.get("recovered_from_error_type") or "unknown"] += 1

    expected = sum(episode_steps.values())
    if record_count != expected:
        raise AssertionError(f"step record count {record_count} != source assistant steps {expected}")
    manifest = {
        "schema": "bird-sft1-step-v1",
        "source_success_trajectories": str(success_path),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "accepted_episodes": len(trajectories),
        "step_records": record_count,
        "episode_step_counts": episode_steps,
        "difficulty_episode_counts": dict(difficulty_hist),
        "difficulty_step_counts": dict(difficulty_steps),
        "tool_step_counts": dict(tool_hist.most_common()),
        "feedback_recovery_step_counts": dict(recovery_steps),
        "recovered_success_episodes": sum(
            trajectory.get("rollout_generation", {}).get("outcome") == "recovered_success"
            for trajectory in trajectories
        ),
        "validation": {
            "all_success_episodes_replayed": True,
            "one_human_one_gpt_per_record": True,
            "shared_online_renderer": True,
            "no_current_output_or_future_step_ids": True,
            "error_actions_excluded_from_targets": True,
        },
        "output": str(out_path),
        "index_output": str(index_path),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, default=DEFAULT_SUCCESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    if not args.success.is_file():
        parser.error(f"success trajectory file not found: {args.success}")
    print(json.dumps(build(args.success, args.out, args.index), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
