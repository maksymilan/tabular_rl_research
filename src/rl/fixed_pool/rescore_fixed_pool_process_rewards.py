#!/usr/bin/env python3
"""Reallocate process credit from an immutable harness-replayed feature view."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src" / "rl")]

from frameworks.trl.fixed_rollout_pool import (  # noqa: E402
    deserialize_episode,
    load_rows,
    write_rows_atomic,
)
from frameworks.trl.transition_batch import build_transition_updates  # noqa: E402
from process_credit import (  # noqa: E402
    ProcessRewardConfig,
    StepFeature,
    allocate_process_rewards,
)


FEATURE_FIELDS = {item.name for item in fields(StepFeature)}


def load_process_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def immutable_identity(row: dict[str, Any]) -> dict[str, Any]:
    sample = row["sample"]
    audit = sample["audit_record"]
    return {
        "sequence": row["sequence"],
        "environment": row["environment"],
        "policy_turns": row["policy_turns"],
        "trajectory_id": audit["trajectory_id"],
        "sample_index": audit["sample_index"],
        "correct": sample["correct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--feature-source", required=True, type=Path)
    parser.add_argument("--process-reward-config", required=True, type=Path)
    parser.add_argument("--admission-mode", choices=("dense-outcome",), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    outputs = {
        "validated": args.output_dir / "validated_trajectories.jsonl",
        "features": args.output_dir / "process_features.jsonl",
        "counterfactual": args.output_dir / "counterfactual_results.jsonl",
        "transitions": args.output_dir / "transitions.jsonl",
        "summary": args.output_dir / "validation_summary.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite an existing rescored pool artifact")

    config = load_process_config(args.process_reward_config)
    raw_rows = load_rows(args.trajectories)
    source_rows = load_rows(args.feature_source)
    if len(raw_rows) != len(source_rows):
        raise ValueError("raw trajectories and feature-source rows must align")

    validated_rows = []
    feature_rows = []
    for raw, source in zip(raw_rows, source_rows, strict=True):
        if immutable_identity(raw) != immutable_identity(source):
            raise ValueError(
                f"feature-source identity mismatch at sequence {raw.get('sequence')}"
            )
        source_reward = source["sample"]["audit_record"].get("process_reward") or {}
        source_steps = source_reward.get("steps") or []
        if len(source_steps) != len(raw["policy_turns"]):
            raise ValueError(
                f"feature-source step count mismatch at sequence {raw['sequence']}"
            )
        features = []
        for step in source_steps:
            payload = {
                key: value
                for key, value in (step.get("features") or {}).items()
                if key in FEATURE_FIELDS
            }
            payload.setdefault("action_index", step["action_index"])
            payload.setdefault("step_id", step["step_id"])
            payload.setdefault("tool", step.get("tool"))
            payload.setdefault("legal_success", bool(payload.get("legal_success")))
            features.append(StepFeature(**payload))

        reward = allocate_process_rewards(
            str(raw["sample"]["audit_record"]["trajectory_id"]),
            features,
            correct=bool(raw["sample"]["correct"]),
            config=config,
            diagnostics=source_reward.get("diagnostics") or {},
        )
        updated = deepcopy(raw)
        updated_audit = updated["sample"]["audit_record"]
        updated_audit["process_reward"] = reward.to_dict()
        updated_audit["process_admission_policy"] = args.admission_mode
        updated["sample"].update(
            {
                "reward": float(reward.total_reward),
                "step_rewards": [float(step.reward) for step in reward.steps],
                "process_update": bool(reward.process_update),
            }
        )
        validated_rows.append(updated)
        for step_index, step in enumerate(reward.steps):
            feature_rows.append(
                {
                    "sequence": raw["sequence"],
                    "trajectory_id": updated_audit["trajectory_id"],
                    "task_id": raw["environment"]["task_id"],
                    "example_index": raw["environment"]["example_index"],
                    "step_index": step_index,
                    **step.to_dict(),
                }
            )

    episodes_by_example: dict[int, list[Any]] = defaultdict(list)
    for row in validated_rows:
        episodes_by_example[int(row["environment"]["example_index"])].append(
            deserialize_episode(row)
        )
    transition_rows = []
    for example_index in sorted(episodes_by_example):
        episodes = sorted(
            episodes_by_example[example_index],
            key=lambda episode: int(episode.sample.audit_record["sample_index"]),
        )
        for update in build_transition_updates(episodes, reward_mode="process"):
            transition_rows.append(
                {
                    "example_index": update.example_index,
                    "trajectory_id": update.trajectory_id,
                    "turn_index": update.turn_index,
                    "prompt_ids": list(update.prompt_ids),
                    "response_ids": list(update.response_ids),
                    "sampling_logprobs": list(update.sampling_logprobs),
                    "advantage": update.advantage,
                    "trajectory_correct": update.trajectory_correct,
                    "legal_success": update.legal_success,
                    "local_penalty": update.local_penalty,
                }
            )

    write_rows_atomic(outputs["validated"], validated_rows)
    write_rows_atomic(outputs["features"], feature_rows)
    write_rows_atomic(outputs["counterfactual"], [])
    write_rows_atomic(outputs["transitions"], transition_rows)
    summary = {
        "schema_version": "fixed-pool-process-validation-v1",
        "admission_mode": args.admission_mode,
        "trajectories": len(validated_rows),
        "correct_trajectories": sum(row["sample"]["correct"] for row in validated_rows),
        "process_update_trajectories": sum(
            row["sample"]["process_update"] for row in validated_rows
        ),
        "process_steps": len(feature_rows),
        "transitions": len(transition_rows),
        "counterfactual_correct_trajectories": 0,
        "counterfactual_passed": 0,
        "counterfactual_screened": 0,
        "excluded": {},
        "feature_source": str(args.feature_source.resolve()),
        "feature_replay_reused": True,
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
