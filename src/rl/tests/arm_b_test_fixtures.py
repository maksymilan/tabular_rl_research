"""Real builder fixtures for the independent Arm-B staged artifact contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.data.vanilla_grpo_arm_b import (
    GENERATION_SEED_SCHEME,
    INITIAL_ADAPTER_SHA256,
    PENDING_MANIFEST_SCHEMA,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    RUNTIME_TREE_SHA256,
    STUDENT_PROMPT_SHA256,
    TRAJECTORY_SCHEMA,
    jsonl_bytes,
    sha256_bytes,
    task_id,
)


def task(index: int) -> dict[str, Any]:
    identifier = f"bird_train_{index:05d}"
    return {
        "dataset": "bird-sql",
        "split": "train",
        "example_index": index,
        "example_id": identifier,
        "instance_id": identifier,
        "db_id": f"db_{index % 5}",
        "question": f"question {index} " + ("x" * (index % 11)),
        "db_path": f"/remote/db_{index % 5}/db_{index % 5}.sqlite",
        "gold_sql": "SELECT 1",
        "query": "SELECT 1",
        "gold_exec_results": [],
        "external_knowledge": "hint" if index % 3 == 0 else None,
        "metadata": {"tool_round_trip": "verified"},
    }


def write_json(path: Path, value: Any) -> bytes:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> bytes:
    payload = jsonl_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def generation_artifacts(
    root: Path,
    tasks: Sequence[dict[str, Any]],
    *,
    group_size: int,
    seed: int,
    correct_counts: Sequence[int],
) -> tuple[Path, Path, bytes, bytes]:
    if len(correct_counts) != len(tasks):
        raise ValueError("one correct count is required per task")
    tasks_path = root / "tasks.jsonl"
    task_bytes = write_jsonl(tasks_path, tasks)
    trajectories: list[dict[str, Any]] = []
    correct_total = 0
    for task_position, (row, correct_count) in enumerate(
        zip(tasks, correct_counts, strict=True)
    ):
        for sample_index in range(group_size):
            correct = sample_index < correct_count
            correct_total += int(correct)
            result_reward = {
                "profile": "binary",
                "correct": correct,
                "value": float(correct),
            }
            record = {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": PROTOCOL_HASH,
                "example_index": row["example_index"],
                "sample_index": sample_index,
                "trajectory_id": (
                    f"rl_{row['example_index']}_sample_{sample_index}"
                ),
                "correct": correct,
                "failure_type": None,
                "result_reward": result_reward,
                "policy_global_step": 0,
                "optimizer_step": 0,
            }
            trajectories.append(
                {
                    "schema_version": TRAJECTORY_SCHEMA,
                    "sequence": task_position * group_size + sample_index,
                    "environment": {
                        "dataset_split": "train",
                        "example_index": row["example_index"],
                        "task_id": task_id(row),
                        "db_id": row["db_id"],
                        "db_path": row["db_path"],
                        "question": row["question"],
                        "gold_sql": row["gold_sql"],
                        "external_knowledge": row["external_knowledge"],
                    },
                    "sample": {
                        "correct": correct,
                        "reward": float(correct),
                        "step_rewards": None,
                        "process_update": True,
                        "failure_type": None,
                        "audit_record": record,
                    },
                    "policy_turns": [
                        {
                            "prompt_ids": [1, 2],
                            "response_ids": [3],
                            "sampling_logprobs": [-0.5],
                        }
                    ],
                }
            )
    trajectory_path = root / "trajectories.jsonl"
    trajectory_bytes = write_jsonl(trajectory_path, trajectories)
    manifest = {
        "schema_version": PENDING_MANIFEST_SCHEMA,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "student_prompt_sha256": STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": RUNTIME_TREE_SHA256,
        "adapter_sha256": INITIAL_ADAPTER_SHA256,
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": sha256_bytes(task_bytes),
        "tasks": len(tasks),
        "group_size": group_size,
        "trajectories": len(trajectories),
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": seed,
        "generation_seed_scheme": GENERATION_SEED_SCHEME,
        "denotation_comparison": "bird-set",
        "trajectories_sha256": sha256_bytes(trajectory_bytes),
        "correct_trajectories": correct_total,
    }
    manifest_path = root / "manifest.pending.json"
    write_json(manifest_path, manifest)
    return manifest_path, trajectory_path, task_bytes, trajectory_bytes
