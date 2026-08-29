from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from src.rl.diagnostics.audit_vanilla_grpo_boundary_validation import (
    EXPECTED_INITIAL_ADAPTER_SHA256,
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_RUNTIME_TREE_SHA256,
    EXPECTED_STUDENT_PROMPT_SHA256,
    VALIDATION_SEED,
    audit_boundary_validation,
    main,
)


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _tasks() -> list[dict]:
    return [
        {
            "example_id": f"bird_train_{index:05d}",
            "instance_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 8}",
            "db_path": f"/db/db_{index % 8}.sqlite",
            "question": f"question {index}",
            "gold_sql": f"select {index}",
            "external_knowledge": None,
        }
        for index in range(32)
    ]


def _trajectories(tasks: list[dict], *, mixed_groups: int = 20) -> list[dict]:
    rows = []
    for task_position, task in enumerate(tasks):
        task_id = task["example_id"]
        for sample_index in range(8):
            correct = sample_index < 4 if task_position < mixed_groups else True
            sequence = task_position * 8 + sample_index
            rows.append(
                {
                    "schema_version": "table-agent-fixed-policy-episode-v1",
                    "sequence": sequence,
                    "environment": {
                        "dataset_split": "train",
                        "example_index": task["example_index"],
                        "task_id": task_id,
                        "db_id": task["db_id"],
                        "db_path": task["db_path"],
                        "question": task["question"],
                        "gold_sql": task["gold_sql"],
                        "external_knowledge": None,
                    },
                    "sample": {
                        "reward": float(correct),
                        "correct": correct,
                        "failure_type": None,
                        "step_rewards": None,
                        "process_update": True,
                        "audit_record": {
                            "trajectory_id": f"rl_{task['example_index']}_sample_{sample_index}",
                            "example_index": task["example_index"],
                            "sample_index": sample_index,
                            "protocol_version": EXPECTED_PROTOCOL_VERSION,
                            "protocol_hash": EXPECTED_PROTOCOL_HASH,
                            "failure_type": None,
                            "turns": [],
                            "error_events": [],
                            "result_reward": {
                                "profile": "binary",
                                "correct": correct,
                                "executable_terminal": correct,
                                "value": float(correct),
                            },
                        },
                    },
                    "policy_turns": [
                        {
                            "prompt_ids": [1, 2],
                            "response_ids": [3, 4],
                            "sampling_logprobs": [-0.1, -0.2],
                        }
                    ],
                }
            )
    return rows


def _selection(tasks: list[dict], tasks_sha: str) -> dict:
    ids = [row["example_id"] for row in tasks]
    return {
        "schema_version": "policy-boundary-grpo-cohort-v1",
        "status": "frozen_boundary_training_cohort",
        "selection": {
            "seed": "qwen3-v26-boundary332-v1-20260812",
            "acceptance_gates": {"all": True},
        },
        "contract": {
            "boundary_records": 332,
            "train_records": 300,
            "validation_records": 32,
            "formal_training": {
                "optimizer_updates": 20,
                "prompts_per_update": 30,
                "group_size": 8,
                "train_passes": 2,
                "prompt_appearances": 600,
                "fresh_online_trajectories": 4800,
                "sampler": "trl-0.29-repeat-sampler-v1",
                "shuffle_dataset": True,
                "data_seed": 20260812,
                "task_order": "two deterministic data-seed shuffled passes",
                "per_pass_coverage": "each train300 identity exactly once",
                "reward_mode": "result-only",
                "result_reward_profile": "binary",
            },
            "validation": {
                "policy": "fresh initial-SFT1",
                "records": 32,
                "group_size": 8,
                "seed": 20260813,
                "screen_seed_must_differ": 20260812,
                "generation_seed_scheme": "sha256-task-sample-turn-v1",
                "gate": "same >=20/32 mixed probe gate",
                "runtime_contamination_allowed": False,
                "screen_trajectories_reused": False,
            },
            "primary_checkpoint": "final-step20-only",
        },
        "outputs": {
            "validation": {
                "path": "/selection/validation32.jsonl",
                "sha256": tasks_sha,
                "records": 32,
            }
        },
        "task_ids": {"validation": ids},
    }


def _generation(tasks_sha: str, trajectory_sha: str, correct: int) -> dict:
    return {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_path": "/selection/validation32.jsonl",
        "tasks_sha256": tasks_sha,
        "tasks": 32,
        "group_size": 8,
        "trajectories": 256,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": VALIDATION_SEED,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "trajectories_sha256": trajectory_sha,
        "correct_trajectories": correct,
    }


def _case(mixed_groups: int = 20):
    tasks = _tasks()
    trajectories = _trajectories(tasks, mixed_groups=mixed_groups)
    task_bytes = _jsonl(tasks)
    trajectory_bytes = _jsonl(trajectories)
    selection = _selection(tasks, hashlib.sha256(task_bytes).hexdigest())
    selection_bytes = (json.dumps(selection) + "\n").encode()
    generation = _generation(
        hashlib.sha256(task_bytes).hexdigest(),
        hashlib.sha256(trajectory_bytes).hexdigest(),
        sum(row["sample"]["correct"] for row in trajectories),
    )
    return tasks, trajectories, selection, selection_bytes, generation, task_bytes, trajectory_bytes


def _audit(case) -> dict:
    tasks, trajectories, selection, selection_bytes, generation, task_bytes, trajectory_bytes = case
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    return audit_boundary_validation(
        selection_manifest=selection,
        generation_manifest=generation,
        tasks=tasks,
        trajectories=trajectories,
        expected_selection_manifest_sha256=selection_sha,
        selection_manifest_sha256=selection_sha,
        generation_manifest_sha256="a" * 64,
        tasks_sha256=hashlib.sha256(task_bytes).hexdigest(),
        trajectories_sha256=hashlib.sha256(trajectory_bytes).hexdigest(),
    )


def test_fresh_boundary_validation_passes_only_with_independent_seed_and_clean_k8() -> None:
    result = _audit(_case())
    assert result["status"] == {"passes": True, "validation_admitted": True}
    assert all(result["checks"].values())
    assert result["observed"]["eligible_trajectories"] == 256
    assert result["observed"]["mixed_outcome_groups"] == 20


def test_old_screen_seed_cannot_be_used_as_boundary_validation() -> None:
    case = list(_case())
    case[4] = copy.deepcopy(case[4])
    case[4]["seed"] = 20260812
    result = _audit(tuple(case))
    assert not result["checks"]["independent_validation_seed"]
    assert not result["status"]["passes"]


def test_wrong_selection_digest_or_task_binding_fails_closed() -> None:
    case = _case()
    result = audit_boundary_validation(
        selection_manifest=case[2],
        generation_manifest=case[4],
        tasks=case[0],
        trajectories=case[1],
        expected_selection_manifest_sha256="0" * 64,
        selection_manifest_sha256=hashlib.sha256(case[3]).hexdigest(),
        generation_manifest_sha256="a" * 64,
        tasks_sha256=hashlib.sha256(case[5]).hexdigest(),
        trajectories_sha256=hashlib.sha256(case[6]).hexdigest(),
    )
    assert not result["checks"]["explicit_selection_manifest_digest"]

    case = list(_case())
    case[1] = copy.deepcopy(case[1])
    case[1][0]["environment"]["question"] = "wrong"
    # Keep the actual bytes/hash aligned: semantic task binding must still fail.
    case[6] = _jsonl(case[1])
    case[4] = copy.deepcopy(case[4])
    case[4]["trajectories_sha256"] = hashlib.sha256(case[6]).hexdigest()
    result = _audit(tuple(case))
    assert not result["checks"]["exact_trajectory_task_binding"]


def test_any_runtime_exclusion_or_fewer_than_20_mixed_fails() -> None:
    case = list(_case())
    case[1] = copy.deepcopy(case[1])
    case[1][0]["sample"]["process_update"] = False
    case[1][0]["sample"]["failure_type"] = "generation_length"
    case[1][0]["sample"]["audit_record"]["failure_type"] = "generation_length"
    case[1][0]["sample"]["audit_record"]["generation_truncation"] = {"kind": "length"}
    case[6] = _jsonl(case[1])
    case[4] = copy.deepcopy(case[4])
    case[4]["trajectories_sha256"] = hashlib.sha256(case[6]).hexdigest()
    result = _audit(tuple(case))
    assert not result["checks"]["no_runtime_contamination"]

    result = _audit(_case(mixed_groups=19))
    assert not result["checks"]["at_least_20_mixed_outcome_groups"]


def test_cli_failure_revokes_a_stale_admission(tmp_path: Path) -> None:
    tasks, trajectories, selection, selection_bytes, generation, task_bytes, trajectory_bytes = _case(19)
    selection_path = tmp_path / "boundary_cohort_manifest.json"
    generation_path = tmp_path / "manifest.pending.json"
    tasks_path = tmp_path / "validation32.jsonl"
    trajectories_path = tmp_path / "trajectories.jsonl"
    output = tmp_path / "audit.json"
    selection["outputs"]["validation"]["path"] = str(tasks_path.resolve())
    selection_bytes = (json.dumps(selection) + "\n").encode()
    generation["tasks_path"] = str(tasks_path.resolve())
    selection_path.write_bytes(selection_bytes)
    generation_path.write_text(json.dumps(generation))
    tasks_path.write_bytes(task_bytes)
    trajectories_path.write_bytes(trajectory_bytes)
    output.write_text('{"status":{"passes":true}}')
    code = main(
        [
            "--selection-manifest", str(selection_path),
            "--expected-selection-manifest-sha256", hashlib.sha256(selection_bytes).hexdigest(),
            "--manifest", str(generation_path),
            "--trajectories", str(trajectories_path),
            "--tasks", str(tasks_path),
            "--output", str(output),
            "--overwrite",
        ]
    )
    assert code == 1
    assert not output.exists()
