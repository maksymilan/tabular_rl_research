from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rl.scenarios.diagnostics.audit_vanilla_grpo_boundary_screen import (
    EXPECTED_INITIAL_ADAPTER_SHA256,
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_RUNTIME_TREE_SHA256,
    EXPECTED_STUDENT_PROMPT_SHA256,
    audit,
    main,
)


def _tasks() -> list[dict]:
    return [
        {
            "example_id": f"bird_train_{index:05d}",
            "instance_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 60}",
            "question": f"question {index}",
            "external_knowledge": "hint" if index % 2 else None,
        }
        for index in range(600)
    ]


def _rows(*, mixed: int = 360, core: int = 240) -> list[dict]:
    rows = []
    for task_index in range(600):
        correct_count = 8
        if task_index < mixed:
            correct_count = 4 if task_index < core else 1
        for sample_index in range(8):
            correct = sample_index < correct_count
            rows.append(
                {
                    "schema_version": "table-agent-fixed-policy-episode-v1",
                    "sequence": task_index * 8 + sample_index,
                    "environment": {
                        "dataset_split": "train",
                        "task_id": f"bird_train_{task_index:05d}",
                        "example_index": task_index,
                        "db_id": f"db_{task_index % 60}",
                        "db_path": None,
                        "question": f"question {task_index}",
                        "gold_sql": None,
                        "external_knowledge": "hint" if task_index % 2 else None,
                    },
                    "sample": {
                        "reward": float(correct),
                        "correct": correct,
                        "failure_type": None,
                        "step_rewards": None,
                        "process_update": True,
                        "audit_record": {
                            "example_index": task_index,
                            "sample_index": sample_index,
                            "trajectory_id": (
                                f"rl_{task_index}_sample_{sample_index}"
                            ),
                            "protocol_version": EXPECTED_PROTOCOL_VERSION,
                            "protocol_hash": EXPECTED_PROTOCOL_HASH,
                            "correct": correct,
                            "legal": correct,
                            "failure_type": None,
                            "turns": [],
                            "error_events": [],
                            "result_reward": {
                                "profile": "binary",
                                "correct": correct,
                                "value": float(correct),
                            },
                        },
                    },
                    "policy_turns": [
                        {
                            "prompt_ids": [1],
                            "response_ids": [2],
                            "sampling_logprobs": [-0.1],
                        }
                    ],
                }
            )
    return rows


def _bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()


def _manifest(
    rows: list[dict], task_bytes: bytes, trajectory_bytes: bytes, tasks_path: str = "/tasks.jsonl"
) -> dict:
    return {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_path": tasks_path,
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "tasks": 600,
        "group_size": 8,
        "trajectories": 4800,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "denotation_comparison": "bird-set",
        "seed": 20260812,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "trajectories_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
        "correct_trajectories": sum(row["sample"]["correct"] for row in rows),
    }


def _audit(rows: list[dict], tasks: list[dict] | None = None) -> dict:
    tasks = tasks or _tasks()
    task_bytes = _bytes(tasks)
    trajectory_bytes = _bytes(rows)
    manifest = _manifest(rows, task_bytes, trajectory_bytes)
    return audit(
        manifest,
        rows,
        tasks,
        manifest_sha256="manifest-sha",
        trajectories_sha256=hashlib.sha256(trajectory_bytes).hexdigest(),
        tasks_sha256=hashlib.sha256(task_bytes).hexdigest(),
        tasks_path="/tasks.jsonl",
    )


def test_valid_s1_is_selection_ready_at_frozen_thresholds() -> None:
    result = _audit(_rows())
    assert result["status"]["audit_passes"]
    assert result["status"]["selection_ready_without_s2"]
    assert result["status"]["next_stage"] == "select_s1"
    assert result["observed"]["mixed_boundary_groups"] == 360
    assert result["observed"]["core_boundary_groups"] == 240


def test_low_signal_pool_is_valid_but_directs_s2() -> None:
    result = _audit(_rows(mixed=359, core=240))
    assert result["status"]["audit_passes"]
    assert not result["status"]["selection_ready_without_s2"]
    assert result["status"]["next_stage"] == "screen_s2"


def test_runtime_contamination_invalidates_whole_group_not_audit() -> None:
    rows = _rows()
    record = rows[0]["sample"]["audit_record"]
    record["error_events"] = [
        {"error_type": "timeout_error", "error_code": "tool_execution_timeout"}
    ]
    result = _audit(rows)
    assert result["status"]["audit_passes"]
    assert not result["groups"][0]["usable"]
    assert result["groups"][0]["contamination"] == ["timeout_evidence"]
    assert result["observed"]["mixed_boundary_groups"] == 359


def test_first_turn_runtime_failure_may_have_no_policy_evidence() -> None:
    rows = _rows()
    sample = rows[0]["sample"]
    sample["correct"] = False
    sample["reward"] = 0.0
    sample["failure_type"] = "context_overflow"
    sample["process_update"] = False
    record = sample["audit_record"]
    record["correct"] = False
    record["legal"] = False
    record["failure_type"] = "context_overflow"
    record["optimization_exclusion"] = "nonsemantic_runtime_failure"
    record["result_reward"]["correct"] = False
    record["result_reward"]["value"] = 0.0
    rows[0]["policy_turns"] = []
    result = _audit(rows)
    assert result["status"]["audit_passes"]
    assert not result["groups"][0]["usable"]
    assert "context_overflow" in result["groups"][0]["contamination"]


def test_clean_row_cannot_omit_policy_evidence() -> None:
    rows = _rows()
    rows[0]["policy_turns"] = []
    result = _audit(rows)
    assert not result["status"]["audit_passes"]
    assert result["issue_counts"]["policy_evidence_invalid"] == 1


def test_environment_and_record_correctness_are_bound_to_task_and_sample() -> None:
    rows = _rows()
    rows[0]["environment"]["db_id"] = "wrong"
    rows[1]["sample"]["audit_record"]["correct"] = False
    result = _audit(rows)
    assert not result["status"]["audit_passes"]
    assert result["issue_counts"]["trajectory_environment_mismatch"] == 1
    assert result["issue_counts"]["nonbinary_reward_contract"] == 1


def test_wrong_policy_step_or_protocol_fails_structural_audit() -> None:
    rows = _rows()
    rows[0]["sample"]["audit_record"]["policy_global_step"] = 1
    rows[1]["sample"]["audit_record"]["protocol_hash"] = "wrong"
    result = _audit(rows)
    assert not result["status"]["audit_passes"]
    assert result["issue_counts"] == {
        "optimizer_update_evidence": 1,
        "trajectory_protocol_mismatch": 1,
    }


def test_trajectory_identity_and_manifest_task_path_are_bound() -> None:
    tasks = _tasks()
    rows = _rows()
    rows[0]["sample"]["audit_record"]["trajectory_id"] = "wrong"
    task_bytes = _bytes(tasks)
    trajectory_bytes = _bytes(rows)
    result = audit(
        _manifest(rows, task_bytes, trajectory_bytes, "/other.jsonl"),
        rows,
        tasks,
        manifest_sha256="manifest-sha",
        trajectories_sha256=hashlib.sha256(trajectory_bytes).hexdigest(),
        tasks_sha256=hashlib.sha256(task_bytes).hexdigest(),
        tasks_path="/tasks.jsonl",
    )
    assert not result["status"]["audit_passes"]
    assert result["issue_counts"]["manifest_contract_mismatch"] == 1
    assert result["issue_counts"]["trajectory_identity_invalid"] == 1


def test_cli_writes_fixed_output_even_when_s2_is_needed(tmp_path: Path) -> None:
    tasks = _tasks()
    rows = _rows(mixed=100, core=80)
    task_path = tmp_path / "tasks.jsonl"
    trajectory_path = tmp_path / "trajectories.jsonl"
    manifest_path = tmp_path / "manifest.json"
    task_path.write_bytes(_bytes(tasks))
    trajectory_path.write_bytes(_bytes(rows))
    manifest_path.write_text(
        json.dumps(
            _manifest(
                rows,
                task_path.read_bytes(),
                trajectory_path.read_bytes(),
                str(task_path),
            )
        )
    )
    output_dir = tmp_path / "selection"
    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--trajectories",
            str(trajectory_path),
            "--tasks",
            str(task_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    result = json.loads((output_dir / "boundary_screen_audit.json").read_text())
    assert result["status"]["next_stage"] == "screen_s2"


def test_cli_revokes_stale_output_on_invalid_input(tmp_path: Path) -> None:
    output_dir = tmp_path / "selection"
    output_dir.mkdir()
    output = output_dir / "boundary_screen_audit.json"
    output.write_text('{"status":{"audit_passes":true}}')
    missing = tmp_path / "missing"
    assert main(
        [
            "--manifest",
            str(missing),
            "--trajectories",
            str(missing),
            "--tasks",
            str(missing),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    ) == 2
    assert not output.exists()
