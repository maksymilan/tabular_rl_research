from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.rl.diagnostics import merge_boundary_screen_crosshost as module
from src.rl.diagnostics import audit_vanilla_grpo_boundary_screen as boundary_auditor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _group_rows(task: dict, position: int) -> list[dict]:
    rows = []
    task_id = task["example_id"]
    example_index = task["example_index"]
    for sample_index in range(module.EXPECTED_GROUP_SIZE):
        correct = sample_index % 2 == 0
        rows.append(
            {
                "schema_version": module.TRAJECTORY_SCHEMA_VERSION,
                "sequence": position * module.EXPECTED_GROUP_SIZE + sample_index,
                "environment": {
                    "dataset_split": "train",
                    "example_index": example_index,
                    "task_id": task_id,
                    "db_id": task["db_id"],
                    "db_path": task["db_path"],
                    "question": task["question"],
                    "gold_sql": task["gold_sql"],
                    "external_knowledge": task["external_knowledge"],
                },
                "sample": {
                    "reward": float(correct),
                    "correct": correct,
                    "failure_type": None,
                    "step_rewards": None,
                    "process_update": True,
                    "audit_record": {
                        "example_index": example_index,
                        "sample_index": sample_index,
                        "trajectory_id": f"rl_{example_index}_sample_{sample_index}",
                        "protocol_version": module.EXPECTED_PROTOCOL_VERSION,
                        "protocol_hash": module.EXPECTED_PROTOCOL_HASH,
                        "correct": correct,
                        "legal": True,
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
                        "prompt_ids": [1, position + 2],
                        "response_ids": [sample_index + 10],
                        "sampling_logprobs": [-0.1],
                    }
                ],
            }
        )
    return rows


def _group_bytes(rows: list[dict]) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


def _configure_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EXPECTED_TASKS", 12)
    monkeypatch.setattr(module, "EXPECTED_FIRST32", 4)
    monkeypatch.setattr(module, "EXPECTED_PLAN_COMPLETED", 6)
    monkeypatch.setattr(module, "EXPECTED_PLAN_REMAINING", 6)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    _configure_small_contract(monkeypatch)
    tasks = [
        {
            "example_id": f"bird_train_{index:05d}",
            "instance_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 3}",
            "db_path": f"/db/db_{index % 3}.sqlite",
            "question": f"question {index}",
            "gold_sql": f"SELECT {index}",
            "external_knowledge": "hint" if index % 2 else None,
        }
        for index in range(module.EXPECTED_TASKS)
    ]
    tasks_path = tmp_path / "train600.jsonl"
    _jsonl(tasks_path, tasks)
    tasks_sha = _sha(tasks_path)
    monkeypatch.setattr(module, "EXPECTED_TASKS_SHA256", tasks_sha)
    task_ids = [row["example_id"] for row in tasks]
    tasks_manifest_path = tmp_path / "train600.manifest.json"
    tasks_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.TASK_MANIFEST_SCHEMA_VERSION,
                "status": "frozen_training_cohort",
                "all_acceptance_gates_passed": True,
                "selection": {"seed": "test-frozen-seed"},
                "output": {
                    "sha256": tasks_sha,
                    "records": module.EXPECTED_TASKS,
                    "task_ids_in_frozen_order": task_ids,
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(module, "EXPECTED_TASKS_MANIFEST_SHA256", _sha(tasks_manifest_path))

    group_payloads = {
        task["example_id"]: _group_bytes(_group_rows(task, position))
        for position, task in enumerate(tasks)
    }
    first32 = tmp_path / "first32"
    (first32 / "groups").mkdir(parents=True)
    first_rows = []
    for task in tasks[: module.EXPECTED_FIRST32]:
        payload = group_payloads[task["example_id"]]
        (first32 / "groups" / f"{task['example_id']}.json").write_bytes(payload)
        first_rows.extend(json.loads(payload))
    first_trajectories = first32 / "trajectories.jsonl"
    _jsonl(first_trajectories, first_rows)
    first_trajectories_sha = _sha(first_trajectories)
    first_manifest = {
        "schema_version": module.PENDING_MANIFEST_SCHEMA_VERSION,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": module.EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": module.EXPECTED_PROTOCOL_HASH,
        "protocol_runtime_root": module.EXPECTED_PROTOCOL_RUNTIME_ROOT,
        "protocol_runtime_content_tree_sha256": module.EXPECTED_RUNTIME_TREE_SHA256,
        "model_path": module.EXPECTED_MODEL_PATH,
        "adapter_path": module.EXPECTED_ADAPTER_PATH,
        "adapter_sha256": module.EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_path": str(tasks_path),
        "tasks_sha256": tasks_sha,
        "tasks": module.EXPECTED_FIRST32,
        "group_size": module.EXPECTED_GROUP_SIZE,
        "trajectories": module.EXPECTED_FIRST32 * module.EXPECTED_GROUP_SIZE,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": module.EXPECTED_SEED,
        "generation_task_batch_size": 1,
        "generation_scheduler": "dynamic",
        "generation_question_window": 4,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "student_prompt_sha256": module.EXPECTED_STUDENT_PROMPT_SHA256,
        "tool_schema_sha256": "a" * 64,
        "trajectories_sha256": first_trajectories_sha,
        "correct_trajectories": sum(row["sample"]["correct"] for row in first_rows),
    }
    first_manifest_path = first32 / "manifest.pending.json"
    first_manifest_path.write_text(json.dumps(first_manifest, indent=2) + "\n")

    completed = tmp_path / "old_s1_groups"
    completed.mkdir()
    completed_records = {}
    for task in tasks[: module.EXPECTED_PLAN_COMPLETED]:
        task_id = task["example_id"]
        payload = group_payloads[task_id]
        path = completed / f"{task_id}.json"
        path.write_bytes(payload)
        completed_records[task_id] = {
            "path": f"/table_rl/old/groups/{task_id}.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": module.EXPECTED_GROUP_SIZE,
        }

    remaining = task_ids[module.EXPECTED_PLAN_COMPLETED :]
    assignment_ids = [remaining[index :: module.EXPECTED_SHARDS] for index in range(module.EXPECTED_SHARDS)]
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    assignment_paths = {}
    shard_dirs = {}
    assignment_records = []
    for shard, ids in enumerate(assignment_ids):
        assignment = plan_dir / f"remaining.shard-{shard:02d}-of-04.txt"
        assignment.write_text("".join(f"{task_id}\n" for task_id in ids))
        assignment_paths[shard] = assignment
        shard_dir = tmp_path / f"shard-{shard}" / "groups"
        shard_dir.mkdir(parents=True)
        shard_dirs[shard] = shard_dir
        for task_id in ids:
            (shard_dir / f"{task_id}.json").write_bytes(group_payloads[task_id])
        assignment_records.append(
            {
                "shard": shard,
                "path": f"/remote/plan/{assignment.name}",
                "sha256": _sha(assignment),
                "tasks": len(ids),
                "task_ids": ids,
            }
        )
    plan = {
        "schema_version": module.PLAN_SCHEMA_VERSION,
        "status": "frozen_continuation_plan",
        "tasks": {"path": "/table_rl/train600.jsonl", "sha256": tasks_sha, "records": module.EXPECTED_TASKS},
        "group_size": module.EXPECTED_GROUP_SIZE,
        "generation_contract": {
            "initial_policy": module.EXPECTED_INITIAL_POLICY,
            "optimizer_updates": 0,
            "temperature": 0.8,
            "top_p": 1.0,
            "seed": module.EXPECTED_SEED,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "enable_thinking": True,
        },
        "completed": {
            "tasks": module.EXPECTED_PLAN_COMPLETED,
            "trajectories": module.EXPECTED_PLAN_COMPLETED * module.EXPECTED_GROUP_SIZE,
            "groups": completed_records,
        },
        "remaining": {
            "tasks": module.EXPECTED_PLAN_REMAINING,
            "trajectories": module.EXPECTED_PLAN_REMAINING * module.EXPECTED_GROUP_SIZE,
        },
        "host_capability_partition": {},
        "assignments": assignment_records,
        "acceptance_gates": {
            "completed_and_remaining_cover_600": True,
            "assignments_cover_remaining_once": True,
            "first32_already_complete": True,
        },
    }
    plan_path = plan_dir / "continuation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    output = tmp_path / "canonical_staging_crosshost_v2"
    return {
        "tasks_path": tasks_path,
        "tasks_manifest_path": tasks_manifest_path,
        "first32_pool": first32,
        "expected_first32_manifest_sha256": _sha(first_manifest_path),
        "expected_first32_trajectories_sha256": first_trajectories_sha,
        "completed_group_dirs": [completed],
        "plan_path": plan_path,
        "expected_plan_sha256": _sha(plan_path),
        "assignment_paths": assignment_paths,
        "shard_group_dirs": shard_dirs,
        "output_dir": output,
        "task_ids": task_ids,
    }


def _merge(arguments: dict) -> dict:
    allowed = {
        key: value
        for key, value in arguments.items()
        if key != "task_ids"
    }
    return module.merge(**allowed)


def test_valid_crosshost_merge_is_exact_and_whole_directory_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    result = _merge(arguments)
    output = arguments["output_dir"]
    assert result["status"] == "verified_and_atomically_published"
    assert result["groups"] == module.EXPECTED_TASKS
    assert result["trajectories"] == module.EXPECTED_TASKS * module.EXPECTED_GROUP_SIZE
    assert len(list((output / "groups").glob("*.json"))) == module.EXPECTED_TASKS
    rows = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    manifest = json.loads((output / "manifest.pending.json").read_text())
    assert manifest["tasks"] == module.EXPECTED_TASKS
    assert manifest["trajectories_sha256"] == _sha(output / "trajectories.jsonl")
    assert manifest["crosshost_merge"]["plan_completed_tasks"] == module.EXPECTED_PLAN_COMPLETED
    audit = json.loads((output / "crosshost_merge_audit.json").read_text())
    # first32 exists in both its immutable source pool and the remapped old S1 root.
    assert audit["observed"]["duplicate_task_groups"] == module.EXPECTED_FIRST32
    assert audit["contract"]["row_index_definition"].startswith("frozen task position")
    monkeypatch.setattr(boundary_auditor, "EXPECTED_TASKS", module.EXPECTED_TASKS)
    external = boundary_auditor.audit(
        manifest,
        rows,
        [json.loads(line) for line in arguments["tasks_path"].read_text().splitlines()],
        manifest_sha256=_sha(output / "manifest.pending.json"),
        trajectories_sha256=_sha(output / "trajectories.jsonl"),
        tasks_sha256=_sha(arguments["tasks_path"]),
        tasks_path=str(arguments["tasks_path"]),
    )
    assert external["status"]["audit_passes"] is True


def test_non_byte_identical_duplicate_is_rejected_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    duplicate = arguments["completed_group_dirs"][0] / f"{arguments['task_ids'][0]}.json"
    duplicate.write_bytes(duplicate.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not byte-identical"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_missing_shard_group_is_rejected_without_partial_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    missing = next(arguments["shard_group_dirs"][0].glob("*.json"))
    missing.unlink()
    with pytest.raises(ValueError, match="do not exactly cover"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()
    assert not list(tmp_path.glob(".canonical_staging_crosshost_v2.next-*"))


def test_row_protocol_or_row_index_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    target = next(arguments["shard_group_dirs"][1].glob("*.json"))
    rows = json.loads(target.read_text())
    rows[0]["sample"]["audit_record"]["protocol_hash"] = "wrong"
    target.write_text(json.dumps(rows, separators=(",", ":")))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_group_task_identity_or_k8_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    target = next(arguments["shard_group_dirs"][2].glob("*.json"))
    rows = json.loads(target.read_text())
    rows[0]["environment"]["question"] = "different question"
    target.write_text(json.dumps(rows, separators=(",", ":")))
    with pytest.raises(ValueError, match="task/environment identity mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_exact_k8_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    target = next(arguments["shard_group_dirs"][3].glob("*.json"))
    rows = json.loads(target.read_text())
    target.write_text(json.dumps(rows[:-1], separators=(",", ":")))
    with pytest.raises(ValueError, match="not exact K8"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_optional_row_policy_identity_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    target = next(arguments["shard_group_dirs"][0].glob("*.json"))
    rows = json.loads(target.read_text())
    rows[0]["sample"]["audit_record"]["adapter_sha256"] = "0" * 64
    target.write_text(json.dumps(rows, separators=(",", ":")))
    with pytest.raises(ValueError, match="optional adapter_sha256 identity mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_assignment_sha_or_plan_task_order_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    assignment = arguments["assignment_paths"][0]
    lines = assignment.read_text().splitlines()
    assignment.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="assignment shard 0 SHA-256 mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_plan_generation_identity_and_completed_group_hash_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    plan = json.loads(arguments["plan_path"].read_text())
    plan["generation_contract"]["seed"] = module.EXPECTED_SEED + 1
    arguments["plan_path"].write_text(json.dumps(plan, indent=2) + "\n")
    arguments["expected_plan_sha256"] = _sha(arguments["plan_path"])
    with pytest.raises(ValueError, match="plan generation contract mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()

    arguments = _fixture(tmp_path / "second", monkeypatch)
    plan = json.loads(arguments["plan_path"].read_text())
    completed_id = arguments["task_ids"][module.EXPECTED_FIRST32]
    plan["completed"]["groups"][completed_id]["sha256"] = "f" * 64
    arguments["plan_path"].write_text(json.dumps(plan, indent=2) + "\n")
    arguments["expected_plan_sha256"] = _sha(arguments["plan_path"])
    with pytest.raises(ValueError, match="plan completed group SHA mismatch"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("student_prompt_sha256", "0" * 64, "student_prompt_sha256 mismatch"),
        ("model_path", "/wrong/model", "model_path mismatch"),
        ("adapter_sha256", "1" * 64, "adapter_sha256 mismatch"),
        ("seed", 17, "seed mismatch"),
    ],
)
def test_first32_model_prompt_adapter_or_seed_identity_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    manifest_path = arguments["first32_pool"] / "manifest.pending.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest) + "\n")
    arguments["expected_first32_manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match=message):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_first32_groups_must_byte_reconstruct_frozen_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    target = next((arguments["first32_pool"] / "groups").glob("*.json"))
    rows = json.loads(target.read_text())
    rows[0] = {"sequence": rows[0]["sequence"], **rows[0]}
    target.write_text(json.dumps(rows, separators=(",", ":")))
    with pytest.raises(ValueError, match="do not byte-reconstruct"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()


def test_existing_output_or_publication_failure_never_mutates_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path, monkeypatch)
    arguments["output_dir"].mkdir()
    marker = arguments["output_dir"] / "owned.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="output directory already exists"):
        _merge(arguments)
    assert marker.read_text() == "keep"

    arguments = _fixture(tmp_path / "second", monkeypatch)

    def refuse_publish(_source: Path, _destination: Path) -> None:
        raise FileExistsError("simulated publication race")

    monkeypatch.setattr(module.os, "rename", refuse_publish)
    with pytest.raises(FileExistsError, match="simulated publication race"):
        _merge(arguments)
    assert not arguments["output_dir"].exists()
    assert not list(arguments["output_dir"].parent.glob(".canonical_staging_crosshost_v2.next-*"))


def test_named_path_parser_requires_exact_four_explicit_shards(tmp_path: Path) -> None:
    values = [f"{index}={tmp_path / str(index)}" for index in range(4)]
    parsed = module._parse_named_paths(values, "assignment")
    assert set(parsed) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="exactly shards 0..3"):
        module._parse_named_paths(values[:3], "assignment")
