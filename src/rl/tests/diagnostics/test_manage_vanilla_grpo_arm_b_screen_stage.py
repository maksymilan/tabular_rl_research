from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import rl.scenarios.diagnostics.manage_vanilla_grpo_arm_b_screen_stage as stage_io


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _fixture(tmp_path: Path, *, records: int = 6, shards: int = 2):
    run_root = tmp_path / "run"
    run_root.mkdir()
    inputs = run_root / "inputs"
    inputs.mkdir()
    tasks = inputs / "tasks.jsonl"
    rows = [
        {
            "example_id": f"bird_train_{index:05d}",
            "instance_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 3}",
            "question": f"question {index}",
        }
        for index in range(records)
    ]
    _write_jsonl(tasks, rows)
    tasks_manifest = inputs / "tasks.manifest.json"
    tasks_manifest.write_text('{"status":"frozen"}\n')
    activation = inputs / "activation.json"
    activation.write_text('{"status":"arm_a_failed_before_training"}\n')
    source = tmp_path / "source.py"
    source.write_text("SOURCE = 1\n")
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    protocol = tmp_path / "protocol"
    for directory in (model, adapter, protocol):
        directory.mkdir()
    request = stage_io.PlanRequest(
        stage="F1",
        run_root=run_root,
        tasks=tasks,
        tasks_manifest=tasks_manifest,
        expected_tasks_sha256=_sha(tasks),
        expected_tasks_manifest_sha256=_sha(tasks_manifest),
        activation=activation,
        expected_activation_sha256=_sha(activation),
        records=records,
        group_size=8,
        seed=20260814,
        gpu_memory_utilization=0.82,
        shards=shards,
        gpu_map=tuple(range(shards)),
        assignments_dir=run_root / "assignments" / "f1",
        plan=run_root / "plans" / "f1.json",
        model_path=model,
        adapter_path=adapter,
        protocol_runtime=protocol,
        bindings=(stage_io.Binding("generator", source, _sha(source)),),
    )
    return request, rows, source


def _episode(task: dict, *, position: int, sample_index: int, group_size: int) -> dict:
    correct = sample_index % 2 == 0
    return {
        "schema_version": "table-agent-fixed-policy-episode-v1",
        "sequence": position * group_size + sample_index,
        "environment": {
            "task_id": task["example_id"],
            "example_index": task["example_index"],
        },
        "sample": {
            "reward": float(correct),
            "correct": correct,
            "failure_type": None if correct else "incorrect_answer",
            "audit_record": {
                "example_index": task["example_index"],
                "sample_index": sample_index,
                "correct": correct,
                "failure_type": None if correct else "incorrect_answer",
                "result_reward": {
                    "profile": "binary",
                    "correct": correct,
                    "executable_terminal": correct,
                    "value": float(correct),
                },
            },
            "step_rewards": None,
            "process_update": True,
        },
        "policy_turns": [
            {
                "prompt_ids": [1, 2],
                "response_ids": [3],
                "sampling_logprobs": [-0.1],
            }
        ],
    }


def _complete_workers(request: stage_io.PlanRequest, rows: list[dict]) -> Path:
    plan = json.loads(request.plan.read_text())
    workers = request.run_root / "workers" / "f1"
    workers.mkdir(parents=True)
    by_id = {row["example_id"]: row for row in rows}
    positions = {row["example_id"]: index for index, row in enumerate(rows)}
    for shard_record in plan["shards"]["assignments"]:
        shard = shard_record["shard_index"]
        worker = workers / f"shard-{shard:03d}-of-{request.shards:03d}"
        groups = worker / "groups"
        groups.mkdir(parents=True)
        assigned = [
            line
            for line in Path(shard_record["path"]).read_text().splitlines()
            if line
        ]
        for task_id in assigned:
            payload = [
                _episode(
                    by_id[task_id],
                    position=positions[task_id],
                    sample_index=sample_index,
                    group_size=request.group_size,
                )
                for sample_index in range(request.group_size)
            ]
            (groups / f"{task_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
        result = stage_io.complete_worker(
            plan_path=request.plan,
            expected_plan_sha256=_sha(request.plan),
            shard_index=shard,
            worker_dir=worker,
            completion=worker / "worker_complete.json",
            quarantine_dir=request.run_root / "quarantine" / f"worker-{shard}",
            check_only=False,
        )
        assert result["status"] == "worker_completed"
    return workers


def _finish_transaction(
    request: stage_io.PlanRequest, rows: list[dict], workers: Path, transaction: Path
) -> None:
    plan = json.loads(request.plan.read_text())
    groups = transaction / "groups"
    trajectory_rows = []
    for row in rows:
        trajectory_rows.extend(json.loads((groups / f'{row["example_id"]}.json').read_text()))
    trajectory_rows.sort(key=lambda item: item["sequence"])
    trajectories = transaction / "trajectories.jsonl"
    _write_jsonl(trajectories, trajectory_rows)
    manifest = stage_io._manifest_expected_fields(plan)
    manifest.update(
        {
            "status": "generated_pending_counterfactual_validation",
            "tool_schema_sha256": "0" * 64,
            "trajectories_sha256": _sha(trajectories),
            "correct_trajectories": sum(
                row["sample"]["correct"] for row in trajectory_rows
            ),
        }
    )
    (transaction / "manifest.pending.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def test_plan_is_deterministic_and_binds_gpu_tasks_sources_and_assignments(
    tmp_path: Path,
) -> None:
    request, rows, _source = _fixture(tmp_path)
    first = stage_io.prepare_plan(request)
    assert first["status"] == "plan_prepared"
    second = stage_io.prepare_plan(request)
    assert second["status"] == "existing_plan_verified"
    plan = json.loads(request.plan.read_text())
    assert plan["stage"] == "F1"
    assert plan["shards"]["gpu_map"] == [0, 1]
    assert plan["tasks"]["sha256"] == _sha(request.tasks)
    assert plan["runtime"]["bindings"][0]["sha256"] == request.bindings[0].sha256
    assignments = plan["shards"]["assignments"]
    assert [record["records"] for record in assignments] == [3, 3]
    assert Path(assignments[0]["path"]).read_text().splitlines() == [
        rows[index]["example_id"] for index in (0, 2, 4)
    ]
    assert Path(assignments[1]["path"]).read_text().splitlines() == [
        rows[index]["example_id"] for index in (1, 3, 5)
    ]


def test_plan_fails_closed_on_changed_task_source_assignment_or_gpu_collision(
    tmp_path: Path,
) -> None:
    request, _rows, source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    source.write_text("SOURCE = 2\n")
    with pytest.raises(ValueError, match="binding SHA-256 mismatch"):
        stage_io.verify_plan(request)
    source.write_text("SOURCE = 1\n")
    assignment = Path(json.loads(request.plan.read_text())["shards"]["assignments"][0]["path"])
    assignment.write_text(assignment.read_text() + "unexpected\n")
    with pytest.raises(ValueError, match="stage plan differs|assignment bytes changed"):
        stage_io.verify_plan(request)

    duplicate_gpu = stage_io.PlanRequest(
        **{**request.__dict__, "gpu_map": (0, 0), "plan": tmp_path / "run/plans/other.json"}
    )
    with pytest.raises(ValueError, match="GPU map entries must be distinct"):
        stage_io.prepare_plan(duplicate_gpu)


def test_worker_marker_recovers_exact_next_and_rejects_group_corruption(
    tmp_path: Path,
) -> None:
    request, rows, _source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    workers = _complete_workers(request, rows)
    worker = workers / "shard-000-of-002"
    completion = worker / "worker_complete.json"
    payload = completion.read_bytes()
    completion.unlink()
    (worker / "worker_complete.json.next").write_bytes(payload)
    result = stage_io.complete_worker(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        shard_index=0,
        worker_dir=worker,
        completion=completion,
        quarantine_dir=request.run_root / "quarantine/worker-0",
        check_only=False,
    )
    assert result["status"] == "worker_completed"
    assert completion.read_bytes() == payload

    record = json.loads(completion.read_text())
    group = worker / "groups" / f'{record["group_files"][0]["task_id"]}.json'
    group.write_text("[]")
    with pytest.raises(ValueError, match="does not contain exact K"):
        stage_io.complete_worker(
            plan_path=request.plan,
            expected_plan_sha256=_sha(request.plan),
            shard_index=0,
            worker_dir=worker,
            completion=completion,
            quarantine_dir=request.run_root / "quarantine/worker-0",
            check_only=True,
        )


def test_recovered_correct_runtime_contamination_is_retained_for_auditor_exclusion(
    tmp_path: Path,
) -> None:
    request, rows, _source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    workers = _complete_workers(request, rows)
    worker = workers / "shard-000-of-002"
    completion = worker / "worker_complete.json"
    record = json.loads(completion.read_text())
    group = worker / "groups" / f'{record["group_files"][0]["task_id"]}.json'
    payload = json.loads(group.read_text())
    sample = payload[0]["sample"]
    sample["correct"] = True
    sample["reward"] = 0.0
    sample["process_update"] = False
    sample["failure_type"] = None
    audit = sample["audit_record"]
    audit["correct"] = True
    audit["failure_type"] = None
    audit["result_reward"] = {
        "profile": "binary",
        "correct": True,
        "executable_terminal": True,
        "value": 0.0,
    }
    audit["optimization_exclusion"] = "nonsemantic_runtime_failure"
    audit["error_events"] = [
        {"error_type": "timeout_error", "error_code": "tool_execution_timeout"}
    ]
    group.write_text(json.dumps(payload, separators=(",", ":")))
    completion.unlink()
    result = stage_io.complete_worker(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        shard_index=0,
        worker_dir=worker,
        completion=completion,
        quarantine_dir=request.run_root / "quarantine/worker-0",
        check_only=False,
    )
    assert result["status"] == "worker_completed"


def test_worker_refuses_plan_changed_after_sha_was_frozen(tmp_path: Path) -> None:
    request, rows, _source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    frozen_plan_sha = _sha(request.plan)
    workers = _complete_workers(request, rows)
    plan = json.loads(request.plan.read_text())
    plan["contract"]["seed"] = 1
    request.plan.write_text(json.dumps(plan) + "\n")
    worker = workers / "shard-000-of-002"
    with pytest.raises(ValueError, match="frozen stage plan SHA-256 mismatch"):
        stage_io.complete_worker(
            plan_path=request.plan,
            expected_plan_sha256=frozen_plan_sha,
            shard_index=0,
            worker_dir=worker,
            completion=worker / "worker_complete.json",
            quarantine_dir=request.run_root / "quarantine/worker-0",
            check_only=True,
        )


def test_pool_is_built_off_to_the_side_and_directory_published_atomically(
    tmp_path: Path,
) -> None:
    request, rows, _source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    workers = _complete_workers(request, rows)
    transaction = request.run_root / "transactions/f1-pool.next"
    pool = request.run_root / "pools/f1"
    result = stage_io.prepare_pool_transaction(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        workers_root=workers,
        transaction=transaction,
        pool=pool,
        quarantine_dir=request.run_root / "quarantine/pool",
    )
    assert result["status"] == "transaction_prepared"
    assert transaction.is_dir() and not pool.exists()
    _finish_transaction(request, rows, workers, transaction)
    verified = stage_io.verify_pool(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        pool=transaction,
        workers_root=workers,
    )
    assert verified["groups"] == 6 and verified["trajectories"] == 48
    published = stage_io.publish_pool(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        transaction=transaction,
        pool=pool,
        workers_root=workers,
    )
    assert published["status"] == "canonical_pool_published"
    assert pool.is_dir() and not transaction.exists()
    assert set(path.name for path in pool.iterdir()) == {
        "groups",
        "trajectories.jsonl",
        "manifest.pending.json",
    }


def test_known_incomplete_transaction_is_quarantined_but_unknown_file_blocks(
    tmp_path: Path,
) -> None:
    request, rows, _source = _fixture(tmp_path)
    stage_io.prepare_plan(request)
    workers = _complete_workers(request, rows)
    transaction = request.run_root / "transactions/f1-pool.next"
    pool = request.run_root / "pools/f1"
    quarantine = request.run_root / "quarantine/pool"
    stage_io.prepare_pool_transaction(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        workers_root=workers,
        transaction=transaction,
        pool=pool,
        quarantine_dir=quarantine,
    )
    (transaction / "trajectories.jsonl.next").write_text("partial")
    result = stage_io.prepare_pool_transaction(
        plan_path=request.plan,
        expected_plan_sha256=_sha(request.plan),
        workers_root=workers,
        transaction=transaction,
        pool=pool,
        quarantine_dir=quarantine,
    )
    assert result["status"] == "transaction_prepared"
    assert any(path.name.startswith("f1-pool.next") for path in quarantine.iterdir())

    (transaction / "unknown.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown entry"):
        stage_io.prepare_pool_transaction(
            plan_path=request.plan,
            expected_plan_sha256=_sha(request.plan),
            workers_root=workers,
            transaction=transaction,
            pool=pool,
            quarantine_dir=quarantine,
        )


def test_owned_nested_symlink_escape_is_rejected(tmp_path: Path) -> None:
    request, _rows, _source = _fixture(tmp_path)
    escape = tmp_path / "escape"
    escape.mkdir()
    (request.run_root / "link").symlink_to(escape, target_is_directory=True)
    escaped = stage_io.PlanRequest(
        **{
            **request.__dict__,
            "assignments_dir": request.run_root / "link/assignments",
            "plan": request.run_root / "link/plan.json",
        }
    )
    with pytest.raises(ValueError, match="escapes run root|crosses a symlink"):
        stage_io.prepare_plan(escaped)
