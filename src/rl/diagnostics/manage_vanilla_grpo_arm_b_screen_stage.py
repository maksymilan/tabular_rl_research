#!/usr/bin/env python3
"""Transactional stage I/O for the preregistered vanilla-GRPO Arm B screen.

This helper deliberately contains no cohort-selection or readiness policy.  A
reviewed external preparer/auditor/selector owns those decisions.  The helper
only freezes one already-selected task file into deterministic GPU shard
assignments, validates atomic per-task rollout groups, and publishes a fully
assembled fixed pool with a directory rename.  In particular, a worker or an
interrupted ``--finalize-only`` pass can never create a partially canonical
pool that a downstream audit could mistake for a completed stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLAN_SCHEMA = "vanilla-grpo-arm-b-screen-stage-plan-v1"
PLAN_STATUS = "frozen_stage_plan"
WORKER_SCHEMA = "vanilla-grpo-arm-b-screen-worker-v1"
WORKER_STATUS = "complete"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
STAGES = {"F1", "F2", "F3"}

EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file: {path}",
    )


def _regular_dir(path: Path, label: str) -> None:
    _require(
        path.is_dir() and not path.is_symlink(),
        f"{label} must be a regular non-symlink directory: {path}",
    )


def _explicit_sha(value: str, label: str) -> None:
    _require(
        SHA256_RE.fullmatch(value) is not None,
        f"{label} must be an explicit lowercase SHA-256",
    )


def _resolve_under(path: Path, root: Path, label: str) -> None:
    _require(path.is_absolute(), f"{label} must be absolute: {path}")
    resolved_root = root.resolve()
    resolved = path.resolve()
    _require(
        resolved == resolved_root or resolved.is_relative_to(resolved_root),
        f"{label} escapes run root: {path}",
    )


def _reject_owned_symlinks(path: Path, root: Path, label: str) -> None:
    """Reject user-controlled symlink components without rejecting /home aliases."""

    _resolve_under(path, root, label)
    current = path
    while current != root.parent and current.parent != current:
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} crosses a symlink below run root: {current}")
        current = current.parent


def _json_object(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(
            isinstance(value, dict), f"{label} row {line_number} is not an object"
        )
        rows.append(value)
    return rows


def _task_id(row: Mapping[str, Any]) -> str:
    example_id = row.get("example_id")
    instance_id = row.get("instance_id")
    if example_id is not None and instance_id is not None:
        _require(str(example_id) == str(instance_id), "task identity fields disagree")
    value = example_id or instance_id
    _require(isinstance(value, str) and bool(value), "task has no stable identity")
    return value


def _task_rows(path: Path, expected_records: int) -> list[dict[str, Any]]:
    rows = _jsonl_rows(path, "stage tasks")
    identifiers = [_task_id(row) for row in rows]
    _require(
        len(rows) == len(set(identifiers)) == expected_records,
        f"stage tasks must contain exactly {expected_records} unique identities",
    )
    example_indices: list[int] = []
    for row in rows:
        value = row.get("example_index")
        _require(type(value) is int and value >= 0, "task example_index is invalid")
        example_indices.append(value)
    _require(
        len(example_indices) == len(set(example_indices)),
        "stage task example_index values must be unique",
    )
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".next")
    if temporary.exists():
        _regular_file(temporary, f"temporary output for {path.name}")
        _require(
            temporary.read_bytes() == payload,
            f"non-identical interrupted write blocks publication: {temporary}",
        )
    else:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    os.replace(temporary, path)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


@dataclass(frozen=True)
class Binding:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class PlanRequest:
    stage: str
    run_root: Path
    tasks: Path
    tasks_manifest: Path
    expected_tasks_sha256: str | None
    expected_tasks_manifest_sha256: str | None
    activation: Path
    expected_activation_sha256: str
    records: int
    group_size: int
    seed: int
    gpu_memory_utilization: float
    shards: int
    gpu_map: tuple[int, ...]
    assignments_dir: Path
    plan: Path
    model_path: Path
    adapter_path: Path
    protocol_runtime: Path
    bindings: tuple[Binding, ...]


def _parse_optional_sha(value: str | None, label: str) -> str | None:
    if value in (None, "-"):
        return None
    _explicit_sha(value, label)
    return value


def _validate_request(request: PlanRequest) -> tuple[list[dict[str, Any]], list[str]]:
    _require(request.stage in STAGES, f"unsupported Arm B stage: {request.stage}")
    _require(request.run_root.is_absolute(), "run root must be absolute")
    _require(request.records > 0, "records must be positive")
    _require(request.group_size in {8, 16}, "group size must be K8 or K16")
    _require(request.seed >= 0, "seed must be nonnegative")
    _require(
        0.0 < request.gpu_memory_utilization < 1.0,
        "GPU memory utilization must be between zero and one",
    )
    _require(1 <= request.shards <= request.records, "invalid shard count")
    _require(
        len(request.gpu_map) == request.shards,
        "GPU map must contain exactly one entry per shard",
    )
    _require(
        all(type(index) is int and index >= 0 for index in request.gpu_map),
        "GPU map entries must be nonnegative integers",
    )
    _require(
        len(set(request.gpu_map)) == len(request.gpu_map),
        "GPU map entries must be distinct",
    )
    for path, label in (
        (request.assignments_dir, "assignments directory"),
        (request.plan, "stage plan"),
    ):
        _reject_owned_symlinks(path, request.run_root, label)
    _regular_file(request.tasks, "stage tasks")
    _regular_file(request.tasks_manifest, "stage task manifest")
    _regular_file(request.activation, "Arm A failure evidence")
    _explicit_sha(request.expected_activation_sha256, "activation SHA-256")
    _require(
        _sha256(request.activation) == request.expected_activation_sha256,
        "Arm A failure evidence SHA-256 mismatch",
    )
    if request.expected_tasks_sha256 is not None:
        _require(
            _sha256(request.tasks) == request.expected_tasks_sha256,
            "stage task SHA-256 mismatch",
        )
    if request.expected_tasks_manifest_sha256 is not None:
        _require(
            _sha256(request.tasks_manifest)
            == request.expected_tasks_manifest_sha256,
            "stage task manifest SHA-256 mismatch",
        )
    rows = _task_rows(request.tasks, request.records)
    identifiers = [_task_id(row) for row in rows]
    names: set[str] = set()
    for binding in request.bindings:
        _require(
            re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", binding.name) is not None,
            f"invalid binding name: {binding.name}",
        )
        _require(binding.name not in names, f"duplicate binding: {binding.name}")
        names.add(binding.name)
        _explicit_sha(binding.sha256, f"{binding.name} SHA-256")
        _regular_file(binding.path, binding.name)
        _require(
            _sha256(binding.path) == binding.sha256,
            f"binding SHA-256 mismatch: {binding.name}",
        )
    _require(names, "at least one runtime/source binding is required")
    return rows, identifiers


def _assignment_name(stage: str, shard: int, shards: int) -> str:
    return f"{stage.lower()}.shard-{shard:03d}-of-{shards:03d}.txt"


def _assignment_payloads(
    identifiers: Sequence[str], *, stage: str, shards: int
) -> list[tuple[str, bytes, list[str]]]:
    result = []
    for shard in range(shards):
        assigned = [
            task_id for position, task_id in enumerate(identifiers) if position % shards == shard
        ]
        _require(bool(assigned), f"shard {shard} has no assigned tasks")
        payload = ("\n".join(assigned) + "\n").encode()
        result.append((_assignment_name(stage, shard, shards), payload, assigned))
    union = [task_id for _, _, assigned in result for task_id in assigned]
    _require(len(union) == len(set(union)) == len(identifiers), "shards overlap")
    _require(set(union) == set(identifiers), "shards do not cover stage tasks")
    return result


def _identity_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "task_id": _task_id(row),
            "example_index": int(row["example_index"]),
            "db_id": str(row.get("db_id") or ""),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _build_plan(
    request: PlanRequest,
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[tuple[str, bytes, list[str]]],
) -> dict[str, Any]:
    task_sha = _sha256(request.tasks)
    task_manifest_sha = _sha256(request.tasks_manifest)
    assignment_records = []
    for index, (name, payload, assigned) in enumerate(assignments):
        assignment_records.append(
            {
                "shard_index": index,
                "path": str((request.assignments_dir / name).resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "records": len(assigned),
                "first_task_id": assigned[0],
                "last_task_id": assigned[-1],
            }
        )
    return {
        "schema_version": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "stage": request.stage,
        "run_root": str(request.run_root.resolve()),
        "activation": {
            "path": str(request.activation.resolve()),
            "sha256": request.expected_activation_sha256,
        },
        "tasks": {
            "path": str(request.tasks.resolve()),
            "sha256": task_sha,
            "manifest_path": str(request.tasks_manifest.resolve()),
            "manifest_sha256": task_manifest_sha,
            "records": request.records,
            "identity_order_sha256": _identity_order_sha256(rows),
        },
        "contract": {
            "policy": "fresh initial-SFT1",
            "group_size": request.group_size,
            "trajectories": request.records * request.group_size,
            "seed": request.seed,
            "generation_seed_scheme": "sha256-task-sample-turn-v1",
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
            "scheduler": "static",
            "task_batch_size": 1,
            "question_window": None,
            "gpu_memory_utilization": request.gpu_memory_utilization,
            "optimizer_updates": 0,
            "screen_trajectories_reused": False,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "initial_adapter_sha256": EXPECTED_ADAPTER_SHA256,
            "protocol_runtime_content_tree_sha256": (
                EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256
            ),
        },
        "runtime": {
            "model_path": str(request.model_path.resolve()),
            "adapter_path": str(request.adapter_path.resolve()),
            "protocol_runtime": str(request.protocol_runtime.resolve()),
            "bindings": [
                {
                    "name": binding.name,
                    "path": str(binding.path.resolve()),
                    "sha256": binding.sha256,
                }
                for binding in sorted(request.bindings, key=lambda item: item.name)
            ],
        },
        "shards": {
            "count": request.shards,
            "gpu_map": list(request.gpu_map),
            "assignment_rule": "zero_based_task_position_modulo_shard_count",
            "assignments": assignment_records,
        },
    }


def _verify_plan_object(request: PlanRequest, plan: Mapping[str, Any]) -> dict[str, Any]:
    rows, identifiers = _validate_request(request)
    assignments = _assignment_payloads(
        identifiers, stage=request.stage, shards=request.shards
    )
    expected = _build_plan(request, rows, assignments)
    _require(plan == expected, "stage plan differs from the frozen expected contract")
    for name, payload, _assigned in assignments:
        path = request.assignments_dir / name
        _regular_file(path, f"assignment {name}")
        _require(path.read_bytes() == payload, f"assignment bytes changed: {path}")
    expected_names = {name for name, _, _ in assignments}
    entries = list(request.assignments_dir.iterdir())
    _require(
        all(
            entry.name in expected_names and entry.is_file() and not entry.is_symlink()
            for entry in entries
        ),
        "assignment directory contains an unknown or non-regular entry",
    )
    _require(
        {entry.name for entry in entries} == expected_names,
        "assignment directory is incomplete",
    )
    return dict(plan)


def prepare_plan(request: PlanRequest) -> dict[str, Any]:
    rows, identifiers = _validate_request(request)
    assignments = _assignment_payloads(
        identifiers, stage=request.stage, shards=request.shards
    )
    expected = _build_plan(request, rows, assignments)
    if request.plan.exists():
        plan = _json_object(request.plan, "stage plan")
        _verify_plan_object(request, plan)
        return {"status": "existing_plan_verified", "plan_sha256": _sha256(request.plan)}
    request.assignments_dir.mkdir(parents=True, exist_ok=True)
    _regular_dir(request.assignments_dir, "assignments directory")
    expected_names = {name for name, _, _ in assignments}
    for entry in request.assignments_dir.iterdir():
        if entry.name.endswith(".next") and entry.name[:-5] in expected_names:
            continue
        _require(
            entry.name in expected_names and entry.is_file() and not entry.is_symlink(),
            f"unknown assignment artifact blocks prepare: {entry}",
        )
    for name, payload, _assigned in assignments:
        path = request.assignments_dir / name
        if path.exists():
            _regular_file(path, f"existing assignment {name}")
            _require(path.read_bytes() == payload, f"existing assignment differs: {path}")
            temporary = path.with_name(path.name + ".next")
            if temporary.exists():
                _regular_file(temporary, f"assignment temporary {name}")
                _require(
                    temporary.read_bytes() == payload,
                    f"assignment temporary differs: {temporary}",
                )
                temporary.unlink()
        else:
            _atomic_write(path, payload)
    request.plan.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(request.plan, _json_bytes(expected))
    _verify_plan_object(request, _json_object(request.plan, "new stage plan"))
    return {"status": "plan_prepared", "plan_sha256": _sha256(request.plan)}


def verify_plan(request: PlanRequest) -> dict[str, Any]:
    plan = _json_object(request.plan, "stage plan")
    _verify_plan_object(request, plan)
    return {"status": "plan_verified", "plan_sha256": _sha256(request.plan)}


def _plan_task_index(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    task_record = plan["tasks"]
    tasks = Path(task_record["path"])
    rows = _task_rows(tasks, int(task_record["records"]))
    _require(_sha256(tasks) == task_record["sha256"], "plan task SHA-256 changed")
    identifiers = [_task_id(row) for row in rows]
    return rows, {task_id: index for index, task_id in enumerate(identifiers)}


def _verify_declared_plan(
    plan_path: Path, expected_plan_sha256: str
) -> dict[str, Any]:
    _explicit_sha(expected_plan_sha256, "frozen stage plan SHA-256")
    _regular_file(plan_path, "stage plan")
    _require(
        _sha256(plan_path) == expected_plan_sha256,
        "frozen stage plan SHA-256 mismatch",
    )
    plan = _json_object(plan_path, "stage plan")
    _require(plan.get("schema_version") == PLAN_SCHEMA, "stage plan schema mismatch")
    _require(plan.get("status") == PLAN_STATUS, "stage plan status mismatch")
    _require(plan.get("stage") in STAGES, "stage plan stage mismatch")
    run_root = Path(str(plan.get("run_root") or ""))
    _require(run_root.is_absolute(), "stage plan run root is not absolute")
    activation = plan.get("activation") or {}
    activation_path = Path(str(activation.get("path") or ""))
    activation_sha = str(activation.get("sha256") or "")
    _explicit_sha(activation_sha, "plan activation SHA-256")
    _regular_file(activation_path, "plan activation evidence")
    _require(_sha256(activation_path) == activation_sha, "plan activation changed")
    task_record = plan.get("tasks") or {}
    for key, label in (("path", "plan tasks"), ("manifest_path", "plan task manifest")):
        path = Path(str(task_record.get(key) or ""))
        _regular_file(path, label)
        sha_key = "sha256" if key == "path" else "manifest_sha256"
        _require(_sha256(path) == task_record.get(sha_key), f"{label} changed")
    bindings = ((plan.get("runtime") or {}).get("bindings") or [])
    _require(isinstance(bindings, list) and bool(bindings), "plan has no source bindings")
    seen: set[str] = set()
    for record in bindings:
        name = str(record.get("name") or "")
        _require(name and name not in seen, "plan source bindings are invalid")
        seen.add(name)
        path = Path(str(record.get("path") or ""))
        sha = str(record.get("sha256") or "")
        _explicit_sha(sha, f"plan binding {name} SHA-256")
        _regular_file(path, f"plan binding {name}")
        _require(_sha256(path) == sha, f"plan binding changed: {name}")
    return plan


def _assignment_for_shard(
    plan: Mapping[str, Any], shard_index: int
) -> tuple[Path, list[str]]:
    shards = plan["shards"]
    count = int(shards["count"])
    _require(0 <= shard_index < count, "shard index is outside the plan")
    records = shards["assignments"]
    _require(len(records) == count, "plan assignment record count mismatch")
    record = records[shard_index]
    _require(record["shard_index"] == shard_index, "plan shard ordering mismatch")
    path = Path(record["path"])
    _regular_file(path, "worker assignment")
    _require(_sha256(path) == record["sha256"], "worker assignment SHA-256 changed")
    identifiers = [line for line in path.read_text().splitlines() if line]
    _require(
        len(identifiers) == len(set(identifiers)) == record["records"],
        "worker assignment record count/identity mismatch",
    )
    return path, identifiers


def _verify_group(
    path: Path,
    *,
    task_id: str,
    task_row: Mapping[str, Any],
    position: int,
    group_size: int,
) -> list[dict[str, Any]]:
    _regular_file(path, f"group {task_id}")
    value = json.loads(path.read_bytes())
    _require(
        isinstance(value, list) and len(value) == group_size,
        f"group {task_id} does not contain exact K={group_size}",
    )
    for sample_index, row in enumerate(value):
        _require(isinstance(row, dict), f"group {task_id} row is not an object")
        _require(
            row.get("schema_version") == "table-agent-fixed-policy-episode-v1",
            f"group {task_id} episode schema mismatch",
        )
        _require(
            row.get("sequence") == position * group_size + sample_index,
            f"group {task_id} sequence mismatch",
        )
        environment = row.get("environment") or {}
        _require(environment.get("task_id") == task_id, f"group {task_id} task mismatch")
        _require(
            environment.get("example_index") == task_row["example_index"],
            f"group {task_id} example_index mismatch",
        )
        sample = row.get("sample") or {}
        audit = sample.get("audit_record")
        _require(
            isinstance(audit, dict),
            f"group {task_id} audit record is not an object",
        )
        correct = sample.get("correct")
        reward = sample.get("reward")
        process_update = sample.get("process_update")
        _require(type(correct) is bool, f"group {task_id} correctness is not boolean")
        _require(
            type(process_update) is bool,
            f"group {task_id} process_update is not an explicit boolean",
        )
        expected_reward = 0.0 if not process_update else float(correct)
        _require(
            type(reward) in {int, float} and float(reward) == expected_reward,
            f"group {task_id} reward does not match binary eligibility",
        )
        _require(
            sample.get("failure_type") == audit.get("failure_type"),
            f"group {task_id} sample/audit failure identity differs",
        )
        _require(
            audit.get("correct") is correct,
            f"group {task_id} sample/audit correctness differs",
        )
        result_reward = audit.get("result_reward")
        _require(
            isinstance(result_reward, dict)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and type(result_reward.get("value")) in {int, float}
            and float(result_reward["value"]) == expected_reward,
            f"group {task_id} audit result_reward is inconsistent",
        )
        _require(
            (
                audit.get("optimization_exclusion")
                == "nonsemantic_runtime_failure"
                if not process_update
                else "optimization_exclusion" not in audit
            ),
            f"group {task_id} optimization exclusion is inconsistent",
        )
        _require(
            audit.get("example_index") == task_row["example_index"],
            f"group {task_id} audit example_index mismatch",
        )
        _require(
            audit.get("sample_index") == sample_index,
            f"group {task_id} audit sample_index mismatch",
        )
        turns = row.get("policy_turns")
        _require(isinstance(turns, list), f"group {task_id} policy turns are invalid")
        for turn in turns:
            _require(isinstance(turn, dict), f"group {task_id} policy turn is invalid")
            prompt = turn.get("prompt_ids")
            response = turn.get("response_ids")
            logprobs = turn.get("sampling_logprobs")
            _require(
                isinstance(prompt, list)
                and isinstance(response, list)
                and isinstance(logprobs, list)
                and len(response) == len(logprobs),
                f"group {task_id} token/logprob evidence is malformed",
            )
    return value


def _expected_worker_record(
    plan_path: Path,
    plan: Mapping[str, Any],
    shard_index: int,
    worker_dir: Path,
) -> dict[str, Any]:
    assignment, identifiers = _assignment_for_shard(plan, shard_index)
    rows, positions = _plan_task_index(plan)
    by_id = {_task_id(row): row for row in rows}
    group_size = int(plan["contract"]["group_size"])
    groups_dir = worker_dir / "groups"
    _regular_dir(groups_dir, "worker groups")
    expected_names = {f"{task_id}.json" for task_id in identifiers}
    entries = list(groups_dir.iterdir())
    _require(
        len(entries) == len(expected_names)
        and all(
            entry.name in expected_names and entry.is_file() and not entry.is_symlink()
            for entry in entries
        ),
        "worker group directory is incomplete or contains an unknown entry",
    )
    files = []
    for task_id in identifiers:
        path = groups_dir / f"{task_id}.json"
        _verify_group(
            path,
            task_id=task_id,
            task_row=by_id[task_id],
            position=positions[task_id],
            group_size=group_size,
        )
        files.append({"task_id": task_id, "sha256": _sha256(path)})
    return {
        "schema_version": WORKER_SCHEMA,
        "status": WORKER_STATUS,
        "stage": plan["stage"],
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)},
        "assignment": {
            "path": str(assignment.resolve()),
            "sha256": _sha256(assignment),
            "records": len(identifiers),
        },
        "shard_index": shard_index,
        "shards": int(plan["shards"]["count"]),
        "tasks": len(identifiers),
        "group_size": group_size,
        "trajectories": len(identifiers) * group_size,
        "seed": int(plan["contract"]["seed"]),
        "group_files": files,
    }


def complete_worker(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    shard_index: int,
    worker_dir: Path,
    completion: Path,
    quarantine_dir: Path,
    check_only: bool,
) -> dict[str, Any]:
    plan = _verify_declared_plan(plan_path, expected_plan_sha256)
    run_root = Path(plan["run_root"])
    for path, label in (
        (worker_dir, "worker directory"),
        (completion, "worker completion"),
        (quarantine_dir, "worker marker quarantine"),
    ):
        _reject_owned_symlinks(path, run_root, label)
    _regular_dir(worker_dir, "worker directory")
    _require(
        completion.parent.resolve() == worker_dir.resolve(),
        "worker completion must be directly under the worker directory",
    )
    expected = _expected_worker_record(plan_path, plan, shard_index, worker_dir)
    payload = _json_bytes(expected)
    allowed = {"groups", completion.name, completion.name + ".next"}
    _require(
        all(entry.name in allowed for entry in worker_dir.iterdir()),
        "worker directory contains an unknown entry",
    )
    temporary = completion.with_name(completion.name + ".next")
    if completion.exists():
        _regular_file(completion, "worker completion")
        _require(completion.read_bytes() == payload, "worker completion bytes changed")
        if temporary.exists():
            _regular_file(temporary, "worker completion temporary")
            _require(
                temporary.read_bytes() == payload,
                "worker completion temporary differs from expected bytes",
            )
            if check_only:
                raise ValueError("completed worker retains an uncommitted marker")
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            destination = _unique_destination(quarantine_dir, temporary.name)
            os.replace(temporary, destination)
        return {"status": "worker_verified", "completion_sha256": _sha256(completion)}
    _require(not check_only, f"worker completion is missing: {completion}")
    if temporary.exists():
        _regular_file(temporary, "worker completion temporary")
        _require(
            temporary.read_bytes() == payload,
            "interrupted worker completion differs from expected bytes",
        )
        os.replace(temporary, completion)
    else:
        _atomic_write(completion, payload)
    _require(completion.read_bytes() == payload, "worker completion publication failed")
    return {"status": "worker_completed", "completion_sha256": _sha256(completion)}


def _unique_destination(directory: Path, name: str) -> Path:
    candidate = directory / name
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{name}.{suffix}"
        suffix += 1
    return candidate


def _worker_directory(workers_root: Path, shard: int, shards: int) -> Path:
    return workers_root / f"shard-{shard:03d}-of-{shards:03d}"


def _verify_all_workers(
    plan_path: Path,
    expected_plan_sha256: str,
    plan: Mapping[str, Any],
    workers_root: Path,
) -> dict[str, Path]:
    run_root = Path(plan["run_root"])
    _reject_owned_symlinks(workers_root, run_root, "workers root")
    _regular_dir(workers_root, "workers root")
    shards = int(plan["shards"]["count"])
    expected_dirs = {
        _worker_directory(workers_root, shard, shards).name for shard in range(shards)
    }
    entries = list(workers_root.iterdir())
    _require(
        {entry.name for entry in entries} == expected_dirs
        and all(entry.is_dir() and not entry.is_symlink() for entry in entries),
        "workers root is incomplete or contains an unknown entry",
    )
    sources: dict[str, Path] = {}
    for shard in range(shards):
        worker = _worker_directory(workers_root, shard, shards)
        completion = worker / "worker_complete.json"
        result = complete_worker(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            shard_index=shard,
            worker_dir=worker,
            completion=completion,
            quarantine_dir=worker / ".unused-check-only-quarantine",
            check_only=True,
        )
        _require(result["status"] == "worker_verified", "worker verification failed")
        record = _json_object(completion, "worker completion")
        for item in record["group_files"]:
            task_id = item["task_id"]
            _require(task_id not in sources, f"worker groups overlap: {task_id}")
            source = worker / "groups" / f"{task_id}.json"
            _require(_sha256(source) == item["sha256"], f"worker group changed: {task_id}")
            sources[task_id] = source
    rows, _positions = _plan_task_index(plan)
    identifiers = [_task_id(row) for row in rows]
    _require(set(sources) == set(identifiers), "workers do not exactly cover stage tasks")
    return sources


def _allowed_incomplete_transaction(
    transaction: Path, expected_group_names: set[str]
) -> None:
    allowed_root = {
        "groups",
        "trajectories.jsonl",
        "trajectories.jsonl.next",
        "manifest.pending.json",
        "manifest.pending.json.next",
    }
    entries = list(transaction.iterdir())
    _require(
        all(entry.name in allowed_root for entry in entries),
        "pool transaction contains an unknown entry",
    )
    groups = transaction / "groups"
    if groups.exists():
        _regular_dir(groups, "transaction groups")
        allowed_groups = expected_group_names | {
            f"{name}.next" for name in expected_group_names
        }
        _require(
            all(
                entry.name in allowed_groups
                and entry.is_file()
                and not entry.is_symlink()
                for entry in groups.iterdir()
            ),
            "pool transaction groups contain an unknown entry",
        )


def prepare_pool_transaction(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    workers_root: Path,
    transaction: Path,
    pool: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    plan = _verify_declared_plan(plan_path, expected_plan_sha256)
    run_root = Path(plan["run_root"])
    for path, label in (
        (transaction, "pool transaction"),
        (pool, "canonical pool"),
        (quarantine_dir, "pool transaction quarantine"),
    ):
        _reject_owned_symlinks(path, run_root, label)
    sources = _verify_all_workers(
        plan_path, expected_plan_sha256, plan, workers_root
    )
    rows, _positions = _plan_task_index(plan)
    identifiers = [_task_id(row) for row in rows]
    expected_names = {f"{task_id}.json" for task_id in identifiers}
    if pool.exists():
        verify_pool(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            pool=pool,
            workers_root=workers_root,
        )
        return {"status": "canonical_pool_already_verified", "pool": str(pool.resolve())}
    if transaction.exists():
        _regular_dir(transaction, "pool transaction")
        try:
            verify_pool(
                plan_path=plan_path,
                expected_plan_sha256=expected_plan_sha256,
                pool=transaction,
                workers_root=workers_root,
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            _allowed_incomplete_transaction(transaction, expected_names)
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            destination = _unique_destination(quarantine_dir, transaction.name)
            os.replace(transaction, destination)
        else:
            return {
                "status": "complete_transaction_verified",
                "transaction": str(transaction.resolve()),
            }
    transaction.parent.mkdir(parents=True, exist_ok=True)
    transaction.mkdir()
    groups = transaction / "groups"
    groups.mkdir()
    for task_id in identifiers:
        source = sources[task_id]
        target = groups / f"{task_id}.json"
        try:
            os.link(source, target)
        except OSError as exc:
            raise ValueError(
                f"cannot hard-link immutable worker group into transaction: {source}: {exc}"
            ) from exc
    _require(
        {entry.name for entry in groups.iterdir()} == expected_names,
        "new pool transaction group coverage mismatch",
    )
    return {"status": "transaction_prepared", "transaction": str(transaction.resolve())}


def _manifest_expected_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan["contract"]
    runtime = plan["runtime"]
    return {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "protocol_version": contract["protocol_version"],
        "protocol_hash": contract["protocol_hash"],
        "protocol_runtime_root": runtime["protocol_runtime"],
        "protocol_runtime_content_tree_sha256": contract[
            "protocol_runtime_content_tree_sha256"
        ],
        "model_path": runtime["model_path"],
        "adapter_path": runtime["adapter_path"],
        "adapter_sha256": contract["initial_adapter_sha256"],
        "tasks_path": plan["tasks"]["path"],
        "tasks_sha256": plan["tasks"]["sha256"],
        "tasks": plan["tasks"]["records"],
        "group_size": contract["group_size"],
        "trajectories": contract["trajectories"],
        "reward_mode": contract["reward_mode"],
        "result_reward_profile": contract["result_reward_profile"],
        "temperature": contract["temperature"],
        "top_p": contract["top_p"],
        "max_steps": contract["max_steps"],
        "max_new_tokens": contract["max_new_tokens"],
        "max_context_tokens": contract["max_context_tokens"],
        "history_turns": contract["history_turns"],
        "enable_thinking": contract["enable_thinking"],
        "seed": contract["seed"],
        "generation_task_batch_size": contract["task_batch_size"],
        "generation_scheduler": contract["scheduler"],
        "generation_question_window": contract["question_window"],
        "generation_seed_scheme": contract["generation_seed_scheme"],
        "denotation_comparison": contract["denotation_comparison"],
        "student_prompt_sha256": contract["student_prompt_sha256"],
    }


def verify_pool(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    pool: Path,
    workers_root: Path,
) -> dict[str, Any]:
    plan = _verify_declared_plan(plan_path, expected_plan_sha256)
    run_root = Path(plan["run_root"])
    _reject_owned_symlinks(pool, run_root, "fixed pool")
    _regular_dir(pool, "fixed pool")
    entries = list(pool.iterdir())
    _require(
        {entry.name for entry in entries}
        == {"groups", "trajectories.jsonl", "manifest.pending.json"},
        "fixed pool contains a partial or unknown root entry",
    )
    for entry in entries:
        _require(not entry.is_symlink(), f"fixed pool entry is a symlink: {entry}")
    sources = _verify_all_workers(
        plan_path, expected_plan_sha256, plan, workers_root
    )
    rows, positions = _plan_task_index(plan)
    by_id = {_task_id(row): row for row in rows}
    identifiers = [_task_id(row) for row in rows]
    group_size = int(plan["contract"]["group_size"])
    groups = pool / "groups"
    _regular_dir(groups, "fixed pool groups")
    group_entries = list(groups.iterdir())
    expected_names = {f"{task_id}.json" for task_id in identifiers}
    _require(
        {entry.name for entry in group_entries} == expected_names
        and all(entry.is_file() and not entry.is_symlink() for entry in group_entries),
        "fixed pool group coverage mismatch",
    )
    expected_rows: list[dict[str, Any]] = []
    for task_id in identifiers:
        path = groups / f"{task_id}.json"
        _require(_sha256(path) == _sha256(sources[task_id]), f"pool group differs: {task_id}")
        expected_rows.extend(
            _verify_group(
                path,
                task_id=task_id,
                task_row=by_id[task_id],
                position=positions[task_id],
                group_size=group_size,
            )
        )
    expected_rows.sort(key=lambda row: int(row["sequence"]))
    trajectories = pool / "trajectories.jsonl"
    actual_rows = _jsonl_rows(trajectories, "fixed pool trajectories")
    _require(actual_rows == expected_rows, "fixed pool trajectories differ from groups")
    _require(
        [int(row["sequence"]) for row in actual_rows]
        == list(range(len(expected_rows))),
        "fixed pool trajectories are not contiguous from zero",
    )
    manifest = _json_object(pool / "manifest.pending.json", "fixed pool manifest")
    for key, expected in _manifest_expected_fields(plan).items():
        _require(manifest.get(key) == expected, f"fixed pool manifest mismatch: {key}")
    _require(
        manifest.get("trajectories_sha256") == _sha256(trajectories),
        "fixed pool trajectory digest mismatch",
    )
    _require(
        manifest.get("correct_trajectories")
        == sum(bool(row["sample"]["correct"]) for row in actual_rows),
        "fixed pool correctness total mismatch",
    )
    return {
        "status": "pool_verified",
        "manifest_sha256": _sha256(pool / "manifest.pending.json"),
        "trajectories_sha256": _sha256(trajectories),
        "groups": len(identifiers),
        "trajectories": len(actual_rows),
    }


def publish_pool(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    transaction: Path,
    pool: Path,
    workers_root: Path,
) -> dict[str, Any]:
    plan = _verify_declared_plan(plan_path, expected_plan_sha256)
    run_root = Path(plan["run_root"])
    for path, label in ((transaction, "pool transaction"), (pool, "canonical pool")):
        _reject_owned_symlinks(path, run_root, label)
    if pool.exists():
        result = verify_pool(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            pool=pool,
            workers_root=workers_root,
        )
        result["status"] = "canonical_pool_already_published"
        return result
    verify_pool(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        pool=transaction,
        workers_root=workers_root,
    )
    pool.parent.mkdir(parents=True, exist_ok=True)
    _require(
        transaction.parent.stat().st_dev == pool.parent.stat().st_dev,
        "pool transaction and canonical parent are on different filesystems",
    )
    os.replace(transaction, pool)
    result = verify_pool(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        pool=pool,
        workers_root=workers_root,
    )
    result["status"] = "canonical_pool_published"
    return result


def _bindings(values: Sequence[Sequence[str]]) -> tuple[Binding, ...]:
    result = []
    for value in values:
        _require(len(value) == 3, "--bind requires NAME PATH SHA256")
        name, raw_path, sha = value
        result.append(Binding(name=name, path=Path(raw_path), sha256=sha))
    return tuple(result)


def _plan_request(args: argparse.Namespace) -> PlanRequest:
    try:
        gpu_map = tuple(int(value) for value in args.gpu_map.split(","))
    except ValueError as exc:
        raise ValueError("--gpu-map must be a comma-separated integer list") from exc
    return PlanRequest(
        stage=args.stage,
        run_root=args.run_root,
        tasks=args.tasks,
        tasks_manifest=args.tasks_manifest,
        expected_tasks_sha256=_parse_optional_sha(
            args.expected_tasks_sha256, "expected task SHA-256"
        ),
        expected_tasks_manifest_sha256=_parse_optional_sha(
            args.expected_tasks_manifest_sha256, "expected task manifest SHA-256"
        ),
        activation=args.activation,
        expected_activation_sha256=args.expected_activation_sha256,
        records=args.records,
        group_size=args.group_size,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        shards=args.shards,
        gpu_map=gpu_map,
        assignments_dir=args.assignments_dir,
        plan=args.plan,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        protocol_runtime=args.protocol_runtime,
        bindings=_bindings(args.bind),
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--expected-tasks-sha256", default="-")
    parser.add_argument("--expected-tasks-manifest-sha256", default="-")
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--expected-activation-sha256", required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gpu-map", required=True)
    parser.add_argument("--assignments-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--protocol-runtime", type=Path, required=True)
    parser.add_argument(
        "--bind",
        nargs=3,
        action="append",
        metavar=("NAME", "PATH", "SHA256"),
        default=[],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-plan")
    _add_plan_arguments(prepare)
    verify = subparsers.add_parser("verify-plan")
    _add_plan_arguments(verify)

    for name in ("complete-worker", "verify-worker"):
        worker = subparsers.add_parser(name)
        worker.add_argument("--plan", type=Path, required=True)
        worker.add_argument("--expected-plan-sha256", required=True)
        worker.add_argument("--shard-index", type=int, required=True)
        worker.add_argument("--worker-dir", type=Path, required=True)
        worker.add_argument("--completion", type=Path, required=True)
        worker.add_argument("--quarantine-dir", type=Path, required=True)

    transaction = subparsers.add_parser("prepare-pool-transaction")
    transaction.add_argument("--plan", type=Path, required=True)
    transaction.add_argument("--expected-plan-sha256", required=True)
    transaction.add_argument("--workers-root", type=Path, required=True)
    transaction.add_argument("--transaction", type=Path, required=True)
    transaction.add_argument("--pool", type=Path, required=True)
    transaction.add_argument("--quarantine-dir", type=Path, required=True)

    for name in ("verify-pool", "publish-pool"):
        pool = subparsers.add_parser(name)
        pool.add_argument("--plan", type=Path, required=True)
        pool.add_argument("--expected-plan-sha256", required=True)
        pool.add_argument("--workers-root", type=Path, required=True)
        pool.add_argument("--pool", type=Path, required=True)
        if name == "publish-pool":
            pool.add_argument("--transaction", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-plan":
            result = prepare_plan(_plan_request(args))
        elif args.command == "verify-plan":
            result = verify_plan(_plan_request(args))
        elif args.command in {"complete-worker", "verify-worker"}:
            result = complete_worker(
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                shard_index=args.shard_index,
                worker_dir=args.worker_dir,
                completion=args.completion,
                quarantine_dir=args.quarantine_dir,
                check_only=args.command == "verify-worker",
            )
        elif args.command == "prepare-pool-transaction":
            result = prepare_pool_transaction(
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                workers_root=args.workers_root,
                transaction=args.transaction,
                pool=args.pool,
                quarantine_dir=args.quarantine_dir,
            )
        elif args.command == "verify-pool":
            result = verify_pool(
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                pool=args.pool,
                workers_root=args.workers_root,
            )
        elif args.command == "publish-pool":
            result = publish_pool(
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                transaction=args.transaction,
                pool=args.pool,
                workers_root=args.workers_root,
            )
        else:  # pragma: no cover - argparse prevents this branch.
            raise AssertionError(args.command)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Arm B stage I/O blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
