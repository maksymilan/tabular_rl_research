#!/usr/bin/env python3
"""Fail-closed cross-host merge for the frozen Qwen3 v26 S1 K8 screen.

This is intentionally a pure-CPU, standard-library artifact operation.  It
never loads a model, opens a database, contacts a host, or mutates any source
pool.  All source groups are audited before a new staging directory is
published with one atomic directory rename.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "qwen3-v26-boundary-screen-crosshost-merge-v1"
PLAN_SCHEMA_VERSION = "qwen3-v26-boundary-screen-continuation-plan-v1"
TRAJECTORY_SCHEMA_VERSION = "table-agent-fixed-policy-episode-v1"
PENDING_MANIFEST_SCHEMA_VERSION = "table-agent-fixed-rollout-pool-pending-v1"
TASK_MANIFEST_SCHEMA_VERSION = "bird-train-vanilla-grpo-cohort-v1"

EXPECTED_TASKS_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
EXPECTED_TASKS_MANIFEST_SHA256 = (
    "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6"
)
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_MODEL_PATH = "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline"
EXPECTED_ADAPTER_PATH = (
    "/home/dengyan/tabular_rl_outputs/checkpoints/"
    "qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
)
EXPECTED_PROTOCOL_RUNTIME_ROOT = (
    "/home/dengyan/tabular_rl_outputs/"
    "runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
)
EXPECTED_INITIAL_POLICY = "qwen3-8b-atomic-v26-sft1-checkpoint560"
EXPECTED_TASKS = 600
EXPECTED_GROUP_SIZE = 8
EXPECTED_FIRST32 = 32
EXPECTED_PLAN_COMPLETED = 69
EXPECTED_PLAN_REMAINING = 531
EXPECTED_SHARDS = 4
EXPECTED_SEED = 20260812
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceGroup:
    task_id: str
    path: Path
    payload: bytes
    rows: list[dict[str, Any]]
    role: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_file(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} must be a regular non-symlink file: {path}")


def _require_dir(path: Path, label: str) -> None:
    _require(path.is_dir() and not path.is_symlink(), f"{label} must be a regular non-symlink directory: {path}")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require_file(path, label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(isinstance(value, dict), f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def _task_id(task: Mapping[str, Any]) -> str:
    value = task.get("example_id") or task.get("instance_id")
    _require(isinstance(value, str) and bool(value), "task lacks a stable identity")
    return value


def _validate_tasks(tasks_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(sha256_file(tasks_path) == EXPECTED_TASKS_SHA256, "train600 task SHA-256 mismatch")
    _require(sha256_file(manifest_path) == EXPECTED_TASKS_MANIFEST_SHA256, "train600 manifest SHA-256 mismatch")
    tasks = _load_jsonl(tasks_path, "train600 tasks")
    _require(len(tasks) == EXPECTED_TASKS, f"train600 must contain exactly {EXPECTED_TASKS} rows")
    ids: list[str] = []
    indices: list[int] = []
    for task in tasks:
        task_id = _task_id(task)
        index = task.get("example_index")
        _require(type(index) is int, f"{task_id}: example_index is not an integer")
        _require(task.get("instance_id") == task_id, f"{task_id}: instance_id differs")
        _require(task_id == f"bird_train_{index:05d}", f"{task_id}: identity/index mismatch")
        _require(isinstance(task.get("db_id"), str), f"{task_id}: missing db_id")
        _require(isinstance(task.get("db_path"), str), f"{task_id}: missing db_path")
        _require(isinstance(task.get("question"), str), f"{task_id}: missing question")
        _require(isinstance(task.get("gold_sql") or task.get("query"), str), f"{task_id}: missing harness gold SQL")
        ids.append(task_id)
        indices.append(index)
    _require(len(set(ids)) == EXPECTED_TASKS, "train600 task ids are not unique")
    _require(len(set(indices)) == EXPECTED_TASKS, "train600 example_index values are not unique")
    manifest = _load_object(manifest_path, "train600 manifest")
    _require(manifest.get("schema_version") == TASK_MANIFEST_SCHEMA_VERSION, "train600 manifest schema mismatch")
    _require(manifest.get("status") == "frozen_training_cohort", "train600 manifest is not frozen")
    _require(manifest.get("all_acceptance_gates_passed") is True, "train600 acceptance gates did not pass")
    output = manifest.get("output") or {}
    _require(output.get("sha256") == EXPECTED_TASKS_SHA256, "train600 manifest/output SHA mismatch")
    _require(output.get("records") == EXPECTED_TASKS, "train600 manifest record count mismatch")
    _require(output.get("task_ids_in_frozen_order") == ids, "train600 manifest frozen order mismatch")
    return tasks, manifest


def _validate_plan(
    plan_path: Path,
    expected_plan_sha256: str,
    tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], set[str], set[str]]:
    _require(SHA_PATTERN.fullmatch(expected_plan_sha256) is not None, "expected plan SHA-256 must be explicit lowercase hex")
    _require(sha256_file(plan_path) == expected_plan_sha256, "continuation plan SHA-256 mismatch")
    plan = _load_object(plan_path, "continuation plan")
    _require(plan.get("schema_version") == PLAN_SCHEMA_VERSION, "continuation plan schema mismatch")
    _require(plan.get("status") == "frozen_continuation_plan", "continuation plan is not frozen")
    plan_tasks = plan.get("tasks") or {}
    _require(plan_tasks.get("sha256") == EXPECTED_TASKS_SHA256, "plan task SHA-256 mismatch")
    _require(plan_tasks.get("records") == EXPECTED_TASKS, "plan task count mismatch")
    _require(plan.get("group_size") == EXPECTED_GROUP_SIZE, "plan is not K8")
    generation = plan.get("generation_contract") or {}
    expected_generation = {
        "initial_policy": EXPECTED_INITIAL_POLICY,
        "optimizer_updates": 0,
        "temperature": 0.8,
        "top_p": 1.0,
        "seed": EXPECTED_SEED,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "enable_thinking": True,
    }
    _require(generation == expected_generation, f"plan generation contract mismatch: {generation!r}")
    completed = plan.get("completed") or {}
    remaining = plan.get("remaining") or {}
    _require(completed.get("tasks") == EXPECTED_PLAN_COMPLETED, "plan completed task count mismatch")
    _require(completed.get("trajectories") == EXPECTED_PLAN_COMPLETED * EXPECTED_GROUP_SIZE, "plan completed trajectory count mismatch")
    _require(remaining.get("tasks") == EXPECTED_PLAN_REMAINING, "plan remaining task count mismatch")
    _require(remaining.get("trajectories") == EXPECTED_PLAN_REMAINING * EXPECTED_GROUP_SIZE, "plan remaining trajectory count mismatch")
    completed_groups = completed.get("groups")
    _require(isinstance(completed_groups, dict) and len(completed_groups) == EXPECTED_PLAN_COMPLETED, "plan completed group map mismatch")
    completed_ids = set(completed_groups)
    task_ids = [_task_id(task) for task in tasks]
    _require(completed_ids <= set(task_ids), "plan completed set contains unknown tasks")
    for task_id, record in completed_groups.items():
        _require(isinstance(record, dict), f"plan completed record is invalid: {task_id}")
        _require(record.get("rows") == EXPECTED_GROUP_SIZE, f"plan completed group is not K8: {task_id}")
        _require(SHA_PATTERN.fullmatch(str(record.get("sha256") or "")) is not None, f"plan completed group lacks SHA-256: {task_id}")

    raw_assignments = plan.get("assignments")
    _require(isinstance(raw_assignments, list) and len(raw_assignments) == EXPECTED_SHARDS, "plan must contain exactly four assignments")
    assignments: dict[int, dict[str, Any]] = {}
    assigned_ids: list[str] = []
    for record in raw_assignments:
        _require(isinstance(record, dict), "plan assignment is not an object")
        shard = record.get("shard")
        _require(type(shard) is int and 0 <= shard < EXPECTED_SHARDS and shard not in assignments, "plan shard ids must be unique 0..3")
        ids = record.get("task_ids")
        _require(isinstance(ids, list) and all(isinstance(value, str) and value for value in ids), f"plan shard {shard} has invalid task ids")
        _require(record.get("tasks") == len(ids), f"plan shard {shard} count mismatch")
        _require(SHA_PATTERN.fullmatch(str(record.get("sha256") or "")) is not None, f"plan shard {shard} lacks assignment SHA")
        assignments[shard] = record
        assigned_ids.extend(ids)
    _require(set(assignments) == set(range(EXPECTED_SHARDS)), "plan shard ids are not exactly 0..3")
    _require(len(assigned_ids) == len(set(assigned_ids)) == EXPECTED_PLAN_REMAINING, "plan assignments overlap or have wrong cardinality")
    assigned_set = set(assigned_ids)
    _require(not completed_ids & assigned_set, "plan completed and remaining tasks overlap")
    _require(completed_ids | assigned_set == set(task_ids), "plan does not cover train600 exactly")
    gates = plan.get("acceptance_gates") or {}
    _require(
        gates
        == {
            "completed_and_remaining_cover_600": True,
            "assignments_cover_remaining_once": True,
            "first32_already_complete": True,
        },
        "plan acceptance gates differ from the frozen continuation schema",
    )
    _require(set(task_ids[:EXPECTED_FIRST32]) <= completed_ids, "plan does not bind the frozen first32 as complete")
    return plan, assignments, completed_ids, assigned_set


def _parse_named_paths(values: Sequence[str], label: str) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        shard_text, separator, raw_path = value.partition("=")
        _require(bool(separator) and shard_text.isdigit() and bool(raw_path), f"{label} must use SHARD=/absolute/path")
        shard = int(shard_text)
        _require(shard not in parsed, f"duplicate {label} shard {shard}")
        path = Path(raw_path)
        _require(path.is_absolute(), f"{label} path must be absolute: {path}")
        parsed[shard] = path
    _require(set(parsed) == set(range(EXPECTED_SHARDS)), f"{label} must map exactly shards 0..3")
    return parsed


def _validate_assignments(plan_assignments: Mapping[int, dict[str, Any]], paths: Mapping[int, Path]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for shard in range(EXPECTED_SHARDS):
        path = paths[shard]
        _require_file(path, f"assignment shard {shard}")
        payload = path.read_bytes()
        record = plan_assignments[shard]
        _require(sha256_bytes(payload) == record["sha256"], f"assignment shard {shard} SHA-256 mismatch")
        ids = [line.decode("utf-8").strip() for line in payload.splitlines() if line.strip()]
        _require(ids == record["task_ids"], f"assignment shard {shard} task order differs from plan")
        _require(len(ids) == len(set(ids)) == record["tasks"], f"assignment shard {shard} has duplicate/wrong task count")
        result[shard] = ids
    return result


def _read_group(path: Path, *, task_id: str, role: str) -> SourceGroup:
    _require_file(path, f"{role} group")
    payload = path.read_bytes()
    value = json.loads(payload)
    _require(isinstance(value, list) and all(isinstance(row, dict) for row in value), f"{role} group is not a JSON object array: {path}")
    return SourceGroup(task_id=task_id, path=path, payload=payload, rows=value, role=role)


def _directory_groups(path: Path, role: str) -> dict[str, SourceGroup]:
    _require_dir(path, role)
    result: dict[str, SourceGroup] = {}
    for entry in sorted(path.iterdir(), key=lambda item: item.name):
        _require(entry.is_file() and not entry.is_symlink() and entry.suffix == ".json", f"{role} contains unknown/non-regular entry: {entry}")
        task_id = entry.stem
        _require(task_id not in result, f"duplicate filename in {role}: {task_id}")
        result[task_id] = _read_group(entry, task_id=task_id, role=role)
    return result


def _is_finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_policy_evidence(row: Mapping[str, Any], failure_type: Any, process_update: Any, location: str) -> None:
    turns = row.get("policy_turns")
    empty_runtime = turns == [] and process_update is False and failure_type in {"generation_oom", "context_overflow"}
    if empty_runtime:
        return
    _require(isinstance(turns, list) and bool(turns), f"{location}: missing policy token evidence")
    for turn_index, turn in enumerate(turns):
        _require(isinstance(turn, dict), f"{location}: turn {turn_index} is not an object")
        prompt = turn.get("prompt_ids")
        response = turn.get("response_ids")
        logprobs = turn.get("sampling_logprobs")
        _require(isinstance(prompt, list) and bool(prompt) and all(type(value) is int for value in prompt), f"{location}: invalid prompt token ids")
        _require(isinstance(response, list) and bool(response) and all(type(value) is int for value in response), f"{location}: invalid response token ids")
        _require(isinstance(logprobs, list) and len(logprobs) == len(response) and all(_is_finite(value) for value in logprobs), f"{location}: invalid sampled logprobs")


def _validate_group(group: SourceGroup, *, task: Mapping[str, Any], row_index: int) -> None:
    _require(len(group.rows) == EXPECTED_GROUP_SIZE, f"{group.role} group is not exact K8: {group.path}")
    task_id = _task_id(task)
    example_index = task.get("example_index")
    expected_environment = {
        "dataset_split": "train",
        "example_index": example_index,
        "task_id": task_id,
        "db_id": task.get("db_id"),
        "db_path": task.get("db_path"),
        "question": task.get("question"),
        "gold_sql": task.get("gold_sql") or task.get("query"),
        "external_knowledge": task.get("external_knowledge"),
    }
    for sample_index, row in enumerate(group.rows):
        location = f"{group.path}[{sample_index}]"
        sequence = row_index * EXPECTED_GROUP_SIZE + sample_index
        _require(row.get("schema_version") == TRAJECTORY_SCHEMA_VERSION, f"{location}: trajectory schema mismatch")
        _require(row.get("sequence") == sequence, f"{location}: row_index/sequence mismatch; expected {sequence}")
        environment = row.get("environment")
        sample = row.get("sample")
        _require(isinstance(environment, dict) and isinstance(sample, dict), f"{location}: missing environment/sample")
        _require(all(environment.get(key) == value for key, value in expected_environment.items()), f"{location}: task/environment identity mismatch")
        audit = sample.get("audit_record")
        _require(isinstance(audit, dict), f"{location}: missing audit record")
        _require(audit.get("example_index") == example_index, f"{location}: audit example_index mismatch")
        _require(audit.get("sample_index") == sample_index, f"{location}: audit sample_index mismatch")
        _require(audit.get("trajectory_id") == f"rl_{example_index}_sample_{sample_index}", f"{location}: trajectory identity mismatch")
        _require(audit.get("protocol_version") == EXPECTED_PROTOCOL_VERSION, f"{location}: protocol version mismatch")
        _require(audit.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, f"{location}: protocol hash mismatch")
        for key, expected in {
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "model_path": EXPECTED_MODEL_PATH,
            "seed": EXPECTED_SEED,
        }.items():
            if key in audit:
                _require(audit[key] == expected, f"{location}: optional {key} identity mismatch")
        optimizer = {
            key: audit[key]
            for key in ("policy_global_step", "policy_micro_step", "optimizer_step", "global_step")
            if key in audit
        }
        _require(all(type(value) is int and value == 0 for value in optimizer.values()), f"{location}: nonzero/invalid optimizer evidence")
        correct = sample.get("correct")
        reward = sample.get("reward")
        _require(isinstance(correct, bool), f"{location}: correctness is not boolean")
        _require(audit.get("correct") is correct, f"{location}: sample/audit correctness mismatch")
        _require(_is_finite(reward), f"{location}: reward is not finite")
        _require(sample.get("step_rewards") is None, f"{location}: nonterminal step rewards are present")
        _require(sample.get("failure_type") == audit.get("failure_type"), f"{location}: failure identity mismatch")
        result_reward = audit.get("result_reward")
        _require(isinstance(result_reward, dict) and result_reward.get("profile") == "binary", f"{location}: result reward profile mismatch")
        _require(result_reward.get("correct") is correct and _is_finite(result_reward.get("value")), f"{location}: binary result reward identity mismatch")
        expected_reward = 0.0 if sample.get("process_update") is False else float(correct)
        _require(float(reward) == expected_reward and float(result_reward["value"]) == float(reward), f"{location}: binary reward value mismatch")
        _require(isinstance(sample.get("process_update"), bool), f"{location}: process_update is not explicit boolean")
        _validate_policy_evidence(row, sample.get("failure_type"), sample.get("process_update"), location)


def _validate_first32(
    pool: Path,
    *,
    expected_manifest_sha256: str,
    expected_trajectories_sha256: str,
    tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, SourceGroup]]:
    _require_dir(pool, "first32 pool")
    _require(SHA_PATTERN.fullmatch(expected_manifest_sha256) is not None, "first32 manifest SHA must be explicit lowercase hex")
    _require(SHA_PATTERN.fullmatch(expected_trajectories_sha256) is not None, "first32 trajectories SHA must be explicit lowercase hex")
    manifest_path = pool / "manifest.pending.json"
    trajectories_path = pool / "trajectories.jsonl"
    _require(sha256_file(manifest_path) == expected_manifest_sha256, "first32 manifest SHA-256 mismatch")
    _require(sha256_file(trajectories_path) == expected_trajectories_sha256, "first32 trajectories SHA-256 mismatch")
    manifest = _load_object(manifest_path, "first32 manifest")
    expected = {
        "schema_version": PENDING_MANIFEST_SCHEMA_VERSION,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "protocol_runtime_root": EXPECTED_PROTOCOL_RUNTIME_ROOT,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "model_path": EXPECTED_MODEL_PATH,
        "adapter_path": EXPECTED_ADAPTER_PATH,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "tasks": EXPECTED_FIRST32,
        "group_size": EXPECTED_GROUP_SIZE,
        "trajectories": EXPECTED_FIRST32 * EXPECTED_GROUP_SIZE,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": EXPECTED_SEED,
        "generation_task_batch_size": 1,
        "generation_scheduler": "dynamic",
        "generation_question_window": 4,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "trajectories_sha256": expected_trajectories_sha256,
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"first32 manifest {key} mismatch")
    _require(
        SHA_PATTERN.fullmatch(str(manifest.get("tool_schema_sha256") or ""))
        is not None,
        "first32 manifest tool schema SHA-256 is missing or invalid",
    )
    groups = _directory_groups(pool / "groups", "first32 groups")
    first_ids = [_task_id(task) for task in tasks[:EXPECTED_FIRST32]]
    _require(set(groups) == set(first_ids), "first32 group identities differ from frozen first32")
    trajectory_rows = _load_jsonl(trajectories_path, "first32 trajectories")
    _require(len(trajectory_rows) == EXPECTED_FIRST32 * EXPECTED_GROUP_SIZE, "first32 trajectory count mismatch")
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        task_id = (row.get("environment") or {}).get("task_id")
        _require(isinstance(task_id, str), "first32 trajectory lacks task identity")
        by_id.setdefault(task_id, []).append(row)
    for position, task_id in enumerate(first_ids):
        _require(groups[task_id].rows == by_id.get(task_id), f"first32 group/trajectories differ: {task_id}")
        _validate_group(groups[task_id], task=tasks[position], row_index=position)
    reconstructed = _compact_jsonl(
        [row for task_id in first_ids for row in groups[task_id].rows]
    )
    _require(
        reconstructed == trajectories_path.read_bytes(),
        "first32 groups do not byte-reconstruct the frozen trajectories",
    )
    _require(manifest.get("correct_trajectories") == sum(bool(row["sample"]["correct"]) for row in trajectory_rows), "first32 correct count mismatch")
    return manifest, groups


def _compact_jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _source_record(group: SourceGroup) -> dict[str, Any]:
    return {
        "role": group.role,
        "path": str(group.path.resolve()),
        "sha256": sha256_bytes(group.payload),
    }


def merge(
    *,
    tasks_path: Path,
    tasks_manifest_path: Path,
    first32_pool: Path,
    expected_first32_manifest_sha256: str,
    expected_first32_trajectories_sha256: str,
    completed_group_dirs: Sequence[Path],
    plan_path: Path,
    expected_plan_sha256: str,
    assignment_paths: Mapping[int, Path],
    shard_group_dirs: Mapping[int, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate all inputs, then atomically publish a new canonical staging pool."""

    _require(output_dir.is_absolute(), "output directory must be absolute")
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(output_dir.name not in {"pool_current600_s1", "groups"}, "refusing a legacy/canonical source directory as output")
    _require(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(), "output parent must already be a regular directory")
    tasks, tasks_manifest = _validate_tasks(tasks_path, tasks_manifest_path)
    task_ids = [_task_id(task) for task in tasks]
    task_by_id = {task_id: task for task_id, task in zip(task_ids, tasks, strict=True)}
    plan, plan_assignments, completed_ids, assigned_ids = _validate_plan(plan_path, expected_plan_sha256, tasks)
    assignment_ids = _validate_assignments(plan_assignments, assignment_paths)
    first_manifest, first_groups = _validate_first32(
        first32_pool,
        expected_manifest_sha256=expected_first32_manifest_sha256,
        expected_trajectories_sha256=expected_first32_trajectories_sha256,
        tasks=tasks,
    )

    source_paths = [tasks_path, tasks_manifest_path, first32_pool, plan_path, *completed_group_dirs, *assignment_paths.values(), *shard_group_dirs.values()]
    output_resolved = output_dir.resolve()
    for source in source_paths:
        resolved = source.resolve()
        _require(output_resolved != resolved and output_resolved not in resolved.parents and resolved not in output_resolved.parents, f"output/source paths overlap: {output_dir} / {source}")

    candidates: dict[str, list[SourceGroup]] = {task_id: [] for task_id in task_ids}
    for group in first_groups.values():
        candidates[group.task_id].append(group)
    completed_source_reports: list[dict[str, Any]] = []
    for source_index, directory in enumerate(completed_group_dirs):
        groups = _directory_groups(directory, f"completed source {source_index}")
        _require(set(groups) <= set(task_ids), f"completed source {source_index} contains unknown task ids")
        for group in groups.values():
            candidates[group.task_id].append(group)
        completed_source_reports.append({"path": str(directory.resolve()), "groups": len(groups)})

    shard_reports: list[dict[str, Any]] = []
    for shard in range(EXPECTED_SHARDS):
        directory = shard_group_dirs[shard]
        groups = _directory_groups(directory, f"crosshost shard {shard}")
        expected_ids = assignment_ids[shard]
        _require(set(groups) == set(expected_ids), f"crosshost shard {shard} groups do not exactly cover its assignment")
        for group in groups.values():
            candidates[group.task_id].append(group)
        shard_reports.append(
            {
                "shard": shard,
                "assignment_path": str(assignment_paths[shard].resolve()),
                "assignment_sha256": sha256_file(assignment_paths[shard]),
                "groups_path": str(directory.resolve()),
                "tasks": len(groups),
            }
        )

    chosen: dict[str, SourceGroup] = {}
    duplicate_reports: list[dict[str, Any]] = []
    for task_id in task_ids:
        values = candidates[task_id]
        _require(bool(values), f"missing K8 group for task {task_id}")
        reference = values[0]
        for other in values[1:]:
            _require(other.payload == reference.payload, f"duplicate task group is not byte-identical: {task_id}: {reference.path} / {other.path}")
        if len(values) > 1:
            duplicate_reports.append(
                {
                    "task_id": task_id,
                    "sha256": sha256_bytes(reference.payload),
                    "sources": [_source_record(value) for value in values],
                }
            )
        chosen[task_id] = reference

    completed_map = plan["completed"]["groups"]
    for task_id in completed_ids:
        expected_sha = completed_map[task_id]["sha256"]
        matching_completed_source = any(
            group.role == "first32 groups" or group.role.startswith("completed source")
            for group in candidates[task_id]
        )
        _require(matching_completed_source, f"plan completed group has no remapped completed source: {task_id}")
        _require(sha256_bytes(chosen[task_id].payload) == expected_sha, f"plan completed group SHA mismatch: {task_id}")
    _require(set(chosen) == set(task_ids), "merged task coverage is not exactly train600")
    _require(len(chosen) == EXPECTED_TASKS, "merged group count is not exactly 600")

    all_rows: list[dict[str, Any]] = []
    group_index: list[dict[str, Any]] = []
    for row_index, task_id in enumerate(task_ids):
        group = chosen[task_id]
        _validate_group(group, task=task_by_id[task_id], row_index=row_index)
        all_rows.extend(group.rows)
        group_index.append(
            {
                "row_index": row_index,
                "task_id": task_id,
                "example_index": task_by_id[task_id]["example_index"],
                "sha256": sha256_bytes(group.payload),
                "selected_source": _source_record(group),
                "candidate_sources": len(candidates[task_id]),
            }
        )
    _require(len(all_rows) == EXPECTED_TASKS * EXPECTED_GROUP_SIZE, "merged trajectory count is not exactly 4800")
    _require([row["sequence"] for row in all_rows] == list(range(EXPECTED_TASKS * EXPECTED_GROUP_SIZE)), "merged row_index/sequence topology is not contiguous")
    trajectories_payload = _compact_jsonl(all_rows)
    trajectories_sha = sha256_bytes(trajectories_payload)
    correct = sum(bool(row["sample"]["correct"]) for row in all_rows)

    manifest = {
        "schema_version": PENDING_MANIFEST_SCHEMA_VERSION,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "protocol_runtime_root": EXPECTED_PROTOCOL_RUNTIME_ROOT,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "model_path": EXPECTED_MODEL_PATH,
        "adapter_path": EXPECTED_ADAPTER_PATH,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "tasks": EXPECTED_TASKS,
        "group_size": EXPECTED_GROUP_SIZE,
        "trajectories": EXPECTED_TASKS * EXPECTED_GROUP_SIZE,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": EXPECTED_SEED,
        "generation_task_batch_size": first_manifest.get("generation_task_batch_size", 1),
        "generation_scheduler": "dynamic",
        "generation_question_window": 4,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "tool_schema_sha256": first_manifest.get("tool_schema_sha256"),
        "trajectories_sha256": trajectories_sha,
        "correct_trajectories": correct,
        "crosshost_merge": {
            "schema_version": SCHEMA_VERSION,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": expected_plan_sha256,
            "plan_completed_tasks": len(completed_ids),
            "plan_remaining_tasks": len(assigned_ids),
            "first32_manifest_sha256": expected_first32_manifest_sha256,
            "first32_trajectories_sha256": expected_first32_trajectories_sha256,
            "duplicate_groups_byte_identical": len(duplicate_reports),
            "publication": "whole-directory-atomic-rename",
        },
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_and_atomically_published",
        "contract": {
            "tasks": EXPECTED_TASKS,
            "group_size": EXPECTED_GROUP_SIZE,
            "trajectories": EXPECTED_TASKS * EXPECTED_GROUP_SIZE,
            "row_index_definition": "frozen task position * 8 + sample_index; serialized as sequence",
            "seed": EXPECTED_SEED,
            "generation_seed_scheme": "sha256-task-sample-turn-v1",
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "model_path": EXPECTED_MODEL_PATH,
            "adapter_path": EXPECTED_ADAPTER_PATH,
            "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        },
        "inputs": {
            "tasks": {"path": str(tasks_path.resolve()), "sha256": EXPECTED_TASKS_SHA256},
            "tasks_manifest": {"path": str(tasks_manifest_path.resolve()), "sha256": EXPECTED_TASKS_MANIFEST_SHA256},
            "first32_pool": str(first32_pool.resolve()),
            "continuation_plan": {"path": str(plan_path.resolve()), "sha256": expected_plan_sha256},
            "completed_group_sources": completed_source_reports,
            "crosshost_shards": shard_reports,
        },
        "observed": {
            "groups": len(chosen),
            "trajectories": len(all_rows),
            "correct_trajectories": correct,
            "duplicate_task_groups": len(duplicate_reports),
            "plan_completed_tasks": len(completed_ids),
            "plan_remaining_tasks": len(assigned_ids),
        },
        "outputs": {
            "trajectories_sha256": trajectories_sha,
            "manifest_sha256": sha256_bytes(manifest_payload),
        },
        "duplicates": duplicate_reports,
        "groups": group_index,
        "source_task_manifest_summary": {
            "schema_version": tasks_manifest.get("schema_version"),
            "selection_seed": (tasks_manifest.get("selection") or {}).get("seed"),
        },
    }
    audit_payload = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.next-", dir=output_dir.parent))
    try:
        groups_output = temporary / "groups"
        groups_output.mkdir()
        for task_id in task_ids:
            _write_file(groups_output / f"{task_id}.json", chosen[task_id].payload)
        _write_file(temporary / "trajectories.jsonl", trajectories_payload)
        _write_file(temporary / "manifest.pending.json", manifest_payload)
        _write_file(temporary / "crosshost_merge_audit.json", audit_payload)
        _require(len(list(groups_output.iterdir())) == EXPECTED_TASKS, "staging group publication count mismatch")
        _require(sha256_file(temporary / "trajectories.jsonl") == trajectories_sha, "staging trajectories changed during write")
        _require(sha256_file(temporary / "manifest.pending.json") == audit["outputs"]["manifest_sha256"], "staging manifest changed during write")
        _require(
            not output_dir.exists() and not output_dir.is_symlink(),
            f"output directory appeared before atomic publication: {output_dir}",
        )
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "verified_and_atomically_published",
        "output_dir": str(output_dir),
        "groups": EXPECTED_TASKS,
        "trajectories": EXPECTED_TASKS * EXPECTED_GROUP_SIZE,
        "trajectories_sha256": trajectories_sha,
        "manifest_sha256": audit["outputs"]["manifest_sha256"],
        "audit_sha256": sha256_file(output_dir / "crosshost_merge_audit.json"),
        "duplicate_task_groups": len(duplicate_reports),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--first32-pool", type=Path, required=True)
    parser.add_argument("--expected-first32-manifest-sha256", required=True)
    parser.add_argument("--expected-first32-trajectories-sha256", required=True)
    parser.add_argument("--completed-group-dir", type=Path, action="append", required=True)
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--assignment", action="append", required=True, help="SHARD=/absolute/path")
    parser.add_argument("--shard-group-dir", action="append", required=True, help="SHARD=/absolute/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge(
            tasks_path=args.tasks,
            tasks_manifest_path=args.tasks_manifest,
            first32_pool=args.first32_pool,
            expected_first32_manifest_sha256=args.expected_first32_manifest_sha256,
            expected_first32_trajectories_sha256=args.expected_first32_trajectories_sha256,
            completed_group_dirs=args.completed_group_dir,
            plan_path=args.continuation_plan,
            expected_plan_sha256=args.expected_plan_sha256,
            assignment_paths=_parse_named_paths(args.assignment, "assignment"),
            shard_group_dirs=_parse_named_paths(args.shard_group_dir, "shard-group-dir"),
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"crosshost boundary merge blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
