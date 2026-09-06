#!/usr/bin/env python3
"""Audit one frozen K=8 initial-policy screen for boundary-cohort GRPO.

The audit deliberately separates structural validity from selection readiness.  A
runtime-contaminated K group is retained for audit but is never a boundary
candidate; an otherwise valid S1 audit may therefore direct the preregistered S2
screen instead of silently retrying individual trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "vanilla-grpo-boundary-screen-audit-v1"
PENDING_MANIFEST_SCHEMA = "table-agent-fixed-rollout-pool-pending-v1"
TRAJECTORY_SCHEMA = "table-agent-fixed-policy-episode-v1"
EXPECTED_TASKS = 600
EXPECTED_GROUP_SIZE = 8
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
MIN_MIXED_GROUPS = 360
MIN_CORE_GROUPS = 240
RUNTIME_FAILURE_TYPES = frozenset(
    {"generation_oom", "generation_length", "context_overflow", "timeout_error"}
)
TIMEOUT_TYPES = frozenset({"timeout_error"})
TIMEOUT_CODES = frozenset({"tool_execution_timeout"})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task row is missing example_id/instance_id")
    return value


def _parse_jsonl(data: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _policy_evidence_valid(policy_turns: Any) -> bool:
    if not isinstance(policy_turns, list) or not policy_turns:
        return False
    for turn in policy_turns:
        if not isinstance(turn, dict):
            return False
        prompt = turn.get("prompt_ids")
        response = turn.get("response_ids")
        logprobs = turn.get("sampling_logprobs")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(response, list)
            or not response
            or not isinstance(logprobs, list)
            or len(response) != len(logprobs)
            or not all(type(token) is int for token in prompt + response)
            or not all(_finite_number(value) for value in logprobs)
        ):
            return False
    return True


def _timeout_marker(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("failure_type") in TIMEOUT_TYPES
        or value.get("error_type") in TIMEOUT_TYPES
        or value.get("execution_error_type") in TIMEOUT_TYPES
        or value.get("recovered_from_error_type") in TIMEOUT_TYPES
        or value.get("error_code") in TIMEOUT_CODES
        or value.get("code") in TIMEOUT_CODES
    )


def _has_timeout_evidence(record: Mapping[str, Any]) -> bool:
    if record.get("failure_type") in TIMEOUT_TYPES:
        return True
    if any(_timeout_marker(event) for event in (record.get("error_events") or [])):
        return True
    for turn in record.get("turns") or []:
        if isinstance(turn, dict) and (
            _timeout_marker(turn) or _timeout_marker(turn.get("error_event"))
        ):
            return True
    return False


def _runtime_contamination(
    sample: Mapping[str, Any], record: Mapping[str, Any]
) -> list[str]:
    reasons: set[str] = set()
    for value in (sample.get("failure_type"), record.get("failure_type")):
        if value in RUNTIME_FAILURE_TYPES:
            reasons.add(str(value))
    if record.get("generation_truncation") is not None:
        reasons.add("generation_length")
    if _has_timeout_evidence(record):
        reasons.add("timeout_evidence")
    if sample.get("process_update") is not True:
        reasons.add("process_update_false")
    if record.get("optimization_exclusion") is not None:
        reasons.add("optimization_exclusion")
    return sorted(reasons)


def audit(
    manifest: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    tasks: Sequence[dict[str, Any]],
    *,
    manifest_sha256: str,
    trajectories_sha256: str,
    tasks_sha256: str,
    tasks_path: str | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    issue_counts: Counter[str] = Counter()

    def issue(code: str, location: str, detail: str) -> None:
        issue_counts[code] += 1
        issues.append({"code": code, "location": location, "detail": detail})

    expected_manifest = {
        "schema_version": PENDING_MANIFEST_SCHEMA,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_sha256": tasks_sha256,
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
        "denotation_comparison": "bird-set",
        "seed": 20260812,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "trajectories_sha256": trajectories_sha256,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            issue(
                "manifest_contract_mismatch",
                f"manifest.{key}",
                f"observed {manifest.get(key)!r}; expected {expected!r}",
            )
    manifest_tasks_path = manifest.get("tasks_path")
    if not isinstance(manifest_tasks_path, str) or not manifest_tasks_path:
        issue("manifest_contract_mismatch", "manifest.tasks_path", "missing tasks path")
    elif tasks_path is not None and (
        Path(manifest_tasks_path).resolve() != Path(tasks_path).resolve()
    ):
        issue(
            "manifest_contract_mismatch",
            "manifest.tasks_path",
            f"observed {manifest_tasks_path!r}; audited {tasks_path!r}",
        )

    task_ids: list[str] = []
    task_examples: dict[str, int] = {}
    tasks_by_id: dict[str, dict[str, Any]] = {}
    try:
        for position, task in enumerate(tasks):
            task_id = _task_id(task)
            example_index = task.get("example_index")
            if type(example_index) is not int:
                raise ValueError(f"task {task_id} has invalid example_index")
            if task_id in task_examples:
                raise ValueError(f"duplicate task id {task_id}")
            if example_index in task_examples.values():
                raise ValueError(f"duplicate example_index {example_index}")
            if not isinstance(task.get("db_id"), str) or not isinstance(
                task.get("question"), str
            ):
                raise ValueError(f"task {task_id} lacks public selection fields")
            task_ids.append(task_id)
            task_examples[task_id] = example_index
            tasks_by_id[task_id] = task
    except ValueError as exc:
        issue("task_contract_invalid", "tasks", str(exc))
    if len(tasks) != EXPECTED_TASKS:
        issue(
            "task_count_mismatch",
            "tasks",
            f"observed {len(tasks)}; expected {EXPECTED_TASKS}",
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_total = 0
    for position, row in enumerate(rows):
        location = f"trajectories[{position}]"
        if row.get("schema_version") != TRAJECTORY_SCHEMA:
            issue("trajectory_schema_mismatch", location, "unsupported schema")
        if type(row.get("sequence")) is not int or row.get("sequence") != position:
            issue("trajectory_sequence_mismatch", location, "sequence is not contiguous")
        environment = row.get("environment")
        sample = row.get("sample")
        if not isinstance(environment, dict) or not isinstance(sample, dict):
            issue("trajectory_shape_invalid", location, "missing environment/sample")
            continue
        record = sample.get("audit_record")
        if not isinstance(record, dict):
            issue("trajectory_shape_invalid", location, "missing sample.audit_record")
            continue
        task_id = environment.get("task_id")
        example_index = environment.get("example_index")
        sample_index = record.get("sample_index")
        expected_trajectory_id = (
            f"rl_{example_index}_sample_{sample_index}"
            if type(example_index) is int and type(sample_index) is int
            else None
        )
        if (
            not isinstance(task_id, str)
            or task_id not in task_examples
            or type(example_index) is not int
            or task_examples.get(task_id) != example_index
            or record.get("example_index") != example_index
            or record.get("trajectory_id") != expected_trajectory_id
            or type(sample_index) is not int
        ):
            issue("trajectory_identity_invalid", location, "task/example/sample mismatch")
            continue
        expected_task = tasks_by_id[task_id]
        expected_environment = {
            "dataset_split": "train",
            "example_index": example_index,
            "task_id": task_id,
            "db_id": expected_task.get("db_id"),
            "db_path": expected_task.get("db_path"),
            "question": expected_task.get("question"),
            "gold_sql": expected_task.get("gold_sql") or expected_task.get("query"),
            "external_knowledge": expected_task.get("external_knowledge"),
        }
        if any(
            environment.get(key) != value
            for key, value in expected_environment.items()
        ):
            issue(
                "trajectory_environment_mismatch",
                location,
                "environment differs from the byte-bound task record",
            )
        if (
            record.get("protocol_version") != EXPECTED_PROTOCOL_VERSION
            or record.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
        ):
            issue("trajectory_protocol_mismatch", location, "wrong protocol identity")
        optimizer_evidence = {
            key: record[key]
            for key in ("policy_global_step", "policy_micro_step", "optimizer_step", "global_step")
            if key in record
        }
        if any(type(value) is not int or value != 0 for value in optimizer_evidence.values()):
            issue("optimizer_update_evidence", location, repr(optimizer_evidence))
        if sample.get("step_rewards") is not None:
            issue("nonbinary_reward_contract", location, "step_rewards must be null")
        if sample.get("failure_type") != record.get("failure_type"):
            issue("failure_identity_mismatch", location, "sample/audit failure differs")

        contamination = _runtime_contamination(sample, record)
        policy_turns = row.get("policy_turns")
        # A first-turn context overflow or generation OOM has no authored policy
        # action, hence no token/logprob vector.  It is valid runtime evidence
        # only for excluding the entire K group.  Every clean or authored row
        # still requires complete aligned policy evidence.
        empty_runtime_evidence = (
            policy_turns == []
            and sample.get("process_update") is False
            and sample.get("failure_type")
            in {"generation_oom", "context_overflow"}
        )
        if not empty_runtime_evidence and not _policy_evidence_valid(policy_turns):
            issue("policy_evidence_invalid", location, "invalid token/logprob evidence")
        correct = sample.get("correct")
        reward = sample.get("reward")
        result_reward = record.get("result_reward")
        # New scorers zero excluded runtime trajectories; the already-running S1
        # screen may have been generated by the preceding scorer, which retained
        # binary correctness for a recovered timeout.  Both shapes are auditable:
        # structured runtime evidence still invalidates the whole K group below.
        expected_reward = (
            0.0 if sample.get("process_update") is False else float(bool(correct))
        )
        reward_valid = (
            isinstance(correct, bool)
            and record.get("correct") is correct
            and _finite_number(reward)
            and float(reward) == expected_reward
            and isinstance(result_reward, dict)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and _finite_number(result_reward.get("value"))
            and float(result_reward["value"]) == expected_reward
        )
        if not reward_valid:
            issue("nonbinary_reward_contract", location, "invalid binary result reward")
        correct_total += int(correct is True)
        groups[str(task_id)].append(
            {
                "sample_index": sample_index,
                "example_index": example_index,
                "correct": correct is True,
                "contamination": contamination,
            }
        )

    if len(rows) != EXPECTED_TASKS * EXPECTED_GROUP_SIZE:
        issue(
            "trajectory_count_mismatch",
            "trajectories",
            f"observed {len(rows)}; expected {EXPECTED_TASKS * EXPECTED_GROUP_SIZE}",
        )
    if manifest.get("correct_trajectories") != correct_total:
        issue("correct_count_mismatch", "manifest.correct_trajectories", str(correct_total))

    summaries: list[dict[str, Any]] = []
    observed_order: list[str] = []
    for task_id in task_ids:
        entries = groups.get(task_id, [])
        if entries:
            observed_order.append(task_id)
        structure_valid = (
            len(entries) == EXPECTED_GROUP_SIZE
            and sorted(entry["sample_index"] for entry in entries)
            == list(range(EXPECTED_GROUP_SIZE))
            and {entry["example_index"] for entry in entries}
            == {task_examples.get(task_id)}
        )
        if not structure_valid:
            issue("group_structure_invalid", f"groups.{task_id}", "not one exact K8 group")
        contamination = sorted(
            {reason for entry in entries for reason in entry["contamination"]}
        )
        usable = structure_valid and not contamination
        correct_count = sum(entry["correct"] for entry in entries)
        mixed = usable and 1 <= correct_count <= 7
        core = usable and 2 <= correct_count <= 6
        summaries.append(
            {
                "task_id": task_id,
                "example_index": task_examples.get(task_id),
                "trajectories": len(entries),
                "correct_count": correct_count,
                "uncertainty": correct_count * (8 - correct_count) if mixed else 0,
                "usable": usable,
                "mixed_boundary": mixed,
                "core_boundary": core,
                "contamination": contamination,
            }
        )
    extra_groups = sorted(set(groups) - set(task_ids))
    if extra_groups:
        issue("unknown_groups", "groups", repr(extra_groups[:5]))
    first_seen = []
    seen: set[str] = set()
    for row in rows:
        value = ((row.get("environment") or {}).get("task_id"))
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            first_seen.append(value)
    if first_seen != task_ids:
        issue("group_order_mismatch", "groups", "trajectory groups do not follow task order")

    structural_passes = not issues
    mixed_count = sum(summary["mixed_boundary"] for summary in summaries)
    core_count = sum(summary["core_boundary"] for summary in summaries)
    local_ready = (
        structural_passes
        and mixed_count >= MIN_MIXED_GROUPS
        and core_count >= MIN_CORE_GROUPS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "tasks": EXPECTED_TASKS,
            "group_size": EXPECTED_GROUP_SIZE,
            "optimizer_updates": 0,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "screen_seed": 20260812,
            "generation_seed_scheme": "sha256-task-sample-turn-v1",
            "minimum_mixed_groups": MIN_MIXED_GROUPS,
            "minimum_core_groups": MIN_CORE_GROUPS,
        },
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "trajectories_sha256": trajectories_sha256,
            "tasks_sha256": tasks_sha256,
        },
        "observed": {
            "tasks": len(tasks),
            "trajectories": len(rows),
            "usable_groups": sum(summary["usable"] for summary in summaries),
            "contaminated_groups": sum(not summary["usable"] for summary in summaries),
            "mixed_boundary_groups": mixed_count,
            "core_boundary_groups": core_count,
            "homogeneous_groups": sum(
                summary["usable"] and not summary["mixed_boundary"] for summary in summaries
            ),
            "correct_trajectories": correct_total,
        },
        "groups": summaries,
        "issues": issues,
        "issue_counts": dict(sorted(issue_counts.items())),
        "status": {
            "audit_passes": structural_passes,
            "pool_admitted": structural_passes,
            "selection_ready_without_s2": local_ready,
            "next_stage": "select_s1" if local_ready else "screen_s2",
        },
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir / "boundary_screen_audit.json"
    if output.exists() and not args.overwrite:
        print(f"output exists; pass --overwrite: {output}", file=sys.stderr)
        return 2
    if output.exists():
        output.unlink()
    try:
        manifest_bytes = args.manifest.read_bytes()
        trajectory_bytes = args.trajectories.read_bytes()
        task_bytes = args.tasks.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        rows = _parse_jsonl(trajectory_bytes, path=args.trajectories)
        tasks = _parse_jsonl(task_bytes, path=args.tasks)
        result = audit(
            manifest,
            rows,
            tasks,
            manifest_sha256=_sha256(manifest_bytes),
            trajectories_sha256=_sha256(trajectory_bytes),
            tasks_sha256=_sha256(task_bytes),
            tasks_path=str(args.tasks.resolve()),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"boundary screen input error: {exc}", file=sys.stderr)
        return 2
    if not result["status"]["audit_passes"]:
        print(json.dumps({"status": result["status"], "issues": result["issue_counts"]}), file=sys.stderr)
        return 1
    result["inputs"].update(
        {
            "manifest": str(args.manifest.resolve()),
            "trajectories": str(args.trajectories.resolve()),
            "tasks": str(args.tasks.resolve()),
        }
    )
    _write_atomic(output, result)
    print(json.dumps({"output": str(output.resolve()), **result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
