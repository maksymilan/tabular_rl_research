#!/usr/bin/env python3
"""Fail-closed audit for the frozen no-update vanilla-GRPO rollout probe."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rl.diagnostics.io import (
    parse_jsonl_bytes as _diagnostic_parse_jsonl_bytes,
    read_json as _diagnostic_read_json,
    read_jsonl as _diagnostic_read_jsonl,
    sha256_bytes as _diagnostic_sha256_bytes,
    sha256_file as _diagnostic_sha256_file,
)
from rl.diagnostics.validation import finite_number as _diagnostic_finite_number


SCHEMA_VERSION = "vanilla-grpo-rollout-probe-audit-v1"
PENDING_MANIFEST_SCHEMA = "table-agent-fixed-rollout-pool-pending-v1"
TRAJECTORY_SCHEMA = "table-agent-fixed-policy-episode-v1"
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
EXPECTED_TASKS = 32
EXPECTED_GROUP_SIZE = 8
EXPECTED_TRAJECTORIES = EXPECTED_TASKS * EXPECTED_GROUP_SIZE
MIN_ELIGIBLE_FRACTION = 0.95
MIN_MIXED_OUTCOME_GROUPS = 20

_TIMEOUT_MARKER_KEYS = frozenset(
    {
        "code",
        "error_type",
        "execution_error",
        "execution_error_type",
        "failure_type",
        "message",
    }
)


def sha256_file(path: Path) -> str:
    """Compatibility export backed by :mod:`rl.diagnostics.io`."""
    return _diagnostic_sha256_file(path)


def load_json(path: Path) -> dict[str, Any]:
    return dict(_diagnostic_read_json(path, require_object=True))


def _parse_jsonl(data: bytes, *, path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _diagnostic_parse_jsonl_bytes(data, source=path)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _diagnostic_read_jsonl(path)]


def _contains_timeout_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                str(key).lower() in _TIMEOUT_MARKER_KEYS
                and isinstance(nested, str)
                and "timeout" in nested.lower()
            ):
                return True
            if isinstance(nested, (dict, list)) and _contains_timeout_marker(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_timeout_marker(item) for item in value)
    return False


def _is_finite_number(value: Any) -> bool:
    return _diagnostic_finite_number(value)


def _policy_evidence_is_valid(policy_turns: Any) -> bool:
    if not isinstance(policy_turns, list) or not policy_turns:
        return False
    for turn in policy_turns:
        if not isinstance(turn, dict):
            return False
        prompt_ids = turn.get("prompt_ids")
        response_ids = turn.get("response_ids")
        logprobs = turn.get("sampling_logprobs")
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or not isinstance(response_ids, list)
            or not response_ids
            or not isinstance(logprobs, list)
            or len(response_ids) != len(logprobs)
            or not all(type(token) is int for token in prompt_ids + response_ids)
            or not all(_is_finite_number(value) for value in logprobs)
        ):
            return False
    return True


def audit(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    trajectories_sha256: str,
) -> dict[str, Any]:
    """Audit identity, binary rewards, K=8 topology, and probe eligibility."""
    issues: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()

    def add_issue(code: str, location: str, detail: str) -> None:
        issue_counts[code] += 1
        issues.append({"code": code, "location": location, "detail": detail})

    expected_manifest = {
        "schema_version": PENDING_MANIFEST_SCHEMA,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": (
            EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256
        ),
        "tasks": EXPECTED_TASKS,
        "group_size": EXPECTED_GROUP_SIZE,
        "trajectories": EXPECTED_TRAJECTORIES,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            add_issue(
                "manifest_contract_mismatch",
                f"manifest.{key}",
                f"observed {manifest.get(key)!r}; expected {expected!r}",
            )
    if manifest.get("trajectories_sha256") != trajectories_sha256:
        add_issue(
            "trajectory_digest_mismatch",
            "manifest.trajectories_sha256",
            (
                f"observed {manifest.get('trajectories_sha256')!r}; "
                f"actual {trajectories_sha256!r}"
            ),
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_trajectories = 0
    correct_trajectories = 0
    forbidden_counts: Counter[str] = Counter()
    row_contract_failures = 0
    binary_reward_failures = 0
    protocol_identity_failures = 0
    policy_evidence_failures = 0
    optimizer_evidence_failures = 0

    for position, row in enumerate(rows):
        location = f"trajectories[{position}]"
        row_failed = False

        def row_issue(code: str, detail: str) -> None:
            nonlocal row_failed
            row_failed = True
            add_issue(code, location, detail)

        if row.get("schema_version") != TRAJECTORY_SCHEMA:
            row_issue(
                "trajectory_schema_mismatch",
                f"unsupported schema {row.get('schema_version')!r}",
            )
        if type(row.get("sequence")) is not int or row.get("sequence") != position:
            row_issue(
                "trajectory_sequence_mismatch",
                f"observed sequence {row.get('sequence')!r}; expected {position}",
            )

        environment = row.get("environment")
        sample = row.get("sample")
        if not isinstance(environment, dict) or not isinstance(sample, dict):
            row_issue(
                "trajectory_shape_invalid",
                "environment and sample must both be JSON objects",
            )
            row_contract_failures += 1
            continue
        audit_record = sample.get("audit_record")
        if not isinstance(audit_record, dict):
            row_issue(
                "trajectory_shape_invalid",
                "sample.audit_record must be a JSON object",
            )
            row_contract_failures += 1
            continue

        task_id = environment.get("task_id")
        example_index = audit_record.get("example_index")
        sample_index = audit_record.get("sample_index")
        if not isinstance(task_id, str) or not task_id:
            row_issue("task_identity_invalid", "environment.task_id is missing")
        elif type(example_index) is not int or type(sample_index) is not int:
            row_issue(
                "sample_identity_invalid",
                "audit example_index/sample_index must be integers",
            )
        else:
            environment_example_index = environment.get("example_index")
            if (
                type(environment_example_index) is not int
                or environment_example_index != example_index
            ):
                row_issue(
                    "sample_identity_invalid",
                    "environment and audit example_index do not match",
                )
            groups[task_id].append(
                {
                    "position": position,
                    "example_index": example_index,
                    "sample_index": sample_index,
                    "eligible": sample.get("process_update") is True,
                    "correct": sample.get("correct"),
                    "reward": sample.get("reward"),
                }
            )

        if (
            audit_record.get("protocol_version") != EXPECTED_PROTOCOL_VERSION
            or audit_record.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
        ):
            protocol_identity_failures += 1
            row_issue(
                "trajectory_protocol_mismatch",
                "trajectory protocol identity differs from frozen version26",
            )

        optimizer_evidence = {
            key: audit_record.get(key)
            for key in (
                "policy_global_step",
                "policy_micro_step",
                "optimizer_step",
                "global_step",
            )
            if key in audit_record
        }
        if any(
            type(value) is not int or value != 0
            for value in optimizer_evidence.values()
        ):
            optimizer_evidence_failures += 1
            row_issue(
                "optimizer_update_evidence",
                f"no-update probe contains nonzero/invalid step metadata {optimizer_evidence}",
            )

        correct = sample.get("correct")
        reward = sample.get("reward")
        result_reward = audit_record.get("result_reward")
        binary_reward_valid = (
            isinstance(correct, bool)
            and _is_finite_number(reward)
            and float(reward) == float(int(correct))
            and sample.get("step_rewards") is None
            and not audit_record.get("process_reward")
            and isinstance(result_reward, dict)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and _is_finite_number(result_reward.get("value"))
            and float(result_reward["value"]) == float(reward)
        )
        if not binary_reward_valid:
            binary_reward_failures += 1
            row_issue(
                "binary_reward_contract_invalid",
                "reward must be explicit binary terminal correctness only",
            )
        elif correct:
            correct_trajectories += 1

        if sample.get("process_update") is True:
            eligible_trajectories += 1
        elif not isinstance(sample.get("process_update"), bool):
            row_issue(
                "eligibility_marker_invalid",
                "sample.process_update must be an explicit boolean",
            )

        sample_failure = sample.get("failure_type")
        audit_failure = audit_record.get("failure_type")
        if sample_failure != audit_failure:
            row_issue(
                "failure_identity_mismatch",
                "sample.failure_type and audit_record.failure_type differ",
            )
        if sample_failure == "generation_length" or audit_record.get(
            "generation_truncation"
        ):
            forbidden_counts["generation_length"] += 1
            row_issue(
                "forbidden_generation_length",
                "length-truncated completion is not probe-eligible",
            )
        if sample_failure == "context_overflow":
            forbidden_counts["context_overflow"] += 1
            row_issue(
                "forbidden_context_overflow",
                "context-overflow trajectory is not probe-eligible",
            )
        if _contains_timeout_marker(
            {
                "failure_type": sample_failure,
                "turns": audit_record.get("turns"),
                "error_events": audit_record.get("error_events"),
            }
        ):
            forbidden_counts["timeout"] += 1
            row_issue(
                "forbidden_timeout",
                "trajectory contains a timeout failure or timeout turn",
            )
        if audit_record.get("optimization_exclusion") is not None:
            forbidden_counts["optimization_exclusion"] += 1
            row_issue(
                "forbidden_optimization_exclusion",
                f"observed {audit_record.get('optimization_exclusion')!r}",
            )

        if not _policy_evidence_is_valid(row.get("policy_turns")):
            policy_evidence_failures += 1
            row_issue(
                "policy_evidence_invalid",
                "policy turns must retain aligned nonempty token/logprob vectors",
            )
        if row_failed:
            row_contract_failures += 1

    if len(rows) != EXPECTED_TRAJECTORIES:
        add_issue(
            "trajectory_count_mismatch",
            "trajectories",
            f"observed {len(rows)} rows; expected {EXPECTED_TRAJECTORIES}",
        )
    if manifest.get("correct_trajectories") != correct_trajectories:
        add_issue(
            "correct_count_mismatch",
            "manifest.correct_trajectories",
            (
                f"observed {manifest.get('correct_trajectories')!r}; "
                f"recomputed {correct_trajectories}"
            ),
        )

    group_summaries: list[dict[str, Any]] = []
    mixed_outcome_groups = 0
    group_structure_failures = 0
    example_to_task: dict[int, str] = {}
    for task_id, entries in sorted(groups.items()):
        sample_indices = sorted(entry["sample_index"] for entry in entries)
        example_indices = {entry["example_index"] for entry in entries}
        structure_valid = (
            len(entries) == EXPECTED_GROUP_SIZE
            and sample_indices == list(range(EXPECTED_GROUP_SIZE))
            and len(example_indices) == 1
        )
        if structure_valid:
            example_index = next(iter(example_indices))
            previous_task = example_to_task.setdefault(example_index, task_id)
            if previous_task != task_id:
                structure_valid = False
        if not structure_valid:
            group_structure_failures += 1
            add_issue(
                "group_structure_invalid",
                f"groups.{task_id}",
                "group must contain one example and exact sample indices 0..7",
            )
        eligible_rewards = {
            int(bool(entry["correct"]))
            for entry in entries
            if entry["eligible"] and isinstance(entry["correct"], bool)
        }
        mixed = structure_valid and eligible_rewards == {0, 1}
        mixed_outcome_groups += int(mixed)
        group_summaries.append(
            {
                "task_id": task_id,
                "example_indices": sorted(example_indices),
                "trajectories": len(entries),
                "eligible": sum(bool(entry["eligible"]) for entry in entries),
                "correct": sum(entry["correct"] is True for entry in entries),
                "sample_indices": sample_indices,
                "mixed_eligible_outcomes": mixed,
                "structure_valid": structure_valid,
            }
        )
    if len(groups) != EXPECTED_TASKS:
        add_issue(
            "group_count_mismatch",
            "groups",
            f"observed {len(groups)} groups; expected {EXPECTED_TASKS}",
        )

    eligible_fraction = eligible_trajectories / EXPECTED_TRAJECTORIES
    checks = {
        "manifest_contract": not any(
            code.startswith("manifest_")
            for code in issue_counts
        ),
        "trajectory_digest_bound": issue_counts["trajectory_digest_mismatch"] == 0,
        "exact_trajectory_count": len(rows) == EXPECTED_TRAJECTORIES,
        "contiguous_fixed_pool_rows": (
            issue_counts["trajectory_schema_mismatch"] == 0
            and issue_counts["trajectory_sequence_mismatch"] == 0
        ),
        "trajectory_rows_valid": row_contract_failures == 0,
        "trajectory_protocol_identity": protocol_identity_failures == 0,
        "no_optimizer_update_evidence": optimizer_evidence_failures == 0,
        "binary_result_reward_only": binary_reward_failures == 0,
        "policy_evidence_complete": policy_evidence_failures == 0,
        "exact_k8_groups": (
            len(groups) == EXPECTED_TASKS and group_structure_failures == 0
        ),
        "no_generation_length": forbidden_counts["generation_length"] == 0,
        "no_context_overflow": forbidden_counts["context_overflow"] == 0,
        "no_timeout": forbidden_counts["timeout"] == 0,
        "no_optimization_exclusion": (
            forbidden_counts["optimization_exclusion"] == 0
        ),
        "eligible_at_least_95_percent": (
            eligible_fraction >= MIN_ELIGIBLE_FRACTION
        ),
        "at_least_20_mixed_outcome_groups": (
            mixed_outcome_groups >= MIN_MIXED_OUTCOME_GROUPS
        ),
        "manifest_correct_count_reconciles": (
            issue_counts["correct_count_mismatch"] == 0
        ),
    }
    passes = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "tasks": EXPECTED_TASKS,
            "group_size": EXPECTED_GROUP_SIZE,
            "trajectories": EXPECTED_TRAJECTORIES,
            "minimum_eligible_fraction": MIN_ELIGIBLE_FRACTION,
            "minimum_mixed_outcome_groups": MIN_MIXED_OUTCOME_GROUPS,
            "optimizer_updates": 0,
        },
        "observed": {
            "tasks": len(groups),
            "trajectories": len(rows),
            "eligible_trajectories": eligible_trajectories,
            "eligible_fraction": eligible_fraction,
            "correct_trajectories": correct_trajectories,
            "mixed_outcome_groups": mixed_outcome_groups,
            "row_contract_failures": row_contract_failures,
            "forbidden_counts": dict(sorted(forbidden_counts.items())),
            "trajectories_sha256": trajectories_sha256,
        },
        "checks": checks,
        "groups": group_summaries,
        "issues": issues,
        "issue_counts": dict(sorted(issue_counts.items())),
        "status": {
            "passes": passes,
            "probe_admitted": passes,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a frozen no-update Qwen3 version26 vanilla-GRPO rollout probe."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        print(
            f"output exists; pass --overwrite: {args.output}",
            file=sys.stderr,
        )
        return 2
    if args.output.exists():
        # ``--overwrite`` is also an explicit stale-admission revocation.  A
        # failed re-audit must never leave an older passing JSON at the path a
        # launcher is about to consume.
        try:
            args.output.unlink()
        except OSError as exc:
            print(f"cannot revoke stale output {args.output}: {exc}", file=sys.stderr)
            return 2
    try:
        # Audit immutable byte snapshots, then publish those exact digests.  This
        # avoids binding a hash from one file revision to rows parsed from another
        # if a generator is still writing concurrently.
        manifest_bytes = args.manifest.read_bytes()
        trajectories_bytes = args.trajectories.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest must be a JSON object: {args.manifest}")
        rows = _parse_jsonl(trajectories_bytes, path=args.trajectories)
        manifest_sha256 = _diagnostic_sha256_bytes(manifest_bytes)
        trajectories_sha256 = _diagnostic_sha256_bytes(trajectories_bytes)
        result = audit(
            manifest,
            rows,
            trajectories_sha256=trajectories_sha256,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"rollout probe audit input error: {exc}", file=sys.stderr)
        return 2

    if not result["status"]["passes"]:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checks": result["checks"],
                    "issue_counts": result["issue_counts"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    result["inputs"] = {
        "manifest": str(args.manifest.resolve()),
        "trajectories": str(args.trajectories.resolve()),
        "manifest_sha256": manifest_sha256,
        "trajectories_sha256": trajectories_sha256,
    }
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                **result["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
