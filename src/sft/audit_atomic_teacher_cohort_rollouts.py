#!/usr/bin/env python3
"""Audit causal atomic/native-bundle rollouts against a frozen sampling cohort.

The hidden SQL profile used by the cohort selector is an aggregate sampling target only.  This
audit measures the teacher's *actual* calls after model<->harness interaction and separately
reports selection coverage, verified-success attrition, public-distribution drift, exploration
tool use, execution errors, and protocol/training-admission state.

It does not export SFT records or promote a diagnostic protocol.  In particular, native tool
bundles remain grouped provider turns and are never flattened into synthetic assistant turns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "harness"))

from select_bird_train_baseline import (  # noqa: E402
    distribution_audit,
    length_cutpoints,
    task_id,
)
from sql_atomic_tool_profile import (  # noqa: E402
    NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS,
    SQL_IDENTIFIABLE_ATOMIC_TOOLS,
)
from training_result_quality import (  # noqa: E402
    EMPTY_RESULT_POLICY_VERSION,
    empty_result_target_reason,
    trajectory_has_empty_terminal_evidence,
)


AUDIT_VERSION = "atomic-teacher-cohort-rollout-audit-v1"
SUPPORTED_TOOL_SCHEMES = {"atomic", "native-tool-bundle"}
ELIGIBLE_TRAINING_ADMISSION = "eligible_by_current_context_contract"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def trajectory_task_id(record: Mapping[str, Any]) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    value = (
        source.get("example_id")
        or record.get("example_id")
        or record.get("trajectory_id")
    )
    if not isinstance(value, str) or not value:
        raise ValueError("rollout record has no example_id/trajectory_id")
    return value


def index_unique(
    records: Sequence[dict[str, Any]],
    *,
    id_fn,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = id_fn(record)
        if record_id in indexed:
            raise ValueError(f"{source}: duplicate id {record_id}")
        indexed[record_id] = record
    return indexed


def final_attempts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finals: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = trajectory_task_id(record)
        previous = finals.get(record_id)
        attempt = int(record.get("attempt_index", 1))
        if previous is None or attempt > int(previous.get("attempt_index", 1)):
            finals[record_id] = record
    return finals


def canonical_tool_hist(trajectories: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trajectory in trajectories:
        for step in trajectory.get("steps") or []:
            call = step.get("tool_call") if isinstance(step, dict) else None
            if isinstance(call, dict) and isinstance(call.get("tool"), str):
                counts[call["tool"]] += 1
    return counts


def attempted_call_hists(records: Iterable[Mapping[str, Any]]) -> tuple[Counter, Counter]:
    """Count harness results, including rejected/error primitive calls, from raw attempts."""
    calls: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for record in records:
        for turn in record.get("turns") or []:
            results = turn.get("native_bundle_results")
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict) or not isinstance(result.get("tool"), str):
                        continue
                    calls[result["tool"]] += 1
                    if result.get("status") == "error":
                        errors[result["tool"]] += 1
                continue
            parsed = turn.get("parsed") if isinstance(turn, dict) else None
            if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
                calls[parsed["tool"]] += 1
                if turn.get("tool_status") == "error":
                    errors[parsed["tool"]] += 1
    return calls, errors


def count_vector(
    counts: Mapping[str, int],
    tools: Sequence[str] = SQL_IDENTIFIABLE_ATOMIC_TOOLS,
) -> tuple[int, ...]:
    return tuple(int(counts.get(tool, 0)) for tool in tools)


def total_variation(left: Sequence[int], right: Sequence[int]) -> float | None:
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0 or right_total <= 0:
        return None
    return 0.5 * sum(
        abs(lvalue / left_total - rvalue / right_total)
        for lvalue, rvalue in zip(left, right)
    )


def sum_proxy_counts(
    ids: Iterable[str],
    profiles: Mapping[str, Mapping[str, Any]],
) -> Counter[str]:
    total: Counter[str] = Counter()
    for record_id in ids:
        profile = profiles[record_id]
        total.update(profile.get("tool_proxy_counts") or {})
    return total


def public_attrition_audit(
    attempted_rows: Sequence[dict[str, Any]],
    successful_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not attempted_rows or not successful_rows:
        return None
    return distribution_audit(
        attempted_rows,
        successful_rows,
        cutpoints=length_cutpoints(attempted_rows),
    )


def empty_result_training_audit(
    trajectories: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    empty_steps: list[dict[str, str]] = []
    unmarked_steps: list[dict[str, str]] = []
    empty_terminal_ids: list[str] = []
    for trajectory in trajectories:
        trajectory_id = trajectory_task_id(trajectory)
        steps = trajectory.get("steps") or []
        if trajectory_has_empty_terminal_evidence(steps):
            empty_terminal_ids.append(trajectory_id)
        for step in steps:
            if not isinstance(step, dict) or empty_result_target_reason(step) is None:
                continue
            item = {
                "trajectory_id": trajectory_id,
                "step_id": str(step.get("step_id")),
                "tool": str((step.get("tool_call") or {}).get("tool")),
            }
            empty_steps.append(item)
            if step.get("sft_target_eligible", True) is not False:
                unmarked_steps.append(item)
    return {
        "policy_version": EMPTY_RESULT_POLICY_VERSION,
        "intermediate_empty_result_policy": "retain-as-context-no-loss",
        "terminal_empty_evidence_policy": "exclude-whole-trajectory",
        "empty_result_steps": len(empty_steps),
        "empty_result_trajectories": len({item["trajectory_id"] for item in empty_steps}),
        "unmarked_empty_result_target_steps": len(unmarked_steps),
        "empty_terminal_trajectories": len(empty_terminal_ids),
        "empty_terminal_trajectory_ids": sorted(empty_terminal_ids),
        "training_filter_ready": not unmarked_steps and not empty_terminal_ids,
    }


def build_report(
    *,
    cohort_manifest_path: Path,
    cohort_path: Path,
    profile_path: Path,
    verified_path: Path,
    all_path: Path,
) -> dict[str, Any]:
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    cohort_rows = read_jsonl(cohort_path)
    profile_rows = read_jsonl(profile_path)
    verified = read_jsonl(verified_path)
    attempts = read_jsonl(all_path)

    cohort = index_unique(cohort_rows, id_fn=task_id, source="cohort")
    profiles = index_unique(profile_rows, id_fn=task_id, source="private profiles")
    verified_by_id = index_unique(
        verified,
        id_fn=trajectory_task_id,
        source="verified trajectories",
    )
    finals = final_attempts(attempts)

    if set(profiles) != set(cohort):
        raise ValueError("private SQL profiles do not exactly match the frozen cohort")
    attempted_ids = set(finals)
    verified_ids = set(verified_by_id)
    unknown_attempted = sorted(attempted_ids - set(cohort))
    unknown_verified = sorted(verified_ids - attempted_ids)
    if unknown_attempted:
        raise ValueError(f"rollout attempts outside frozen cohort: {unknown_attempted[:5]}")
    if unknown_verified:
        raise ValueError(f"verified trajectories without attempts: {unknown_verified[:5]}")

    schemes = Counter(str(record.get("tool_scheme")) for record in verified)
    unsupported = sorted(set(schemes) - SUPPORTED_TOOL_SCHEMES)
    if unsupported:
        raise ValueError(f"unsupported tool schemes: {unsupported}")

    canonical_hist = canonical_tool_hist(verified)
    attempted_hist, attempted_error_hist = attempted_call_hists(attempts)
    attempted_proxy = sum_proxy_counts(attempted_ids, profiles)
    verified_proxy = sum_proxy_counts(verified_ids, profiles)
    actual_sql_vector = count_vector(canonical_hist)
    verified_proxy_vector = count_vector(verified_proxy)
    attrition = public_attrition_audit(
        [cohort[record_id] for record_id in sorted(attempted_ids)],
        [cohort[record_id] for record_id in sorted(verified_ids)],
    )

    admissions = Counter(
        str(record.get("training_admission") or "missing") for record in verified
    )
    export_flags = Counter(bool(record.get("sft_export_eligible")) for record in verified)
    structural_ready = all(
        record.get("label_status") == "verified"
        and bool((record.get("steps") or []))
        for record in verified
    )
    explicitly_training_admitted = bool(verified) and set(admissions) == {
        ELIGIBLE_TRAINING_ADMISSION
    } and set(export_flags) == {True}
    empty_result_audit = empty_result_training_audit(verified)

    final_outcomes = Counter(
        "correct" if record.get("correct") else "failed"
        for record in finals.values()
    )
    final_legal = sum(bool(record.get("legal")) for record in finals.values())
    report = {
        "schema_version": AUDIT_VERSION,
        "inputs": {
            "cohort_manifest": str(cohort_manifest_path),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
            "cohort": str(cohort_path),
            "cohort_sha256": sha256_file(cohort_path),
            "private_sampling_profiles": str(profile_path),
            "private_sampling_profiles_sha256": sha256_file(profile_path),
            "verified": str(verified_path),
            "verified_sha256": sha256_file(verified_path),
            "all_attempts": str(all_path),
            "all_attempts_sha256": sha256_file(all_path),
        },
        "selection_boundary": {
            "sql_compilation_role": "sampling-only aggregate profile",
            "sql_profile_teacher_visible": False,
            "sql_profile_is_training_target": False,
            "cohort_selection_gates_passed": bool(
                cohort_manifest.get("all_acceptance_gates_passed")
            ),
        },
        "coverage": {
            "frozen_cohort_tasks": len(cohort),
            "attempt_records": len(attempts),
            "attempted_unique_tasks": len(attempted_ids),
            "unattempted_tasks": len(cohort) - len(attempted_ids),
            "verified_success_tasks": len(verified_ids),
            "verified_success_rate_over_attempted": (
                len(verified_ids) / len(attempted_ids) if attempted_ids else None
            ),
            "final_outcomes": dict(sorted(final_outcomes.items())),
            "final_legal_tasks": final_legal,
            "final_legal_rate": final_legal / len(finals) if finals else None,
        },
        "actual_tool_usage": {
            "verified_legal_calls": dict(canonical_hist.most_common()),
            "all_attempted_harness_calls": dict(attempted_hist.most_common()),
            "all_attempted_error_calls": dict(attempted_error_hist.most_common()),
            "sql_identifiable_tools": list(SQL_IDENTIFIABLE_ATOMIC_TOOLS),
            "exploration_policy_tools": list(NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS),
            "verified_exploration_calls": {
                tool: canonical_hist.get(tool, 0)
                for tool in NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS
            },
        },
        "sampling_proxy_comparison": {
            "attempted_task_proxy_counts": dict(attempted_proxy),
            "verified_task_proxy_counts": dict(verified_proxy),
            "verified_actual_sql_identifiable_counts": {
                tool: canonical_hist.get(tool, 0)
                for tool in SQL_IDENTIFIABLE_ATOMIC_TOOLS
            },
            "verified_proxy_vs_actual_action_share_tv": total_variation(
                verified_proxy_vector,
                actual_sql_vector,
            ),
            "interpretation": (
                "SQL counts are sampling strata, not expected teacher actions; differences measure "
                "policy/execution behavior and are not label errors."
            ),
        },
        "verified_success_attrition_distribution": attrition,
        "protocol_and_training_boundary": {
            "tool_schemes": dict(sorted(schemes.items())),
            "training_admission_values": dict(sorted(admissions.items())),
            "sft_export_eligible_values": {
                str(key).lower(): value for key, value in sorted(export_flags.items())
            },
            "verified_records_structurally_present": structural_ready,
            "explicitly_training_admitted": explicitly_training_admitted,
            "candidate_sft_export_allowed": (
                explicitly_training_admitted
                and empty_result_audit["training_filter_ready"]
            ),
            "native_bundle_flattening_allowed": False,
            "empty_result_training_filter": empty_result_audit,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--private-profiles", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--all", dest="all_attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        cohort_manifest_path=args.cohort_manifest.resolve(),
        cohort_path=args.cohort.resolve(),
        profile_path=args.private_profiles.resolve(),
        verified_path=args.verified.resolve(),
        all_path=args.all_attempts.resolve(),
    )
    write_json_atomic(args.out.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
