#!/usr/bin/env python3
"""Shared, frozen primitives for the independent Arm-B vanilla-GRPO fallback.

This module intentionally contains no Arm-A mutation and no training code.  It
only validates immutable fixed-policy rollout artifacts and implements the
public-field-only deterministic balancing used between Arm-B stages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PENDING_MANIFEST_SCHEMA = "table-agent-fixed-rollout-pool-pending-v1"
TRAJECTORY_SCHEMA = "table-agent-fixed-policy-episode-v1"
PROTOCOL_VERSION = "version26"
PROTOCOL_HASH = "4da19387399bd3a5"
STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
GENERATION_SEED_SCHEME = "sha256-task-sample-turn-v1"
WIDE_COHORT_SCHEMA = "vanilla-grpo-arm-b-wide3000-cohort-v1"
WIDE_COHORT_STATUS = "frozen_arm_b_wide_screen_cohort"
WIDE_COHORT_NAMESPACE = "qwen3-v26-arm-b-wide3000-v1"
F1_AUDIT_SCHEMA = "vanilla-grpo-arm-b-wide-k8-screen-audit-v1"
F2_AUDIT_SCHEMA = "vanilla-grpo-arm-b-k16-confirmation-audit-v1"
SELECTION_SCHEMA = "vanilla-grpo-arm-b-k16-selection-v1"
SELECTION_STATUS = "frozen_arm_b_training_cohort"
SELECTION_NAMESPACE = "qwen3-v26-arm-b-train320-k16-v1"
F3_AUDIT_SCHEMA = "vanilla-grpo-arm-b-k16-validation-audit-v1"

WIDE_SELECTION_SEED = "qwen3-v26-arm-b-wide3000-v1-20260812"
F1_GENERATION_SEED = 20260814
F1_SELECTION_SEED = "qwen3-v26-arm-b-confirmation640-v1-20260812"
F2_GENERATION_SEED = 20260815
F2_SELECTION_SEED = "qwen3-v26-arm-b-k16-selected384-v1-20260812"
F3_GENERATION_SEED = 20260816
TRAIN_DATA_SEED = 20260812

FORMAL_TRAINING_CONTRACT = {
    "optimizer_updates": 32,
    "prompts_per_update": 20,
    "group_size": 16,
    "train_passes": 2,
    "prompt_appearances": 640,
    "fresh_online_trajectories": 10240,
    "sampler": "trl-0.29-repeat-sampler-v1",
    "shuffle_dataset": True,
    "data_seed": TRAIN_DATA_SEED,
    "task_order": "two deterministic data-seed shuffled passes",
    "per_pass_coverage": "each train320 identity exactly once",
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
}
SELECTION_VALIDATION_CONTRACT = {
    "policy": "fresh initial-SFT1",
    "records": 64,
    "group_size": 16,
    "seed": F3_GENERATION_SEED,
    "confirmation_seed_must_differ": F2_GENERATION_SEED,
    "generation_seed_scheme": GENERATION_SEED_SCHEME,
    "exact_clean_trajectories": 1024,
    "minimum_mixed_groups": 56,
    "minimum_core_groups": 48,
    "screen_trajectories_reused": False,
}
VALIDATION_AUDIT_CONTRACT = {
    "selection_schema": SELECTION_SCHEMA,
    "policy": "fresh initial-SFT1",
    "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
    "student_prompt_sha256": STUDENT_PROMPT_SHA256,
    "protocol_version": PROTOCOL_VERSION,
    "protocol_hash": PROTOCOL_HASH,
    "records": 64,
    "group_size": 16,
    "trajectories": 1024,
    "validation_seed": F3_GENERATION_SEED,
    "confirmation_seed_must_differ": F2_GENERATION_SEED,
    "generation_seed_scheme": GENERATION_SEED_SCHEME,
    "exact_clean_trajectories": 1024,
    "minimum_mixed_groups": 56,
    "minimum_core_groups": 48,
    "runtime_contamination_allowed": False,
    "screen_trajectories_reused": False,
    "optimizer_updates": 0,
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
}
ARM_B_SEED_REGISTRY = {
    "wide_cohort_selection": WIDE_SELECTION_SEED,
    "wide_k8_generation": F1_GENERATION_SEED,
    "confirmation640_selection": F1_SELECTION_SEED,
    "k16_confirmation_generation": F2_GENERATION_SEED,
    "selected384_selection": F2_SELECTION_SEED,
    "validation64_generation": F3_GENERATION_SEED,
    "formal_training_data": TRAIN_DATA_SEED,
}
RUNTIME_FAILURE_TYPES = frozenset(
    {"generation_oom", "generation_length", "context_overflow", "timeout_error"}
)
TIMEOUT_CODES = frozenset({"tool_execution_timeout"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_file_digest(path: Path, expected: str, *, role: str) -> str:
    if not valid_sha256(expected):
        raise ValueError(f"{role}: explicit lowercase SHA-256 is required")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{role}: expected a regular non-symlink file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role}: SHA-256 mismatch: actual={actual} expected={expected}")
    return actual


def read_bound_bytes(path: Path, expected: str, *, role: str) -> bytes:
    """Read one immutable snapshot and validate that exact byte string."""

    if not valid_sha256(expected):
        raise ValueError(f"{role}: explicit lowercase SHA-256 is required")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{role}: expected a regular non-symlink file: {path}")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected:
        raise ValueError(f"{role}: SHA-256 mismatch: actual={actual} expected={expected}")
    return data


def parse_json_bytes(data: bytes, *, path: Path) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def parse_jsonl_bytes(data: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(value)
    return rows


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = path.read_bytes()
    return data, parse_json_bytes(data, path=path)


def read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    return data, parse_jsonl_bytes(data, path=path)


def task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task row is missing example_id/instance_id")
    return value


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    if temporary.exists():
        raise ValueError(f"stale temporary output blocks publication: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def publish_exact(outputs: Sequence[tuple[Path, bytes]]) -> None:
    """Recoverably publish deterministic bytes in the supplied commit order.

    Existing finals or ``.next`` files are accepted only when byte-identical.
    Callers put their manifest/audit last, making that final rename the commit
    marker while allowing an interrupted prefix to be completed safely.
    """

    if not outputs:
        raise ValueError("publication set must not be empty")
    resolved: set[Path] = set()
    for path, payload in outputs:
        target = path.resolve()
        if target in resolved:
            raise ValueError(f"duplicate publication target: {path}")
        resolved.add(target)
        if not isinstance(payload, bytes):
            raise TypeError(f"publication payload must be bytes: {path}")
        temporary = path.with_name(path.name + ".next")
        for candidate in (path, temporary):
            if candidate.exists() and (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.read_bytes() != payload
            ):
                raise ValueError(
                    f"existing publication artifact differs or is unsafe: {candidate}"
                )
    for path, payload in outputs:
        temporary = path.with_name(path.name + ".next")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if temporary.exists():
                temporary.unlink()
            continue
        if not temporary.exists():
            temporary.write_bytes(payload)
        os.replace(temporary, path)


def stable_rank(seed: str, namespace: str, identifier: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{namespace}\0{identifier}".encode("utf-8")
    ).hexdigest()


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
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {
                "failure_type",
                "error_type",
                "execution_error_type",
                "recovered_from_error_type",
            } and nested == "timeout_error":
                return True
            if key in {"error_code", "code"} and nested in TIMEOUT_CODES:
                return True
            if isinstance(nested, (dict, list)) and _timeout_marker(nested):
                return True
    elif isinstance(value, list):
        return any(_timeout_marker(item) for item in value)
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
    if _timeout_marker(record.get("turns")) or _timeout_marker(
        record.get("error_events")
    ):
        reasons.add("timeout_evidence")
    if sample.get("process_update") is not True:
        reasons.add("process_update_false")
    if record.get("optimization_exclusion") is not None:
        reasons.add("optimization_exclusion")
    if record.get("rollout_tokenization_warning") is True:
        reasons.add("tokenization_warning")
    return sorted(reasons)


def _index_tasks(
    tasks: Sequence[dict[str, Any]], *, expected_tasks: int
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, int]]:
    if len(tasks) != expected_tasks:
        raise ValueError(f"observed {len(tasks)} tasks; expected {expected_tasks}")
    ordered: list[str] = []
    indexed: dict[str, dict[str, Any]] = {}
    examples: dict[str, int] = {}
    seen_examples: set[int] = set()
    for row in tasks:
        identifier = task_id(row)
        example_index = row.get("example_index")
        if (
            identifier in indexed
            or type(example_index) is not int
            or example_index in seen_examples
        ):
            raise ValueError(f"duplicate/invalid task identity: {identifier}")
        if not all(
            isinstance(row.get(field), str) and bool(str(row.get(field)).strip())
            for field in ("db_id", "db_path", "question")
        ):
            raise ValueError(f"task lacks db_id/db_path/question: {identifier}")
        indexed[identifier] = row
        examples[identifier] = example_index
        seen_examples.add(example_index)
        ordered.append(identifier)
    return ordered, indexed, examples


def audit_fixed_policy_groups(
    *,
    generation_manifest: Mapping[str, Any],
    tasks: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    tasks_sha256: str,
    trajectories_sha256: str,
    tasks_path: Path | None,
    expected_tasks: int,
    group_size: int,
    seed: int,
) -> dict[str, Any]:
    """Validate a no-update binary rollout pool and classify clean K groups.

    Valid runtime failures are retained as contamination evidence and exclude
    their entire group.  Identity, protocol, reward, ordering, or logprob
    corruption is structural and raises ``ValueError``.
    """

    if expected_tasks <= 0 or group_size <= 1:
        raise ValueError("expected_tasks and group_size must define a nontrivial pool")
    expected_trajectories = expected_tasks * group_size
    expected_manifest = {
        "schema_version": PENDING_MANIFEST_SCHEMA,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "student_prompt_sha256": STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": RUNTIME_TREE_SHA256,
        "adapter_sha256": INITIAL_ADAPTER_SHA256,
        "tasks_sha256": tasks_sha256,
        "tasks": expected_tasks,
        "group_size": group_size,
        "trajectories": expected_trajectories,
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
        "trajectories_sha256": trajectories_sha256,
    }
    mismatches = {
        key: {"observed": generation_manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if generation_manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"generation manifest contract mismatch: {mismatches}")
    if tasks_path is not None:
        declared = generation_manifest.get("tasks_path")
        if (
            not isinstance(declared, str)
            or Path(declared).resolve() != tasks_path.resolve()
        ):
            raise ValueError("generation manifest tasks_path is not byte-bound input")

    ordered_ids, indexed_tasks, examples = _index_tasks(
        tasks, expected_tasks=expected_tasks
    )
    if len(trajectories) != expected_trajectories:
        raise ValueError(
            f"observed {len(trajectories)} trajectories; expected {expected_trajectories}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_total = 0
    first_seen: list[str] = []
    seen: set[str] = set()
    for position, row in enumerate(trajectories):
        location = f"trajectories[{position}]"
        if (
            row.get("schema_version") != TRAJECTORY_SCHEMA
            or type(row.get("sequence")) is not int
            or row.get("sequence") != position
        ):
            raise ValueError(f"{location}: schema/sequence mismatch")
        environment = row.get("environment")
        sample = row.get("sample")
        if not isinstance(environment, dict) or not isinstance(sample, dict):
            raise ValueError(f"{location}: missing environment/sample")
        record = sample.get("audit_record")
        if not isinstance(record, dict):
            raise ValueError(f"{location}: missing sample.audit_record")
        identifier = environment.get("task_id")
        example_index = environment.get("example_index")
        sample_index = record.get("sample_index")
        expected_identifier = ordered_ids[position // group_size]
        expected_sample_index = position % group_size
        if (
            not isinstance(identifier, str)
            or identifier not in indexed_tasks
            or identifier != expected_identifier
            or type(example_index) is not int
            or examples[identifier] != example_index
            or record.get("example_index") != example_index
            or type(sample_index) is not int
            or sample_index != expected_sample_index
            or record.get("trajectory_id")
            != f"rl_{example_index}_sample_{sample_index}"
        ):
            raise ValueError(f"{location}: task/example/sample identity mismatch")
        task = indexed_tasks[identifier]
        expected_environment = {
            "dataset_split": "train",
            "example_index": example_index,
            "task_id": identifier,
            "db_id": task.get("db_id"),
            "db_path": task.get("db_path"),
            "question": task.get("question"),
            "gold_sql": task.get("gold_sql") or task.get("query"),
            "external_knowledge": task.get("external_knowledge"),
        }
        if any(
            environment.get(key) != value
            for key, value in expected_environment.items()
        ):
            raise ValueError(f"{location}: environment differs from task bytes")
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("protocol_hash") != PROTOCOL_HASH
        ):
            raise ValueError(f"{location}: protocol identity mismatch")
        optimizer_evidence = {
            key: record[key]
            for key in (
                "policy_global_step",
                "policy_micro_step",
                "optimizer_step",
                "global_step",
            )
            if key in record
        }
        if any(type(value) is not int or value != 0 for value in optimizer_evidence.values()):
            raise ValueError(f"{location}: optimizer update evidence in fixed pool")
        if sample.get("step_rewards") is not None or record.get("process_reward"):
            raise ValueError(f"{location}: non-result reward evidence")
        if sample.get("failure_type") != record.get("failure_type"):
            raise ValueError(f"{location}: sample/audit failure mismatch")
        if type(sample.get("process_update")) is not bool:
            raise ValueError(f"{location}: process_update must be an explicit boolean")

        contamination = _runtime_contamination(sample, record)
        policy_turns = row.get("policy_turns")
        empty_runtime_evidence = (
            policy_turns == []
            and sample.get("process_update") is False
            and sample.get("failure_type") in {"generation_oom", "context_overflow"}
        )
        if not empty_runtime_evidence and not _policy_evidence_valid(policy_turns):
            raise ValueError(f"{location}: invalid token/logprob evidence")

        correct = sample.get("correct")
        expected_reward = (
            0.0 if sample.get("process_update") is False else float(bool(correct))
        )
        reward = sample.get("reward")
        result_reward = record.get("result_reward")
        if not (
            isinstance(correct, bool)
            and record.get("correct") is correct
            and _finite_number(reward)
            and float(reward) == expected_reward
            and isinstance(result_reward, dict)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and _finite_number(result_reward.get("value"))
            and float(result_reward["value"]) == expected_reward
        ):
            raise ValueError(f"{location}: invalid binary result-only reward")
        correct_total += int(correct)
        groups[identifier].append(
            {
                "sample_index": sample_index,
                "correct": correct,
                "contamination": contamination,
            }
        )
        if identifier not in seen:
            seen.add(identifier)
            first_seen.append(identifier)

    if first_seen != ordered_ids:
        raise ValueError("trajectory groups do not follow frozen task order")
    if generation_manifest.get("correct_trajectories") != correct_total:
        raise ValueError("manifest correct_trajectories does not reconcile")

    summaries: list[dict[str, Any]] = []
    for identifier in ordered_ids:
        entries = groups.get(identifier, [])
        if (
            len(entries) != group_size
            or sorted(entry["sample_index"] for entry in entries)
            != list(range(group_size))
        ):
            raise ValueError(f"{identifier}: not one exact K={group_size} group")
        contamination = sorted(
            {reason for entry in entries for reason in entry["contamination"]}
        )
        usable = not contamination
        correct_count = sum(bool(entry["correct"]) for entry in entries)
        mixed = usable and 1 <= correct_count <= group_size - 1
        core = usable and 2 <= correct_count <= group_size - 2
        summaries.append(
            {
                "task_id": identifier,
                "example_index": examples[identifier],
                "trajectories": group_size,
                "correct_count": correct_count,
                "uncertainty": (
                    correct_count * (group_size - correct_count) if mixed else 0
                ),
                "usable": usable,
                "mixed_boundary": mixed,
                "core_boundary": core,
                "contamination": contamination,
            }
        )
    return {
        "tasks": expected_tasks,
        "group_size": group_size,
        "trajectories": expected_trajectories,
        "usable_groups": sum(int(group["usable"]) for group in summaries),
        "contaminated_groups": sum(int(not group["usable"]) for group in summaries),
        "mixed_boundary_groups": sum(int(group["mixed_boundary"]) for group in summaries),
        "core_boundary_groups": sum(int(group["core_boundary"]) for group in summaries),
        "correct_trajectories": correct_total,
        "groups": summaries,
    }


def _quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute length quantiles from an empty reference")
    return int(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def length_cutpoints(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    values = [len(str(row.get("question") or "").strip()) for row in rows]
    if any(value < 1 for value in values):
        raise ValueError("every task needs a nonempty question")
    return tuple(_quantile(values, probability) for probability in (0.2, 0.4, 0.6, 0.8))


def length_bin(row: Mapping[str, Any], cutpoints: Sequence[int]) -> str:
    value = len(str(row.get("question") or "").strip())
    for index, boundary in enumerate(cutpoints, start=1):
        if value <= boundary:
            return f"q{index}"
    return "q5"


def knowledge_bin(row: Mapping[str, Any]) -> str:
    value = row.get("external_knowledge")
    return "present" if isinstance(value, str) and value.strip() else "absent"


def total_variation(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        raise ValueError("TV distance requires two nonempty populations")
    left_counts = Counter(left)
    right_counts = Counter(right)
    return 0.5 * sum(
        abs(left_counts[key] / len(left) - right_counts[key] / len(right))
        for key in set(left_counts) | set(right_counts)
    )


def balanced_uncertainty_select(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: str,
    reference: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Uncertainty-first selection with public DB/length/knowledge balancing."""

    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} candidates; need {count}")
    cutpoints = length_cutpoints(reference)
    features = (
        lambda row: str(row.get("db_id") or ""),
        lambda row: length_bin(row, cutpoints),
        knowledge_bin,
    )
    reference_proportions: list[dict[str, float]] = []
    for feature in features:
        counts = Counter(feature(row) for row in reference)
        reference_proportions.append(
            {key: value / len(reference) for key, value in counts.items()}
        )
    selected_counts = [Counter() for _ in features]
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not isinstance(candidate.get("task"), dict):
            raise ValueError("candidate lacks task bytes")
        buckets[int(candidate["uncertainty"])].append(candidate)

    selected: list[dict[str, Any]] = []
    for uncertainty in sorted(buckets, reverse=True):
        available = list(buckets[uncertainty])
        while available and len(selected) < count:
            next_size = len(selected) + 1

            def score(candidate: dict[str, Any]) -> tuple[float, str]:
                objective = 0.0
                row = candidate["task"]
                for axis, feature in enumerate(features):
                    candidate_value = feature(row)
                    categories = (
                        set(reference_proportions[axis])
                        | set(selected_counts[axis])
                        | {candidate_value}
                    )
                    objective += 0.5 * sum(
                        abs(
                            (
                                selected_counts[axis][category]
                                + int(category == candidate_value)
                            )
                            / next_size
                            - reference_proportions[axis].get(category, 0.0)
                        )
                        for category in categories
                    )
                return (
                    objective,
                    stable_rank(seed, "public-distribution", str(candidate["task_id"])),
                )

            chosen = min(available, key=score)
            available.remove(chosen)
            selected.append(chosen)
            for axis, feature in enumerate(features):
                selected_counts[axis][feature(chosen["task"])] += 1
        if len(selected) == count:
            break

    selected_rows = [candidate["task"] for candidate in selected]
    distribution = {
        "question_length_quintile_cutpoints_characters": list(cutpoints),
        "unique_databases": len({str(row.get("db_id") or "") for row in selected_rows}),
        "database_tv": total_variation(
            [str(row.get("db_id") or "") for row in reference],
            [str(row.get("db_id") or "") for row in selected_rows],
        ),
        "length_tv": total_variation(
            [length_bin(row, cutpoints) for row in reference],
            [length_bin(row, cutpoints) for row in selected_rows],
        ),
        "knowledge_tv": total_variation(
            [knowledge_bin(row) for row in reference],
            [knowledge_bin(row) for row in selected_rows],
        ),
    }
    return selected, distribution
