#!/usr/bin/env python3
"""Select a representative BIRD-train cohort for causal atomic-teacher rollout.

The selector balances three independent views of the full training population:

* public problem shape: question-length quintile and external-knowledge presence;
* database identity;
* a harness-only SQL structural profile over SQL-identifiable atomic tool families.

The SQL profile is used only for sampling.  It is neither an executable plan nor a teacher-visible
label.  The selected full task records retain gold fields solely because the local harness needs
them for hidden terminal scoring.  A separate teacher-visible projection proves that no gold SQL
or sampling profile is exported to the provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "harness"))

from select_bird_train_baseline import (  # noqa: E402
    cell_key,
    distribution_audit,
    length_cutpoints,
    representative_order,
    select_rows,
    sha256_file,
    task_id,
)
from sql_atomic_tool_profile import (  # noqa: E402
    NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS,
    PROFILE_VERSION,
    SQL_IDENTIFIABLE_ATOMIC_TOOLS,
    profile_task,
)
from filter_nonempty_training_tasks import FILTER_VERSION  # noqa: E402


HISTORICAL_SELECTION_VERSION = "representative-atomic-teacher-cohort-v1"
SELECTION_VERSION = "representative-atomic-teacher-cohort-v2-nonempty"
TEACHER_VISIBLE_FIELDS = (
    "example_id",
    "example_index",
    "db_id",
    "question",
    "external_knowledge",
)
FORBIDDEN_TEACHER_FIELDS = (
    "gold_sql",
    "query",
    "SQL",
    "gold_exec_results",
    "gold_sql_path",
    "sql_atomic_tool_profile",
    "tool_proxy_counts",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_rank(value: str, seed: str, namespace: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def verify_nonempty_filter_manifest(
    manifest_path: Path,
    *,
    reference_path: Path,
    eligible_path: Path,
) -> dict[str, Any]:
    """Bind selection inputs to a successful private gold-denotation filter run."""
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != FILTER_VERSION:
        raise ValueError(
            f"nonempty filter manifest must use schema_version {FILTER_VERSION!r}"
        )
    if value.get("all_acceptance_gates_passed") is not True:
        raise ValueError("nonempty filter manifest did not pass all acceptance gates")
    outputs = value.get("outputs") or {}
    expected = (
        ("reference_nonempty", reference_path),
        ("eligible_nonempty", eligible_path),
    )
    for key, path in expected:
        record = outputs.get(key) or {}
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"nonempty filter manifest hash mismatch for {key}")
        if record.get("records") != len(read_jsonl(path)):
            raise ValueError(f"nonempty filter manifest record-count mismatch for {key}")
    status_counts = value.get("status_counts") or {}
    eligible_status = status_counts.get("eligible") or {}
    if eligible_status.get("nonempty") != outputs["eligible_nonempty"].get("records"):
        raise ValueError("nonempty filter manifest eligible certification count mismatch")
    private_audit = outputs.get("private_status_audit") or {}
    if private_audit.get("teacher_visible") is not False:
        raise ValueError("nonempty filter private status audit must not be teacher-visible")
    return value


def index_unique(rows: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = task_id(row, source=source)
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate task id {row_id}")
        indexed[row_id] = row
    return indexed


def build_profiles(
    rows: Sequence[dict[str, Any]],
    *,
    source: str,
) -> dict[str, tuple[int, ...]]:
    profiles: dict[str, tuple[int, ...]] = {}
    for row in rows:
        row_id = task_id(row, source=source)
        try:
            profile = profile_task(row)
        except Exception as exc:  # noqa: BLE001 - include task identity without SQL text
            raise ValueError(
                f"{source}: failed to parse hidden SQL for {row_id}: {type(exc).__name__}"
            ) from exc
        profiles[row_id] = tuple(profile.counts[tool] for tool in SQL_IDENTIFIABLE_ATOMIC_TOOLS)
    return profiles


def vector_counts(
    rows: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, tuple[int, ...]],
) -> tuple[int, ...]:
    totals = [0] * len(SQL_IDENTIFIABLE_ATOMIC_TOOLS)
    for row in rows:
        vector = profiles[task_id(row)]
        for index, value in enumerate(vector):
            totals[index] += value
    return tuple(totals)


def total_variation(left: Sequence[int], right: Sequence[int]) -> float:
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0 or right_total <= 0:
        raise ValueError("tool-count distribution cannot be empty")
    return 0.5 * sum(
        abs(lvalue / left_total - rvalue / right_total)
        for lvalue, rvalue in zip(left, right)
    )


def tool_balance_objective(
    reference_counts: Sequence[int],
    selected_counts: Sequence[int],
    *,
    reference_size: int,
    selected_size: int,
) -> float:
    """Balance common action share while retaining pressure on rare tool families."""
    tv = total_variation(reference_counts, selected_counts)
    relative_rate_errors = []
    for reference, selected in zip(reference_counts, selected_counts):
        reference_rate = reference / reference_size
        selected_rate = selected / selected_size
        relative_rate_errors.append(
            abs(selected_rate - reference_rate) / max(reference_rate, 1 / reference_size)
        )
    return 0.7 * tv + 0.3 * mean(relative_rate_errors)


def _add_swap(
    counts: tuple[int, ...],
    outgoing: tuple[int, ...],
    incoming: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        current - old + new
        for current, old, new in zip(counts, outgoing, incoming)
    )


def optimize_tool_balance(
    selected: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    profiles: Mapping[str, tuple[int, ...]],
    *,
    cutpoints: Sequence[int],
    seed: str,
    max_swaps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Improve tool balance through swaps that preserve exact DB/problem strata."""
    candidate_index = {task_id(row): row for row in candidates}
    chosen = {task_id(row): row for row in selected}
    reference_counts = vector_counts(reference, profiles)
    selected_counts = vector_counts(selected, profiles)
    initial_counts = selected_counts
    initial_objective = tool_balance_objective(
        reference_counts,
        selected_counts,
        reference_size=len(reference),
        selected_size=len(selected),
    )

    def stratum(row: Mapping[str, Any]) -> tuple[str, tuple[str, str]]:
        return str(row.get("db_id") or ""), cell_key(row, cutpoints)

    candidate_strata: dict[
        tuple[str, tuple[str, str]], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in candidates:
        candidate_strata[stratum(row)].append(row)

    swaps = 0
    epsilon = 1e-12
    while swaps < max_swaps:
        current_objective = tool_balance_objective(
            reference_counts,
            selected_counts,
            reference_size=len(reference),
            selected_size=len(selected),
        )
        best: tuple[
            float,
            str,
            dict[str, Any],
            dict[str, Any],
            tuple[int, ...],
        ] | None = None

        for key in sorted(candidate_strata, key=str):
            selected_by_vector: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
            available_by_vector: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in candidate_strata[key]:
                row_id = task_id(row)
                target = selected_by_vector if row_id in chosen else available_by_vector
                target[profiles[row_id]].append(row)
            for outgoing_vector in sorted(selected_by_vector):
                for incoming_vector in sorted(available_by_vector):
                    if outgoing_vector == incoming_vector:
                        continue
                    proposed_counts = _add_swap(
                        selected_counts,
                        outgoing_vector,
                        incoming_vector,
                    )
                    objective = tool_balance_objective(
                        reference_counts,
                        proposed_counts,
                        reference_size=len(reference),
                        selected_size=len(selected),
                    )
                    if objective >= current_objective - epsilon:
                        continue
                    outgoing = min(
                        selected_by_vector[outgoing_vector],
                        key=lambda row: stable_rank(task_id(row), seed, "tool-balance-out"),
                    )
                    incoming = min(
                        available_by_vector[incoming_vector],
                        key=lambda row: stable_rank(task_id(row), seed, "tool-balance-in"),
                    )
                    tie_rank = stable_rank(
                        f"{task_id(outgoing)}->{task_id(incoming)}",
                        seed,
                        "tool-balance-swap",
                    )
                    proposal = (objective, tie_rank, outgoing, incoming, proposed_counts)
                    if best is None or proposal[:2] < best[:2]:
                        best = proposal

        if best is None:
            break
        _objective, _tie_rank, outgoing, incoming, selected_counts = best
        del chosen[task_id(outgoing)]
        chosen[task_id(incoming)] = candidate_index[task_id(incoming)]
        swaps += 1

    final_rows = list(chosen.values())
    final_objective = tool_balance_objective(
        reference_counts,
        selected_counts,
        reference_size=len(reference),
        selected_size=len(selected),
    )
    return final_rows, {
        "swap_count": swaps,
        "max_swaps": max_swaps,
        "initial_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, initial_counts)),
        "final_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, selected_counts)),
        "reference_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, reference_counts)),
        "initial_objective": round(initial_objective, 10),
        "final_objective": round(final_objective, 10),
        "objective_nonincreasing": final_objective <= initial_objective + epsilon,
    }


def teacher_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in TEACHER_VISIBLE_FIELDS}


def tool_distribution_audit(
    reference: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    profiles: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    reference_counts = vector_counts(reference, profiles)
    selected_counts = vector_counts(selected, profiles)
    reference_total = sum(reference_counts)
    selected_total = sum(selected_counts)
    per_tool: dict[str, Any] = {}
    for tool, reference_count, selected_count in zip(
        SQL_IDENTIFIABLE_ATOMIC_TOOLS,
        reference_counts,
        selected_counts,
    ):
        per_tool[tool] = {
            "reference_count": reference_count,
            "selected_count": selected_count,
            "reference_per_task": round(reference_count / len(reference), 8),
            "selected_per_task": round(selected_count / len(selected), 8),
            "reference_action_share": round(reference_count / reference_total, 8),
            "selected_action_share": round(selected_count / selected_total, 8),
        }
    return {
        "profile_version": PROFILE_VERSION,
        "scope": "hidden sampling-only SQL structural requirements; not actions or trajectories",
        "sql_identifiable_tools": list(SQL_IDENTIFIABLE_ATOMIC_TOOLS),
        "non_sql_identifiable_tools": list(NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS),
        "reference_total_proxy_actions": reference_total,
        "selected_total_proxy_actions": selected_total,
        "action_share_total_variation_distance": round(
            total_variation(reference_counts, selected_counts), 8
        ),
        "balance_objective": round(
            tool_balance_objective(
                reference_counts,
                selected_counts,
                reference_size=len(reference),
                selected_size=len(selected),
            ),
            10,
        ),
        "per_tool": per_tool,
    }


def select_cohort(
    reference_rows: Sequence[dict[str, Any]],
    eligible_rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: str,
    max_swaps: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]], dict[str, Any]]:
    reference_index = index_unique(reference_rows, source="reference population")
    eligible_index = index_unique(eligible_rows, source="eligible population")
    unknown = sorted(set(eligible_index) - set(reference_index))
    if unknown:
        raise ValueError(f"eligible ids absent from reference population: {unknown[:5]}")

    reference_profiles = build_profiles(reference_rows, source="reference population")
    # Reuse the full-reference profile for eligible rows with the same frozen task identity.
    initial, public_details = select_rows(
        reference_rows,
        eligible_rows,
        set(),
        count=count,
        seed=seed,
    )
    cutpoints = tuple(public_details["question_length_quintile_cutpoints_characters"])
    optimized, optimization = optimize_tool_balance(
        initial,
        eligible_rows,
        reference_rows,
        reference_profiles,
        cutpoints=cutpoints,
        seed=seed,
        max_swaps=max_swaps,
    )
    ordered = representative_order(
        optimized,
        reference_rows,
        cutpoints=cutpoints,
        seed=seed,
    )
    details = {
        "public_selection": public_details,
        "tool_balance_optimization": optimization,
    }
    return ordered, reference_profiles, details


def preserve_certified_cohort(
    reference_rows: Sequence[dict[str, Any]],
    eligible_rows: Sequence[dict[str, Any]],
    preserved_rows: Sequence[dict[str, Any]],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]], dict[str, Any]]:
    """Keep a frozen cohort/order when every member survives the new task gate."""
    eligible_index = index_unique(eligible_rows, source="eligible population")
    preserved_ids = [task_id(row, source="preserved cohort") for row in preserved_rows]
    if len(preserved_ids) != count:
        raise ValueError(
            f"preserved cohort contains {len(preserved_ids)} tasks; expected {count}"
        )
    if len(set(preserved_ids)) != len(preserved_ids):
        raise ValueError("preserved cohort contains duplicate task ids")
    missing = [row_id for row_id in preserved_ids if row_id not in eligible_index]
    if missing:
        raise ValueError(
            "preserved cohort contains tasks rejected by the certified eligible pool: "
            f"{missing[:5]}"
        )
    selected = [eligible_index[row_id] for row_id in preserved_ids]
    profiles = build_profiles(reference_rows, source="reference population")
    reference_counts = vector_counts(reference_rows, profiles)
    selected_counts = vector_counts(selected, profiles)
    objective = tool_balance_objective(
        reference_counts,
        selected_counts,
        reference_size=len(reference_rows),
        selected_size=len(selected),
    )
    cutpoints = length_cutpoints(reference_rows)
    details = {
        "public_selection": {
            "selection_mode": "preserve_existing_certified_cohort_and_order",
            "question_length_quintile_cutpoints_characters": list(cutpoints),
        },
        "tool_balance_optimization": {
            "swap_count": 0,
            "max_swaps": 0,
            "initial_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, selected_counts)),
            "final_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, selected_counts)),
            "reference_counts": dict(zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, reference_counts)),
            "initial_objective": round(objective, 10),
            "final_objective": round(objective, 10),
            "objective_nonincreasing": True,
        },
    }
    return selected, profiles, details


def build_outputs(
    *,
    reference_path: Path,
    eligible_path: Path,
    output_path: Path,
    projection_path: Path,
    profile_path: Path,
    manifest_path: Path,
    count: int,
    seed: str,
    max_swaps: int,
    nonempty_filter_manifest_path: Path | None = None,
    allow_unfiltered_historical_reproduction: bool = False,
    preserve_cohort_path: Path | None = None,
) -> dict[str, Any]:
    if nonempty_filter_manifest_path and allow_unfiltered_historical_reproduction:
        raise ValueError(
            "choose a nonempty filter manifest or historical unfiltered reproduction, not both"
        )
    if nonempty_filter_manifest_path is None and not allow_unfiltered_historical_reproduction:
        raise ValueError(
            "training-task cohort selection requires --nonempty-filter-manifest; only an "
            "explicit historical reproduction may bypass this task-admission gate"
        )
    if preserve_cohort_path is not None and nonempty_filter_manifest_path is None:
        raise ValueError("preserving a cohort requires a nonempty filter manifest")
    reference_rows = read_jsonl(reference_path)
    eligible_rows = read_jsonl(eligible_path)
    nonempty_filter_manifest = None
    if nonempty_filter_manifest_path is not None:
        nonempty_filter_manifest = verify_nonempty_filter_manifest(
            nonempty_filter_manifest_path,
            reference_path=reference_path,
            eligible_path=eligible_path,
        )
    unverified = [
        task_id(row)
        for row in eligible_rows
        if (row.get("metadata") or {}).get("tool_round_trip") != "verified"
    ]
    if unverified:
        raise ValueError(
            "eligible population contains tasks without verified tool compatibility: "
            f"{unverified[:5]}"
        )

    preserved_rows = read_jsonl(preserve_cohort_path) if preserve_cohort_path else None
    if preserved_rows is not None:
        selected, profiles, details = preserve_certified_cohort(
            reference_rows,
            eligible_rows,
            preserved_rows,
            count=count,
        )
    else:
        selected, profiles, details = select_cohort(
            reference_rows,
            eligible_rows,
            count=count,
            seed=seed,
            max_swaps=max_swaps,
        )
    projections = [teacher_projection(row) for row in selected]
    private_profiles = [
        {
            "example_id": task_id(row),
            "profile_version": PROFILE_VERSION,
            "tool_proxy_counts": dict(
                zip(SQL_IDENTIFIABLE_ATOMIC_TOOLS, profiles[task_id(row)])
            ),
            "sampling_only": True,
            "teacher_visible": False,
        }
        for row in selected
    ]

    write_jsonl_atomic(output_path, selected)
    write_jsonl_atomic(projection_path, projections)
    write_jsonl_atomic(profile_path, private_profiles)

    cutpoints = tuple(
        details["public_selection"]["question_length_quintile_cutpoints_characters"]
    )
    public_audit = distribution_audit(reference_rows, selected, cutpoints=cutpoints)
    tool_audit = tool_distribution_audit(reference_rows, selected, profiles)
    selected_ids = [task_id(row) for row in selected]
    forbidden_projection_keys = sorted(
        {
            key
            for row in projections
            for key in row
            if key in FORBIDDEN_TEACHER_FIELDS
        }
    )
    gates = {
        "record_count_exact": len(selected) == count,
        "unique_task_ids": len(set(selected_ids)) == count,
        "all_reference_databases_covered": len({row["db_id"] for row in selected})
        == len({row["db_id"] for row in reference_rows}),
        "database_tv_at_most_0_025": public_audit["axes"]["database"][
            "total_variation_distance"
        ] <= 0.025,
        "question_length_quintile_tv_at_most_0_01": public_audit["axes"][
            "question_length_quintile"
        ]["total_variation_distance"] <= 0.01,
        "external_knowledge_tv_at_most_0_01": public_audit["axes"][
            "external_knowledge"
        ]["total_variation_distance"] <= 0.01,
        "joint_problem_shape_tv_at_most_0_015": public_audit["axes"][
            "knowledge_x_length"
        ]["total_variation_distance"] <= 0.015,
        "question_length_mean_relative_delta_at_most_0_02": public_audit[
            "question_length_characters"
        ]["relative_mean_difference"] <= 0.02,
        "sql_tool_proxy_action_share_tv_at_most_0_01": tool_audit[
            "action_share_total_variation_distance"
        ] <= 0.01,
        "teacher_projection_has_no_forbidden_fields": not forbidden_projection_keys,
        "tool_balance_objective_nonincreasing": details["tool_balance_optimization"][
            "objective_nonincreasing"
        ],
        "task_admission_mode_explicit": (
            nonempty_filter_manifest is not None
            or allow_unfiltered_historical_reproduction
        ),
        "preserved_task_ids_and_order_exact": (
            preserved_rows is None
            or selected_ids == [task_id(row) for row in preserved_rows]
        ),
    }
    selection_version = (
        SELECTION_VERSION
        if nonempty_filter_manifest is not None
        else HISTORICAL_SELECTION_VERSION
    )
    manifest = {
        "schema_version": selection_version,
        "status": "frozen_teacher_rollout_candidate",
        "training_admission": "pending_causal_rollout_replay_quality_and_protocol_gate",
        "purpose": "representative 1500-task BIRD-train atomic external-teacher cohort",
        "reference_population": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "records": len(reference_rows),
            "databases": len({row["db_id"] for row in reference_rows}),
        },
        "eligible_population": {
            "path": str(eligible_path),
            "sha256": sha256_file(eligible_path),
            "records": len(eligible_rows),
            "rule": (
                "metadata.tool_round_trip == verified and hidden gold denotation is certified "
                "nonempty"
                if nonempty_filter_manifest is not None
                else "metadata.tool_round_trip == verified"
            ),
        },
        "task_admission": {
            "nonempty_gold_result_required": nonempty_filter_manifest is not None,
            "filter_manifest": (
                {
                    "path": str(nonempty_filter_manifest_path),
                    "sha256": sha256_file(nonempty_filter_manifest_path),
                    "schema_version": nonempty_filter_manifest["schema_version"],
                    "criterion": nonempty_filter_manifest["criterion"],
                    "zero_scalar_rule": nonempty_filter_manifest["zero_scalar_rule"],
                    "gold_sql_teacher_visible": False,
                    "gold_rows_teacher_visible": False,
                    "gold_result_status_teacher_visible": False,
                }
                if nonempty_filter_manifest is not None
                else None
            ),
            "historical_unfiltered_reproduction": allow_unfiltered_historical_reproduction,
        },
        "selection": {
            "count": count,
            "seed": seed,
            "algorithm": (
                "preserve the frozen input cohort and order after every member passes the "
                "hash-bound nonempty task gate"
                if preserved_rows is not None
                else (
                    "proportional database and knowledge-by-question-length quotas, followed by "
                    "same-database/same-problem-stratum swaps minimizing hidden SQL tool-profile "
                    "drift"
                )
            ),
            "preserved_cohort": (
                {
                    "path": str(preserve_cohort_path),
                    "sha256": sha256_file(preserve_cohort_path),
                    "records": len(preserved_rows),
                    "task_ids_and_order_preserved": selected_ids
                    == [task_id(row) for row in preserved_rows],
                }
                if preserved_rows is not None
                else None
            ),
            "sql_profile_is_sampling_only": True,
            "sql_profile_is_executable_plan": False,
            "gold_sql_used_for_sampling": True,
            "gold_sql_teacher_visible": False,
            "teacher_visible_fields": list(TEACHER_VISIBLE_FIELDS),
            "non_sql_identifiable_tools_audited_only_after_rollout": list(
                NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS
            ),
        },
        "outputs": {
            "harness_tasks": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "records": len(selected),
            },
            "teacher_visible_projection": {
                "path": str(projection_path),
                "sha256": sha256_file(projection_path),
                "records": len(projections),
                "forbidden_fields_present": forbidden_projection_keys,
            },
            "private_sampling_profiles": {
                "path": str(profile_path),
                "sha256": sha256_file(profile_path),
                "records": len(private_profiles),
                "teacher_visible": False,
            },
            "task_ids_in_frozen_order": selected_ids,
        },
        "distribution_audit": {
            "public_problem_and_database": public_audit,
            "hidden_sql_atomic_tool_proxy": tool_audit,
        },
        "selection_details": details,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "rollout_boundary": (
            "Teachers receive only question, external knowledge, catalog/schema, and causal "
            "read-only harness feedback. Gold SQL and sampling profiles remain local and hidden."
        ),
        "sft_boundary": (
            "No selected task is an SFT example by selection alone. Only terminal-correct, fresh-"
            "replayed, quality/no-leak/structural-audited causal actions may be exported, and the "
            "selected protocol must have explicit training admission."
        ),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-projection", type=Path, required=True)
    parser.add_argument("--private-profiles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nonempty-filter-manifest", type=Path)
    parser.add_argument(
        "--preserve-cohort",
        type=Path,
        help="retain an existing cohort and order when all members survive the certified filter",
    )
    parser.add_argument(
        "--allow-unfiltered-historical-reproduction",
        action="store_true",
        help="bypass the nonempty task gate only to reproduce a frozen historical v1 artifact",
    )
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", default="bird-train-atomic-teacher1500-v2-20260806")
    parser.add_argument("--max-tool-balance-swaps", type=int, default=2000)
    args = parser.parse_args()

    manifest = build_outputs(
        reference_path=args.reference.resolve(),
        eligible_path=args.eligible.resolve(),
        output_path=args.output.resolve(),
        projection_path=args.teacher_projection.resolve(),
        profile_path=args.private_profiles.resolve(),
        manifest_path=args.manifest.resolve(),
        count=args.count,
        seed=args.seed,
        max_swaps=args.max_tool_balance_swaps,
        nonempty_filter_manifest_path=(
            args.nonempty_filter_manifest.resolve()
            if args.nonempty_filter_manifest is not None
            else None
        ),
        allow_unfiltered_historical_reproduction=(
            args.allow_unfiltered_historical_reproduction
        ),
        preserve_cohort_path=(
            args.preserve_cohort.resolve() if args.preserve_cohort is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "records": manifest["outputs"]["harness_tasks"]["records"],
                "all_acceptance_gates_passed": manifest["all_acceptance_gates_passed"],
                "database_tv": manifest["distribution_audit"]["public_problem_and_database"][
                    "axes"
                ]["database"]["total_variation_distance"],
                "problem_shape_tv": manifest["distribution_audit"][
                    "public_problem_and_database"
                ]["axes"]["knowledge_x_length"]["total_variation_distance"],
                "sql_tool_proxy_tv": manifest["distribution_audit"][
                    "hidden_sql_atomic_tool_proxy"
                ]["action_share_total_variation_distance"],
                "manifest": str(args.manifest.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
