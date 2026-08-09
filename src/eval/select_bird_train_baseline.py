#!/usr/bin/env python3
"""Build a deterministic, representative BIRD-train evaluation baseline.

Selection is deliberately independent of gold SQL and gold execution results.  The output records
retain those harness-only fields because the evaluator needs them, but changing their values cannot
change which task ids are selected or their order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


ALGORITHM_VERSION = "bird-train-representative-baseline-v1"
PUBLIC_SELECTION_FIELDS = ("example_id", "db_id", "question", "external_knowledge")
FORBIDDEN_SELECTION_FIELDS = ("gold_sql", "query", "gold_exec_results", "gold_sql_path")
DEFAULT_PREFIX_AUDITS = (32, 50, 100, 200, 300)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def task_id(row: Mapping[str, Any], *, source: str = "row") -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: missing example_id/instance_id")
    return value


def index_unique(rows: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = task_id(row, source=source)
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate task id {row_id}")
        indexed[row_id] = row
    return indexed


def evidence_present(row: Mapping[str, Any]) -> bool:
    value = row.get("external_knowledge")
    return isinstance(value, str) and bool(value.strip())


def question_length(row: Mapping[str, Any]) -> int:
    value = row.get("question")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task_id(row)}: missing question")
    return len(value.strip())


def public_signature(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        task_id(row),
        str(row.get("db_id") or ""),
        str(row.get("question") or ""),
        str(row.get("external_knowledge") or ""),
    )


def quantile(sorted_values: Sequence[int], probability: float) -> int:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty population")
    index = max(0, math.ceil(probability * len(sorted_values)) - 1)
    return int(sorted_values[index])


def length_cutpoints(reference_rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    values = sorted(question_length(row) for row in reference_rows)
    return tuple(quantile(values, probability) for probability in (0.2, 0.4, 0.6, 0.8))


def length_bin(length: int, cutpoints: Sequence[int]) -> str:
    for index, boundary in enumerate(cutpoints):
        if length <= boundary:
            return f"q{index + 1}"
    return f"q{len(cutpoints) + 1}"


def cell_key(row: Mapping[str, Any], cutpoints: Sequence[int]) -> tuple[str, str]:
    return (
        "knowledge" if evidence_present(row) else "no_knowledge",
        length_bin(question_length(row), cutpoints),
    )


def stable_rank(row_id: str, seed: str, namespace: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{row_id}".encode("utf-8")).hexdigest()


def allocate_proportionally(
    reference_counts: Mapping[Any, int],
    capacities: Mapping[Any, int],
    total: int,
    *,
    minimum_one: bool,
) -> dict[Any, int]:
    """Round target proportions deterministically while respecting candidate capacity."""
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    keys = sorted(reference_counts, key=str)
    denominator = sum(reference_counts.values())
    if denominator <= 0:
        raise ValueError("reference population is empty")
    allocation = {
        key: 1 if minimum_one and reference_counts[key] > 0 and capacities.get(key, 0) > 0 else 0
        for key in keys
    }
    if sum(allocation.values()) > total:
        raise ValueError("total is too small to give every represented category one task")
    if sum(capacities.get(key, 0) for key in keys) < total:
        raise ValueError("candidate capacity is smaller than requested allocation")

    targets = {key: total * reference_counts[key] / denominator for key in keys}
    while sum(allocation.values()) < total:
        available = [key for key in keys if allocation[key] < capacities.get(key, 0)]
        if not available:
            raise ValueError("unable to complete proportional allocation")
        chosen = max(available, key=lambda key: (targets[key] - allocation[key], str(key)))
        allocation[chosen] += 1
    return allocation


def select_rows(
    reference_rows: Sequence[dict[str, Any]],
    eligible_rows: Sequence[dict[str, Any]],
    excluded_ids: set[str],
    *,
    count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select rows using only public task identity, DB, question, and evidence fields."""
    if count <= 0:
        raise ValueError("count must be positive")
    reference = index_unique(reference_rows, source="reference population")
    eligible = index_unique(eligible_rows, source="eligible population")
    unknown = sorted(set(eligible) - set(reference))
    if unknown:
        raise ValueError(f"eligible population contains ids absent from reference: {unknown[:5]}")
    for row_id, row in eligible.items():
        if public_signature(row) != public_signature(reference[row_id]):
            raise ValueError(f"public fields differ between reference and eligible row {row_id}")

    candidates = [row for row_id, row in eligible.items() if row_id not in excluded_ids]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible non-excluded rows; need {count}")
    cutpoints = length_cutpoints(reference_rows)

    reference_db_counts = Counter(str(row.get("db_id") or "") for row in reference_rows)
    candidate_db_counts = Counter(str(row.get("db_id") or "") for row in candidates)
    unavailable_databases = sorted(set(reference_db_counts) - set(candidate_db_counts))
    if unavailable_databases:
        raise ValueError(f"reference databases have no eligible candidates: {unavailable_databases}")
    database_quotas = allocate_proportionally(
        reference_db_counts,
        candidate_db_counts,
        count,
        minimum_one=True,
    )

    reference_by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        reference_by_database[str(row.get("db_id") or "")].append(row)
    for row in candidates:
        candidate_by_database[str(row.get("db_id") or "")].append(row)

    selected: list[dict[str, Any]] = []
    initial_cell_quotas: dict[str, dict[str, int]] = {}
    for database in sorted(database_quotas):
        quota = database_quotas[database]
        ref_cells = Counter(cell_key(row, cutpoints) for row in reference_by_database[database])
        candidate_cells = Counter(cell_key(row, cutpoints) for row in candidate_by_database[database])
        allocation = allocate_proportionally(
            ref_cells,
            candidate_cells,
            quota,
            minimum_one=False,
        )
        initial_cell_quotas[database] = {
            "|".join(key): value for key, value in sorted(allocation.items())
        }
        rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_by_database[database]:
            rows_by_cell[cell_key(row, cutpoints)].append(row)
        for cell, cell_quota in sorted(allocation.items()):
            ranked = sorted(
                rows_by_cell[cell],
                key=lambda row: stable_rank(task_id(row), seed, f"select:{database}:{cell}"),
            )
            selected.extend(ranked[:cell_quota])

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    selected_ids = {task_id(row) for row in selected}
    if len(selected_ids) != count:
        raise RuntimeError("selected cohort contains duplicate task ids")
    if selected_ids & excluded_ids:
        raise RuntimeError("selected cohort overlaps excluded ids")

    selected, global_cell_targets = rebalance_joint_cells(
        selected,
        candidates,
        reference_rows,
        cutpoints=cutpoints,
        seed=seed,
    )
    actual_cell_quotas: dict[str, dict[str, int]] = {}
    for database in sorted(database_quotas):
        counts = Counter(
            cell_key(row, cutpoints)
            for row in selected
            if str(row.get("db_id") or "") == database
        )
        actual_cell_quotas[database] = {
            "|".join(key): value for key, value in sorted(counts.items())
        }
    ordered = representative_order(selected, reference_rows, cutpoints=cutpoints, seed=seed)
    details = {
        "question_length_quintile_cutpoints_characters": list(cutpoints),
        "database_quotas": dict(sorted(database_quotas.items())),
        "initial_database_cell_quotas": initial_cell_quotas,
        "global_knowledge_length_targets": {
            "|".join(key): value for key, value in sorted(global_cell_targets.items())
        },
        "final_database_cell_counts": actual_cell_quotas,
    }
    return ordered, details


def rebalance_joint_cells(
    selected: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    cutpoints: Sequence[int],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    """Match global knowledge/length quotas through within-database swaps.

    Database quotas remain exact.  A direct surplus-to-deficit swap is used only when both tasks
    belong to the same database, so improved public-feature balance cannot alter DB coverage.
    """
    reference_counts = Counter(cell_key(row, cutpoints) for row in reference)
    candidate_counts = Counter(cell_key(row, cutpoints) for row in candidates)
    targets = allocate_proportionally(
        reference_counts,
        candidate_counts,
        len(selected),
        minimum_one=False,
    )
    chosen = {task_id(row): row for row in selected}
    candidate_by_database_cell: dict[tuple[str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (str(row.get("db_id") or ""), cell_key(row, cutpoints))
        candidate_by_database_cell[key].append(row)

    for _iteration in range(len(selected) * 2):
        current = Counter(cell_key(row, cutpoints) for row in chosen.values())
        surplus = {key for key in targets if current[key] > targets[key]}
        deficit = {key for key in targets if current[key] < targets[key]}
        if not surplus and not deficit:
            return list(chosen.values()), targets

        selected_by_database_cell: dict[
            tuple[str, tuple[str, str]], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in chosen.values():
            key = (str(row.get("db_id") or ""), cell_key(row, cutpoints))
            selected_by_database_cell[key].append(row)

        swaps: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        databases = sorted({str(row.get("db_id") or "") for row in selected})
        for database in databases:
            for outgoing_cell in sorted(surplus):
                outgoing_rows = selected_by_database_cell.get((database, outgoing_cell), [])
                if not outgoing_rows:
                    continue
                outgoing = min(
                    outgoing_rows,
                    key=lambda row: stable_rank(task_id(row), seed, "rebalance-out"),
                )
                for incoming_cell in sorted(deficit):
                    incoming_rows = [
                        row
                        for row in candidate_by_database_cell.get((database, incoming_cell), [])
                        if task_id(row) not in chosen
                    ]
                    if not incoming_rows:
                        continue
                    incoming = min(
                        incoming_rows,
                        key=lambda row: stable_rank(task_id(row), seed, "rebalance-in"),
                    )
                    rank = stable_rank(
                        f"{task_id(outgoing)}->{task_id(incoming)}",
                        seed,
                        "rebalance-swap",
                    )
                    swaps.append((rank, outgoing, incoming))
        if not swaps:
            break
        _rank, outgoing, incoming = min(swaps, key=lambda item: item[0])
        del chosen[task_id(outgoing)]
        chosen[task_id(incoming)] = incoming

    current = Counter(cell_key(row, cutpoints) for row in chosen.values())
    unresolved = {
        "|".join(key): {"selected": current[key], "target": targets[key]}
        for key in targets
        if current[key] != targets[key]
    }
    if unresolved:
        raise ValueError(f"unable to meet global knowledge/length targets: {unresolved}")
    return list(chosen.values()), targets


def representative_order(
    selected: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    cutpoints: Sequence[int],
    seed: str,
) -> list[dict[str, Any]]:
    """Order the fixed cohort so common prefixes remain reasonably representative."""
    axes = (
        ("database", lambda row: str(row.get("db_id") or ""), 0.55),
        ("length_bin", lambda row: length_bin(question_length(row), cutpoints), 0.20),
        ("knowledge", lambda row: "present" if evidence_present(row) else "absent", 0.10),
        ("knowledge_x_length", lambda row: "|".join(cell_key(row, cutpoints)), 0.15),
    )
    ref_distributions: dict[str, dict[str, float]] = {}
    for name, feature, _weight in axes:
        counts = Counter(feature(row) for row in reference)
        ref_distributions[name] = {key: value / len(reference) for key, value in counts.items()}

    remaining = list(selected)
    ordered: list[dict[str, Any]] = []
    prefix_counts = {name: Counter() for name, _feature, _weight in axes}
    while remaining:
        prefix_size = len(ordered) + 1

        def score(row: dict[str, Any]) -> tuple[float, str]:
            objective = 0.0
            for name, feature, weight in axes:
                candidate_value = feature(row)
                categories = set(ref_distributions[name]) | set(prefix_counts[name]) | {candidate_value}
                absolute = 0.0
                for category in sorted(categories):
                    observed = prefix_counts[name][category] + (category == candidate_value)
                    absolute += abs(observed / prefix_size - ref_distributions[name].get(category, 0.0))
                objective += weight * 0.5 * absolute
            return objective, stable_rank(task_id(row), seed, "order")

        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        ordered.append(chosen)
        for name, feature, _weight in axes:
            prefix_counts[name][feature(chosen)] += 1
    return ordered


def categorical_summary(rows: Sequence[Mapping[str, Any]], feature: Any) -> dict[str, Any]:
    counts = Counter(str(feature(row)) for row in rows)
    total = len(rows)
    return {
        "counts": dict(sorted(counts.items())),
        "proportions": {key: round(value / total, 8) for key, value in sorted(counts.items())},
    }


def total_variation(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    categories = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in categories
    )


def distribution_audit(
    reference: Sequence[Mapping[str, Any]],
    sample: Sequence[Mapping[str, Any]],
    *,
    cutpoints: Sequence[int],
) -> dict[str, Any]:
    features = {
        "database": lambda row: str(row.get("db_id") or ""),
        "question_length_quintile": lambda row: length_bin(question_length(row), cutpoints),
        "external_knowledge": lambda row: "present" if evidence_present(row) else "absent",
        "knowledge_x_length": lambda row: "|".join(cell_key(row, cutpoints)),
    }
    axes: dict[str, Any] = {}
    for name, feature in features.items():
        reference_counts = Counter(str(feature(row)) for row in reference)
        sample_counts = Counter(str(feature(row)) for row in sample)
        axes[name] = {
            "reference": categorical_summary(reference, feature),
            "sample": categorical_summary(sample, feature),
            "total_variation_distance": round(total_variation(reference_counts, sample_counts), 8),
        }

    reference_lengths = [question_length(row) for row in reference]
    sample_lengths = [question_length(row) for row in sample]
    return {
        "axes": axes,
        "question_length_characters": {
            "reference": {
                "mean": round(mean(reference_lengths), 4),
                "stddev": round(pstdev(reference_lengths), 4),
                "p10": quantile(sorted(reference_lengths), 0.10),
                "p50": quantile(sorted(reference_lengths), 0.50),
                "p90": quantile(sorted(reference_lengths), 0.90),
            },
            "sample": {
                "mean": round(mean(sample_lengths), 4),
                "stddev": round(pstdev(sample_lengths), 4),
                "p10": quantile(sorted(sample_lengths), 0.10),
                "p50": quantile(sorted(sample_lengths), 0.50),
                "p90": quantile(sorted(sample_lengths), 0.90),
            },
            "relative_mean_difference": round(
                abs(mean(sample_lengths) - mean(reference_lengths)) / mean(reference_lengths),
                8,
            ),
        },
    }


def build_manifest(
    *,
    reference_path: Path,
    eligible_path: Path,
    exclusion_paths: Sequence[Path],
    output_path: Path,
    reference_rows: Sequence[dict[str, Any]],
    eligible_rows: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    details: Mapping[str, Any],
    count: int,
    seed: str,
    change_date: str,
) -> dict[str, Any]:
    cutpoints = details["question_length_quintile_cutpoints_characters"]
    selected_ids = {task_id(row) for row in selected}
    excluded_sets = []
    all_excluded: set[str] = set()
    for path in exclusion_paths:
        rows = read_jsonl(path)
        ids = set(index_unique(rows, source=str(path)))
        all_excluded.update(ids)
        excluded_sets.append(
            {"path": str(path), "sha256": sha256_file(path), "records": len(rows)}
        )
    audit = distribution_audit(reference_rows, selected, cutpoints=cutpoints)
    gates = {
        "records_exactly_300": len(selected) == count == 300,
        "unique_task_ids": len(selected_ids) == len(selected),
        "overlap_with_deprecated_cohorts_is_zero": not bool(selected_ids & all_excluded),
        "all_reference_databases_covered": len({row["db_id"] for row in selected})
        == len({row["db_id"] for row in reference_rows}),
        "database_tv_at_most_0_035": audit["axes"]["database"]["total_variation_distance"] <= 0.035,
        "length_quintile_tv_at_most_0_03": audit["axes"]["question_length_quintile"]["total_variation_distance"] <= 0.03,
        "knowledge_rate_delta_at_most_0_02": audit["axes"]["external_knowledge"]["total_variation_distance"] <= 0.02,
        "joint_knowledge_length_tv_at_most_0_04": audit["axes"]["knowledge_x_length"]["total_variation_distance"] <= 0.04,
        "question_length_mean_relative_delta_at_most_0_05": audit["question_length_characters"]["relative_mean_difference"] <= 0.05,
    }
    manifest = {
        "schema_version": ALGORITHM_VERSION,
        "status": "active_baseline",
        "activated_on": change_date,
        "purpose": "representative BIRD-train evaluation baseline for new experiments",
        "reference_population": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "records": len(reference_rows),
            "unique_databases": len({row["db_id"] for row in reference_rows}),
            "meaning": "full normalized official BIRD training set; distribution target",
        },
        "eligibility_population": {
            "path": str(eligible_path),
            "sha256": sha256_file(eligible_path),
            "records": len(eligible_rows),
            "unique_databases": len({row["db_id"] for row in eligible_rows}),
            "rule": "metadata.tool_round_trip == verified in the frozen source artifact",
            "meaning": "tasks already verified executable through the current SQLite/tool harness",
        },
        "exclusions": excluded_sets,
        "selection": {
            "count": count,
            "seed": seed,
            "algorithm": (
                "minimum-one proportional database quotas against the full train population; "
                "within each database, proportional quotas over external-knowledge presence x "
                "question-length quintile; SHA-256 deterministic tie-breaking"
            ),
            "public_fields_used": list(PUBLIC_SELECTION_FIELDS),
            "forbidden_fields_not_used": list(FORBIDDEN_SELECTION_FIELDS),
            "gold_used_for_selection": False,
            "difficulty_used_for_selection": False,
            "difficulty_note": (
                "official BIRD train has no difficulty label; historical SQL-derived proxies are "
                "intentionally excluded from baseline selection"
            ),
            "question_length_quintile_cutpoints_characters": list(cutpoints),
            "output_order": (
                "deterministic representative-prefix greedy order over database, question-length, "
                "external-knowledge, and their joint public strata"
            ),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "records": len(selected),
            "unique_task_ids": len(selected_ids),
            "unique_databases": len({row["db_id"] for row in selected}),
            "overlap_with_exclusions": len(selected_ids & all_excluded),
            "task_ids_in_frozen_order": [task_id(row) for row in selected],
        },
        "distribution_audit": audit,
        "representative_prefix_audits": {
            str(prefix): distribution_audit(reference_rows, selected[:prefix], cutpoints=cutpoints)
            for prefix in DEFAULT_PREFIX_AUDITS
            if prefix <= len(selected)
        },
        "selection_details": details,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "gold_visibility": (
            "gold fields remain in task records for harness-only scoring and must never be rendered "
            "to the model or teacher"
        ),
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", default="bird-train-baseline300-v1-20260805")
    parser.add_argument("--change-date", default="2026-08-05")
    args = parser.parse_args()

    reference_rows = read_jsonl(args.reference)
    eligible_rows = read_jsonl(args.eligible)
    unverified_ids = [
        task_id(row)
        for row in eligible_rows
        if (row.get("metadata") or {}).get("tool_round_trip") != "verified"
    ]
    if unverified_ids:
        raise ValueError(
            "eligible population contains rows without verified tool round-trip status: "
            f"{unverified_ids[:5]}"
        )
    exclusion_rows = [row for path in args.exclude for row in read_jsonl(path)]
    excluded_ids = set(index_unique(exclusion_rows, source="combined exclusions")) if exclusion_rows else set()
    selected, details = select_rows(
        reference_rows,
        eligible_rows,
        excluded_ids,
        count=args.count,
        seed=args.seed,
    )
    write_jsonl_atomic(args.output, selected)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = build_manifest(
        reference_path=args.reference,
        eligible_path=args.eligible,
        exclusion_paths=args.exclude,
        output_path=args.output,
        reference_rows=reference_rows,
        eligible_rows=eligible_rows,
        selected=selected,
        details=details,
        count=args.count,
        seed=args.seed,
        change_date=args.change_date,
    )
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "output": manifest["output"],
        "distribution_audit": manifest["distribution_audit"],
        "acceptance_gates": manifest["acceptance_gates"],
        "all_acceptance_gates_passed": manifest["all_acceptance_gates_passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
