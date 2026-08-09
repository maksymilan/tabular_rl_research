#!/usr/bin/env python3
"""Freeze the public-text-only iterative-SQL v6 output-shape target gate.

The manually reviewed target/control identities below were chosen only from task identity,
database id, question, and external knowledge. Gold SQL and execution results are retained in the
emitted task records for local hidden scoring, but are never read by selection or classification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ALGORITHM_VERSION = "iterative-sql-v6-output-shape-target-gate-v1"
SEED = "iterative-sql-v6-target-gate20-controls-v1"
PUBLIC_SELECTION_FIELDS = (
    "example_id",
    "instance_id",
    "example_index",
    "db_id",
    "question",
    "external_knowledge",
)
FORBIDDEN_SELECTION_FIELDS = (
    "gold_sql",
    "query",
    "gold_exec_results",
    "gold_sql_path",
    "gold_sample",
    "gold_row_count",
)

SEPARATE_FIELD_TARGETS = (
    "bird_train_00083",
    "bird_train_00354",
    "bird_train_01150",
    "bird_train_01886",
    "bird_train_02499",
    "bird_train_02789",
)
EXPLICIT_COMBINED_CONTROLS = (
    "bird_train_00641",
    "bird_train_02747",
    "bird_train_02806",
    "bird_train_03067",
)
SINGLE_FIELD_NAME_CONTROLS = (
    "bird_train_00490",
    "bird_train_00741",
    "bird_train_01880",
    "bird_train_05354",
)
ORDINARY_CONTROL_COUNT = 6
NAME_MARKERS = (
    "full name",
    "first name",
    "last name",
    "middle name",
    "firstname",
    "lastname",
    "middlename",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def task_id(row: Mapping[str, Any]) -> str:
    value = row.get("instance_id") or row.get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row is missing instance_id/example_id")
    return value


def public_view(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in PUBLIC_SELECTION_FIELDS}


def public_text(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{task_id(row)}: missing question")
    knowledge = row.get("external_knowledge")
    return f"{question} {knowledge if isinstance(knowledge, str) else ''}".lower()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(row_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{row_id}".encode("utf-8")).hexdigest()


def index_unique(rows: Sequence[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = task_id(row)
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate task id {row_id}")
        indexed[row_id] = row
    return indexed


def validate_reviewed_categories(indexed: Mapping[str, Mapping[str, Any]]) -> None:
    for row_id in SEPARATE_FIELD_TARGETS:
        text = public_text(indexed[row_id])
        if "full name" not in text or "+" in text or not any(
            marker in text for marker in (",", " and ", "includes")
        ):
            raise ValueError(f"{row_id}: separate-field public contract drifted")
    for row_id in EXPLICIT_COMBINED_CONTROLS:
        text = public_text(indexed[row_id])
        if "full name" not in text or "+" not in text:
            raise ValueError(f"{row_id}: explicit-combined public contract drifted")
    single_pattern = re.compile(
        r"full[_ ]names?\s+(?:refers? to|=)\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:;|$)",
        re.I,
    )
    for row_id in SINGLE_FIELD_NAME_CONTROLS:
        row = indexed[row_id]
        knowledge = str(row.get("external_knowledge") or "")
        if not single_pattern.search(knowledge):
            raise ValueError(f"{row_id}: single-field-name public contract drifted")


def select_rows(
    eligible_rows: Sequence[dict[str, Any]],
    excluded_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Select the frozen target categories and deterministic ordinary controls."""
    indexed = index_unique(eligible_rows, "eligible")
    reviewed = (
        *SEPARATE_FIELD_TARGETS,
        *EXPLICIT_COMBINED_CONTROLS,
        *SINGLE_FIELD_NAME_CONTROLS,
    )
    missing = sorted(set(reviewed) - set(indexed))
    if missing:
        raise ValueError(f"reviewed ids missing from eligible source: {missing}")
    overlap = sorted(set(reviewed) & excluded_ids)
    if overlap:
        raise ValueError(f"reviewed ids overlap excluded cohorts: {overlap}")
    validate_reviewed_categories(indexed)

    ordinary_candidates: list[dict[str, Any]] = []
    reviewed_set = set(reviewed)
    for row_id, row in indexed.items():
        if row_id in excluded_ids or row_id in reviewed_set:
            continue
        text = public_text(row)
        if any(marker in text for marker in NAME_MARKERS):
            continue
        knowledge = str(row.get("external_knowledge") or "").strip().lower()
        if knowledge in {"true", "true;", "false", "false;"}:
            continue
        ordinary_candidates.append(row)

    ordinary: list[dict[str, Any]] = []
    used_databases: set[str] = set()
    for row in sorted(ordinary_candidates, key=lambda value: stable_rank(task_id(value))):
        database = str(row.get("db_id") or "")
        if not database or database in used_databases:
            continue
        ordinary.append(row)
        used_databases.add(database)
        if len(ordinary) == ORDINARY_CONTROL_COUNT:
            break
    if len(ordinary) != ORDINARY_CONTROL_COUNT:
        raise ValueError("not enough distinct-database ordinary controls")

    categories = {
        "separate_field_targets": list(SEPARATE_FIELD_TARGETS),
        "explicit_combined_controls": list(EXPLICIT_COMBINED_CONTROLS),
        "single_field_name_controls": list(SINGLE_FIELD_NAME_CONTROLS),
        "ordinary_controls": [task_id(row) for row in ordinary],
    }
    ordered_ids = [row_id for values in categories.values() for row_id in values]
    return [indexed[row_id] for row_id in ordered_ids], categories


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    eligible_path = Path(args.eligible)
    exclude_paths = [Path(value) for value in args.exclude]
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    eligible_rows = read_jsonl(eligible_path)
    excluded_ids = {
        task_id(row)
        for path in exclude_paths
        for row in read_jsonl(path)
    }
    selected, categories = select_rows(eligible_rows, excluded_ids)
    write_jsonl_atomic(output_path, selected)
    manifest = {
        "schema_version": "iterative-sql-target-gate-manifest-v1",
        "algorithm_version": ALGORITHM_VERSION,
        "seed": SEED,
        "eligible": str(eligible_path),
        "eligible_sha256": sha256_file(eligible_path),
        "excluded": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in exclude_paths
        ],
        "public_selection_fields": list(PUBLIC_SELECTION_FIELDS),
        "forbidden_selection_fields": list(FORBIDDEN_SELECTION_FIELDS),
        "gold_fields_used_for_selection": False,
        "task_count": len(selected),
        "database_count": len({str(row.get("db_id") or "") for row in selected}),
        "categories": categories,
        "category_counts": {key: len(value) for key, value in categories.items()},
        "excluded_overlap": sorted({task_id(row) for row in selected} & excluded_ids),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "ordered_task_ids": [task_id(row) for row in selected],
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
