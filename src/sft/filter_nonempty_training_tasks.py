#!/usr/bin/env python3
"""Privately remove training tasks whose hidden gold SQL has no result rows.

This is a task-admission gate, not a supervision compiler.  Gold SQL is executed locally against
the task's read-only SQLite database solely to decide whether at least one row exists.  SQL text,
result rows, and result values are never written to the audit artifacts or exposed to a teacher.

An aggregate result such as ``COUNT(*) == 0`` still contains one row and is therefore retained.
Only a zero-row denotation is classified as ``empty_result``.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sqlglot
from sqlglot import exp


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "harness"))

from select_bird_train_baseline import sha256_file, task_id  # noqa: E402
from sql_atomic_tool_profile import sql_from_task  # noqa: E402


FILTER_VERSION = "gold-denotation-nonempty-task-filter-v1"
NONEMPTY_STATUS = "nonempty"
EMPTY_STATUS = "empty_result"
EXECUTION_ERROR_STATUS = "execution_error"
INPUT_ERROR_STATUS = "input_error"
STATUS_VALUES = (
    NONEMPTY_STATUS,
    EMPTY_STATUS,
    EXECUTION_ERROR_STATUS,
    INPUT_ERROR_STATUS,
)


@dataclass(frozen=True)
class GoldResultAdmission:
    example_id: str
    db_id: str
    status: str
    error_class: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status == NONEMPTY_STATUS

    def private_audit_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "gold_result_status": self.status,
            "training_task_eligible": self.eligible,
            "teacher_visible": False,
        }
        if self.error_class:
            record["error_class"] = self.error_class
        return record


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


def _query_tree(sql: str) -> exp.Query:
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("gold SQL must contain exactly one query statement")
    return statements[0]


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError("task database does not exist")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.text_factory = lambda value: value.decode("utf-8", "replace")
    connection.execute("PRAGMA query_only = ON")
    return connection


def certify_gold_result(
    row: Mapping[str, Any],
    *,
    connection: sqlite3.Connection | None = None,
    timeout_seconds: float = 30.0,
) -> GoldResultAdmission:
    """Return a private at-least-one-row certification without retaining result data."""
    example_id = task_id(row)
    db_id = str(row.get("db_id") or "")
    owned_connection = connection is None
    try:
        sql = sql_from_task(dict(row))
        _query_tree(sql)
        active_connection = connection or _readonly_connection(
            Path(str(row.get("db_path") or ""))
        )
    except Exception as exc:  # noqa: BLE001 - persist only the exception class, never SQL/details
        return GoldResultAdmission(
            example_id=example_id,
            db_id=db_id,
            status=INPUT_ERROR_STATUS,
            error_class=type(exc).__name__,
        )

    deadline = time.monotonic() + timeout_seconds

    def interrupted() -> int:
        return int(time.monotonic() >= deadline)

    active_connection.set_progress_handler(interrupted, 10_000)
    try:
        cursor = active_connection.execute(sql)
        first_row = cursor.fetchone()
        cursor.close()
        status = EMPTY_STATUS if first_row is None else NONEMPTY_STATUS
        return GoldResultAdmission(example_id=example_id, db_id=db_id, status=status)
    except Exception as exc:  # noqa: BLE001 - audit is intentionally value/message free
        return GoldResultAdmission(
            example_id=example_id,
            db_id=db_id,
            status=EXECUTION_ERROR_STATUS,
            error_class=type(exc).__name__,
        )
    finally:
        active_connection.set_progress_handler(None, 0)
        if owned_connection:
            active_connection.close()


def certify_reference_population(
    rows: Sequence[dict[str, Any]],
    *,
    timeout_seconds: float,
) -> dict[str, GoldResultAdmission]:
    admissions: dict[str, GoldResultAdmission] = {}
    connections: dict[Path, sqlite3.Connection] = {}
    try:
        for row in rows:
            row_id = task_id(row, source="reference population")
            if row_id in admissions:
                raise ValueError(f"reference population: duplicate task id {row_id}")
            path = Path(str(row.get("db_path") or "")).expanduser().resolve()
            try:
                connection = connections.get(path)
                if connection is None:
                    connection = _readonly_connection(path)
                    connections[path] = connection
            except Exception:
                connection = None
            admissions[row_id] = certify_gold_result(
                row,
                connection=connection,
                timeout_seconds=timeout_seconds,
            )
    finally:
        for connection in connections.values():
            connection.close()
    return admissions


def _status_counts(admissions: Iterable[GoldResultAdmission]) -> dict[str, int]:
    counts = Counter(admission.status for admission in admissions)
    return {status: counts[status] for status in STATUS_VALUES}


def _safe_source_match(reference: Mapping[str, Any], eligible: Mapping[str, Any]) -> bool:
    """Compare hidden execution inputs without serializing either one."""
    return (
        str(reference.get("db_id") or "") == str(eligible.get("db_id") or "")
        and str(reference.get("db_path") or "") == str(eligible.get("db_path") or "")
        and sql_from_task(dict(reference)) == sql_from_task(dict(eligible))
    )


def build_filtered_outputs(
    *,
    reference_path: Path,
    eligible_path: Path,
    reference_output_path: Path,
    eligible_output_path: Path,
    private_audit_path: Path,
    manifest_path: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    reference_rows = read_jsonl(reference_path)
    eligible_rows = read_jsonl(eligible_path)
    reference_index = {
        task_id(row, source="reference population"): row for row in reference_rows
    }
    if len(reference_index) != len(reference_rows):
        raise ValueError("reference population contains duplicate task ids")

    eligible_ids: list[str] = []
    seen_eligible_ids: set[str] = set()
    for row in eligible_rows:
        row_id = task_id(row, source="eligible population")
        if row_id in seen_eligible_ids:
            raise ValueError(f"eligible population: duplicate task id {row_id}")
        if row_id not in reference_index:
            raise ValueError(f"eligible population: unknown task id {row_id}")
        if not _safe_source_match(reference_index[row_id], row):
            raise ValueError(f"eligible population: hidden source mismatch for {row_id}")
        eligible_ids.append(row_id)
        seen_eligible_ids.add(row_id)

    admissions = certify_reference_population(
        reference_rows,
        timeout_seconds=timeout_seconds,
    )
    reference_nonempty = [
        row for row in reference_rows if admissions[task_id(row)].eligible
    ]
    eligible_nonempty = [
        row for row in eligible_rows if admissions[task_id(row)].eligible
    ]
    private_audit = [
        admissions[task_id(row)].private_audit_record() for row in reference_rows
    ]

    write_jsonl_atomic(reference_output_path, reference_nonempty)
    write_jsonl_atomic(eligible_output_path, eligible_nonempty)
    write_jsonl_atomic(private_audit_path, private_audit)

    reference_counts = _status_counts(admissions.values())
    eligible_counts = _status_counts(admissions[row_id] for row_id in eligible_ids)
    audit_forbidden_keys = sorted(
        {
            key
            for record in private_audit
            for key in record
            if key in {"gold_sql", "query", "SQL", "gold_rows", "gold_exec_results"}
        }
    )
    gates = {
        "reference_partition_complete": sum(reference_counts.values()) == len(reference_rows),
        "eligible_partition_complete": sum(eligible_counts.values()) == len(eligible_rows),
        "reference_output_all_certified_nonempty": len(reference_nonempty)
        == reference_counts[NONEMPTY_STATUS],
        "eligible_output_all_certified_nonempty": len(eligible_nonempty)
        == eligible_counts[NONEMPTY_STATUS],
        "eligible_output_is_reference_subset": set(map(task_id, eligible_nonempty))
        <= set(map(task_id, reference_nonempty)),
        "private_audit_contains_no_gold_sql_or_rows": not audit_forbidden_keys,
    }
    manifest = {
        "schema_version": FILTER_VERSION,
        "status": "active_training_task_admission_gate",
        "criterion": (
            "retain only tasks whose hidden gold SQL executes against the immutable SQLite "
            "database and returns at least one row"
        ),
        "zero_scalar_rule": (
            "a one-row result containing value 0 is nonempty and remains eligible"
        ),
        "privacy_boundary": (
            "gold SQL and result rows/values are used only inside the local harness; the audit "
            "stores task identity, status, and redacted exception class only"
        ),
        "execution": {
            "database_mode": "sqlite_uri_mode_ro_plus_pragma_query_only",
            "query_shape": "exactly_one_sqlglot_query",
            "row_probe": "fetchone_only",
            "result_rows_persisted": False,
            "result_values_persisted": False,
            "timeout_seconds_per_task": timeout_seconds,
            "execution_errors_are_training_eligible": False,
            "source_gold_exec_results_field_used": False,
        },
        "inputs": {
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
                "records": len(reference_rows),
            },
            "eligible": {
                "path": str(eligible_path),
                "sha256": sha256_file(eligible_path),
                "records": len(eligible_rows),
            },
        },
        "status_counts": {
            "reference": reference_counts,
            "eligible": eligible_counts,
        },
        "outputs": {
            "reference_nonempty": {
                "path": str(reference_output_path),
                "sha256": sha256_file(reference_output_path),
                "records": len(reference_nonempty),
            },
            "eligible_nonempty": {
                "path": str(eligible_output_path),
                "sha256": sha256_file(eligible_output_path),
                "records": len(eligible_nonempty),
            },
            "private_status_audit": {
                "path": str(private_audit_path),
                "sha256": sha256_file(private_audit_path),
                "records": len(private_audit),
                "teacher_visible": False,
                "forbidden_fields_present": audit_forbidden_keys,
            },
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--eligible-output", type=Path, required=True)
    parser.add_argument("--private-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    manifest = build_filtered_outputs(
        reference_path=args.reference.resolve(),
        eligible_path=args.eligible.resolve(),
        reference_output_path=args.reference_output.resolve(),
        eligible_output_path=args.eligible_output.resolve(),
        private_audit_path=args.private_audit.resolve(),
        manifest_path=args.manifest.resolve(),
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "status_counts": manifest["status_counts"],
                "reference_nonempty_records": manifest["outputs"]["reference_nonempty"][
                    "records"
                ],
                "eligible_nonempty_records": manifest["outputs"]["eligible_nonempty"][
                    "records"
                ],
                "all_acceptance_gates_passed": manifest["all_acceptance_gates_passed"],
                "manifest": str(args.manifest.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
