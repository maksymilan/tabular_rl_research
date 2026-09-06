"""Exclude RL tasks whose reference denotation is empty or a scalar zero.

This is a conservative task-admission filter, not a reward.  It executes the hidden reference
query against the task's source database in read-only mode and records only the result shape and
classification.  Reference SQL and result values are never written to the audit artifact.
"""
from __future__ import annotations

import hashlib
import json
import numbers
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXCLUDED_REFERENCE_RESULT_KINDS = frozenset({
    "empty_rows",
    "null_scalar",
    "zero_scalar",
})


@dataclass(frozen=True)
class ReferenceResultAudit:
    task_id: str
    example_index: int
    db_id: str
    kind: str
    observed_row_count: int
    observed_column_count: int
    query_sha256: str

    @property
    def excluded(self) -> bool:
        return self.kind in EXCLUDED_REFERENCE_RESULT_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "example_index": self.example_index,
            "db_id": self.db_id,
            "kind": self.kind,
            "excluded": self.excluded,
            "observed_row_count": self.observed_row_count,
            "observed_column_count": self.observed_column_count,
            "query_sha256": self.query_sha256,
        }


def classify_reference_rows(rows: list[tuple[Any, ...]]) -> str:
    """Classify only the result shapes intentionally excluded from this process-RL run."""
    if not rows:
        return "empty_rows"
    if len(rows) == 1 and len(rows[0]) == 1:
        value = rows[0][0]
        if value is None:
            return "null_scalar"
        if isinstance(value, numbers.Number) and value == 0:
            return "zero_scalar"
    return "nonempty"


def _read_reference_rows(db_path: str | Path, query: str) -> list[tuple[Any, ...]]:
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"reference database does not exist: {resolved}")
    uri = f"{resolved.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(query)
        # Two rows are sufficient to distinguish an empty result, a single scalar, and all other
        # non-empty results without materializing a potentially large reference result.
        return [tuple(row) for row in cursor.fetchmany(2)]
    finally:
        connection.close()


def audit_task_environment(environment: dict[str, Any]) -> ReferenceResultAudit:
    query = environment.get("gold_sql")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            f"task {environment.get('task_id')!r} has no hidden reference SQL"
        )
    rows = _read_reference_rows(environment["db_path"], query)
    column_count = len(rows[0]) if rows else 0
    return ReferenceResultAudit(
        task_id=str(environment["task_id"]),
        example_index=int(environment["example_index"]),
        db_id=str(environment["db_id"]),
        kind=classify_reference_rows(rows),
        observed_row_count=len(rows),
        observed_column_count=column_count,
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
    )


def filter_training_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[ReferenceResultAudit]]:
    retained: list[dict[str, Any]] = []
    audits: list[ReferenceResultAudit] = []
    for record in records:
        audit = audit_task_environment(record["environment"])
        audits.append(audit)
        if not audit.excluded:
            retained.append(record)
    return retained, audits


def read_examples_payload(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return None, payload
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return payload, payload["examples"]
    raise ValueError(f"{path} must contain a JSON list or an object with examples")


def example_environment(example: dict[str, Any]) -> dict[str, Any]:
    query = example.get("gold_sql") or example.get("query")
    return {
        "task_id": (
            example.get("example_id")
            or example.get("instance_id")
            or f"bird_train_{int(example['example_index']):05d}"
        ),
        "example_index": int(example["example_index"]),
        "db_id": example["db_id"],
        "db_path": example["db_path"],
        "gold_sql": query,
    }
