#!/usr/bin/env python3
"""Load immutable, task-keyed counterfactual database suites for process RL."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "process-counterfactual-suite-v2"


@lru_cache(maxsize=4096)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CounterfactualTaskSuite:
    task_id: str
    database_paths: tuple[Path, ...]
    min_informative_databases: int
    source_db_sha256: str
    gold_sql_sha256: str

    def validate_task(self, metadata: dict[str, Any]) -> None:
        db_path = metadata.get("db_path")
        gold_sql = metadata.get("gold_sql")
        if not db_path or not gold_sql:
            raise ValueError(f"RL task {self.task_id!r} lacks db_path or gold_sql")
        actual_db = sha256_file(db_path)
        if actual_db != self.source_db_sha256:
            raise ValueError(
                f"counterfactual source DB hash mismatch for {self.task_id}: "
                f"{actual_db} != {self.source_db_sha256}"
            )
        actual_sql = sha256_text(str(gold_sql))
        if actual_sql != self.gold_sql_sha256:
            raise ValueError(
                f"counterfactual gold SQL hash mismatch for {self.task_id}: "
                f"{actual_sql} != {self.gold_sql_sha256}"
            )


@dataclass(frozen=True)
class CounterfactualSuiteManifest:
    path: Path
    tasks: dict[str, CounterfactualTaskSuite]
    generator: dict[str, Any]
    quality_gate: dict[str, Any]

    def suite_for(self, metadata: dict[str, Any]) -> CounterfactualTaskSuite:
        task_id = metadata.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("RL task metadata has no stable task_id")
        try:
            suite = self.tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"counterfactual manifest has no suite for {task_id!r}") from exc
        suite.validate_task(metadata)
        return suite


def _database_path(
    manifest_path: Path,
    item: str | dict[str, Any],
    *,
    task_id: str,
) -> Path:
    if isinstance(item, str):
        raw_path = item
        expected_sha = None
    elif isinstance(item, dict):
        raw_path = item.get("path")
        expected_sha = item.get("sha256")
    else:
        raise ValueError(f"invalid database entry for {task_id}: {item!r}")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"counterfactual database path is missing for {task_id}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"counterfactual database does not exist for {task_id}: {path}")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError(f"counterfactual database sha256 is required for {task_id}: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"counterfactual database hash mismatch for {task_id}: "
            f"{actual_sha} != {expected_sha}"
        )
    return path


def load_counterfactual_suite_manifest(path: str | Path) -> CounterfactualSuiteManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"counterfactual manifest schema_version must be {SCHEMA_VERSION!r}"
        )
    if payload.get("denotation_comparison") != "bird-set":
        raise ValueError("counterfactual manifest must use denotation_comparison='bird-set'")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise ValueError("counterfactual manifest tasks must be a non-empty object")

    tasks: dict[str, CounterfactualTaskSuite] = {}
    for task_id, raw in raw_tasks.items():
        if not isinstance(task_id, str) or not isinstance(raw, dict):
            raise ValueError("counterfactual task entries must be task-id/object pairs")
        database_items = raw.get("databases")
        if not isinstance(database_items, list) or not database_items:
            raise ValueError(f"counterfactual suite for {task_id} has no databases")
        minimum = raw.get("min_informative_databases", 1)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError(
                f"min_informative_databases must be positive for {task_id}"
            )
        source_sha = raw.get("source_db_sha256")
        gold_sha = raw.get("gold_sql_sha256")
        if not isinstance(source_sha, str) or not source_sha:
            raise ValueError(f"source_db_sha256 is required for {task_id}")
        if not isinstance(gold_sha, str) or not gold_sha:
            raise ValueError(f"gold_sql_sha256 is required for {task_id}")
        tasks[task_id] = CounterfactualTaskSuite(
            task_id=task_id,
            database_paths=tuple(
                _database_path(manifest_path, item, task_id=task_id)
                for item in database_items
            ),
            min_informative_databases=minimum,
            source_db_sha256=source_sha,
            gold_sql_sha256=gold_sha,
        )
    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("counterfactual manifest must record generator metadata")
    quality_gate = payload.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("status") != "passed":
        raise ValueError(
            "counterfactual manifest quality_gate.status must be 'passed'"
        )
    audit_sha = quality_gate.get("audit_sha256")
    if (
        not isinstance(audit_sha, str)
        or len(audit_sha) != 64
        or any(character not in "0123456789abcdef" for character in audit_sha)
    ):
        raise ValueError(
            "counterfactual manifest quality_gate.audit_sha256 must be a lowercase SHA-256"
        )
    return CounterfactualSuiteManifest(
        path=manifest_path,
        tasks=tasks,
        generator=generator,
        quality_gate=quality_gate,
    )
