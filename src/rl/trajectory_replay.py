#!/usr/bin/env python3
"""Path-independent replay of one authored tool program on another SQLite database.

The process reward derives local edges from the database on which the actor ran.  That is precise
about dependencies the actor *did* establish, but one database instance cannot prove that the
program contains every question constraint.  This module supplies the complementary test-suite
check: keep the authored actions fixed, execute them on schema-compatible databases, and compare
the terminal evidence relation with hidden gold SQL under the active BIRD metric.

Database generation is intentionally outside this module.  A generator can be upgraded or replaced
without changing replay or reward semantics, and stochastic generation never occurs inside an RL
optimizer step.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from denotation import compare_denotations
from executor import Harness
from rollout import execute_tool, new_ctx, overview


class CounterfactualReplayError(RuntimeError):
    """The supplied trajectory or counterfactual database cannot be audited."""


def remap_replay_handles(value: Any, handle_map: dict[str, str]) -> Any:
    """Translate recorded handles to names allocated by the current legal-only replay."""
    if isinstance(value, list):
        return [remap_replay_handles(item, handle_map) for item in value]
    if isinstance(value, dict):
        return {key: remap_replay_handles(item, handle_map) for key, item in value.items()}
    if not isinstance(value, str) or not handle_map:
        return value
    if value in handle_map:
        return handle_map[value]
    rendered = value
    for recorded, replayed in sorted(handle_map.items(), key=lambda item: -len(item[0])):
        rendered = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(recorded)}(?=\.)",
            replayed,
            rendered,
        )
    return rendered


def _jsonable_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    return {"type": type(value).__name__, "value": value}


def bird_set_fingerprint(rows: Iterable[Iterable[Any]]) -> str:
    """Stable digest with the same row-order/duplicate invariance as ``bird-set``."""
    unique_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        encoded = [_jsonable_cell(cell) for cell in row]
        key = json.dumps(encoded, ensure_ascii=False, sort_keys=True, default=str)
        unique_rows[key] = encoded
    payload = [unique_rows[key] for key in sorted(unique_rows)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def sqlite_schema_fingerprint(db_path: str | Path) -> str:
    """Hash table/column/FK structure while ignoring rows and generated indexes."""
    connection = sqlite3.connect(str(db_path))
    try:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        schema = []
        for table in table_names:
            quoted = '"' + table.replace('"', '""') + '"'
            schema.append({
                "table": table,
                "columns": [list(row) for row in connection.execute(f"PRAGMA table_info({quoted})")],
                "foreign_keys": [
                    list(row) for row in connection.execute(f"PRAGMA foreign_key_list({quoted})")
                ],
            })
    finally:
        connection.close()
    return hashlib.sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CounterfactualReplayResult:
    database_path: str
    correct: bool
    terminal_evidence: str | None
    predicted_row_count: int
    gold_row_count: int
    predicted_fingerprint: str | None
    gold_fingerprint: str | None
    predicted_sample: list[list[Any]]
    gold_sample: list[list[Any]]
    execution_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualSuiteResult:
    trajectory_id: str
    passed: bool
    reason: str
    databases: int
    informative_databases: int
    failures: int
    execution_errors: int
    results: list[CounterfactualReplayResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        return payload


def replay_terminal_evidence(
    trajectory: dict[str, Any],
    database_path: str | Path,
    *,
    denotation_comparison: str = "bird-set",
) -> CounterfactualReplayResult:
    """Execute legal actions on ``database_path`` and score only the cited evidence relation."""
    if denotation_comparison != "bird-set":
        raise ValueError("active counterfactual replay requires denotation_comparison='bird-set'")
    source = trajectory.get("source") or {}
    gold_sql = source.get("gold_sql")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise CounterfactualReplayError("trajectory source has no gold_sql")

    path = str(Path(database_path).resolve())
    harness = Harness(path)
    terminal_evidence: str | None = None
    try:
        try:
            gold_rows = harness.gold(gold_sql)
        except Exception as exc:  # gold failure means the generated database is invalid
            raise CounterfactualReplayError(
                f"gold SQL failed on counterfactual database {path}: {type(exc).__name__}: {exc}"
            ) from exc

        ctx = new_ctx(overview(harness))
        created: set[str] = set()
        handle_map: dict[str, str] = {}
        terminal_seen = False
        for step in trajectory.get("steps") or []:
            call = step.get("tool_call") or {}
            tool = call.get("tool")
            authored_arguments = call.get("arguments")
            if not isinstance(tool, str) or not isinstance(authored_arguments, dict):
                raise CounterfactualReplayError(
                    f"malformed legal step {step.get('step_id')!r}"
                )
            arguments = remap_replay_handles(deepcopy(authored_arguments), handle_map)
            if tool == "answer_from_context":
                terminal_seen = True
                evidence = arguments.get("evidence")
                if (
                    not isinstance(evidence, dict)
                    or set(evidence) != {"table"}
                    or not isinstance(evidence.get("table"), str)
                ):
                    raise CounterfactualReplayError(
                        "counterfactual completeness requires a terminal evidence table"
                    )
                terminal_evidence = evidence["table"]
                try:
                    predicted_rows = harness.rows(terminal_evidence)
                except Exception as exc:
                    return CounterfactualReplayResult(
                        database_path=path,
                        correct=False,
                        terminal_evidence=terminal_evidence,
                        predicted_row_count=0,
                        gold_row_count=len(gold_rows),
                        predicted_fingerprint=None,
                        gold_fingerprint=bird_set_fingerprint(gold_rows),
                        predicted_sample=[],
                        gold_sample=[list(row) for row in gold_rows[:5]],
                        execution_error=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                return CounterfactualReplayResult(
                    database_path=path,
                    correct=compare_denotations(
                        predicted_rows,
                        gold_rows,
                        denotation_comparison,
                    ),
                    terminal_evidence=terminal_evidence,
                    predicted_row_count=len(predicted_rows),
                    gold_row_count=len(gold_rows),
                    predicted_fingerprint=bird_set_fingerprint(predicted_rows),
                    gold_fingerprint=bird_set_fingerprint(gold_rows),
                    predicted_sample=[list(row) for row in predicted_rows[:5]],
                    gold_sample=[list(row) for row in gold_rows[:5]],
                )

            try:
                output, replayed_handle = execute_tool(
                    harness,
                    tool,
                    arguments,
                    ctx,
                    str(step.get("step_id") or ""),
                )
            except Exception as exc:
                return CounterfactualReplayResult(
                    database_path=path,
                    correct=False,
                    terminal_evidence=None,
                    predicted_row_count=0,
                    gold_row_count=len(gold_rows),
                    predicted_fingerprint=None,
                    gold_fingerprint=bird_set_fingerprint(gold_rows),
                    predicted_sample=[],
                    gold_sample=[list(row) for row in gold_rows[:5]],
                    execution_error=f"{type(exc).__name__}: {exc}",
                )
            if replayed_handle:
                created.add(replayed_handle)
                recorded_handle = (step.get("tool_output") or {}).get("table")
                if isinstance(recorded_handle, str):
                    handle_map[recorded_handle] = replayed_handle

        if not terminal_seen:
            raise CounterfactualReplayError("trajectory has no terminal answer_from_context step")
        raise AssertionError("unreachable")
    finally:
        harness.conn.close()


def evaluate_counterfactual_suite(
    trajectory: dict[str, Any],
    database_paths: Iterable[str | Path],
    *,
    min_informative_databases: int = 1,
    require_schema_match: bool = True,
    denotation_comparison: str = "bird-set",
) -> CounterfactualSuiteResult:
    """Require equivalence on every database plus enough non-vacuous gold denotations."""
    if min_informative_databases < 1:
        raise ValueError("min_informative_databases must be positive")
    source = trajectory.get("source") or {}
    source_db = source.get("db_path")
    gold_sql = source.get("gold_sql")
    if not source_db or not gold_sql:
        raise CounterfactualReplayError("trajectory source must contain db_path and gold_sql")
    source_schema = sqlite_schema_fingerprint(source_db)
    source_harness = Harness(str(source_db))
    try:
        original_gold = source_harness.gold(str(gold_sql))
    finally:
        source_harness.conn.close()
    original_gold_fingerprint = bird_set_fingerprint(original_gold)

    results: list[CounterfactualReplayResult] = []
    informative = 0
    for database_path in database_paths:
        if require_schema_match and sqlite_schema_fingerprint(database_path) != source_schema:
            raise CounterfactualReplayError(
                f"counterfactual database schema differs from source: {database_path}"
            )
        result = replay_terminal_evidence(
            trajectory,
            database_path,
            denotation_comparison=denotation_comparison,
        )
        results.append(result)
        if (
            not result.correct
            or result.gold_fingerprint != original_gold_fingerprint
        ):
            informative += 1

    failures = sum(not result.correct for result in results)
    execution_errors = sum(result.execution_error is not None for result in results)
    if not results:
        passed = False
        reason = "empty_suite"
    elif failures:
        passed = False
        reason = "counterexample_found"
    elif informative < min_informative_databases:
        passed = False
        reason = "insufficient_informative_databases"
    else:
        passed = True
        reason = "passed"
    return CounterfactualSuiteResult(
        trajectory_id=str(trajectory.get("trajectory_id") or "unknown"),
        passed=passed,
        reason=reason,
        databases=len(results),
        informative_databases=informative,
        failures=failures,
        execution_errors=execution_errors,
        results=results,
    )
