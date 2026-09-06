#!/usr/bin/env python3
"""Build two deterministic, schema-preserving counterfactual DBs per solvable pool task."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import sqlglot
from sqlglot import exp


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]
from rl.runtime.trajectory_replay import bird_set_fingerprint, sqlite_schema_fingerprint  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def execute_gold(path: Path, sql: str) -> tuple[list[list[Any]], str]:
    connection = sqlite3.connect(str(path))
    try:
        rows = [list(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()
    return rows, bird_set_fingerprint(rows)


def source_tables_and_columns(
    connection: sqlite3.Connection,
    sql: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    tree = sqlglot.parse_one(sql, read="sqlite")
    actual = {
        str(row[0]).lower(): str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    aliases: dict[str, str] = {}
    tables = []
    for table in tree.find_all(exp.Table):
        resolved = actual.get(table.name.lower())
        if resolved is None:
            continue
        if resolved not in tables:
            tables.append(resolved)
        aliases[table.alias_or_name.lower()] = resolved
        aliases[table.name.lower()] = resolved
    candidates = []
    for column in tree.find_all(exp.Column):
        if column.table:
            targets = [aliases.get(column.table.lower())]
        else:
            targets = tables
        for table in targets:
            if table is None:
                continue
            columns = {
                str(row[1]).lower(): str(row[1])
                for row in connection.execute(f"PRAGMA table_info({quote(table)})")
            }
            resolved_column = columns.get(column.name.lower())
            if resolved_column is not None and (table, resolved_column) not in candidates:
                candidates.append((table, resolved_column))
    if not candidates:
        for table in tables:
            for row in connection.execute(f"PRAGMA table_info({quote(table)})"):
                if not bool(row[5]):
                    candidates.append((table, str(row[1])))
                    break
    return tables, candidates


def column_mutations(
    connection: sqlite3.Connection,
    candidates: list[tuple[str, str]],
) -> Iterable[tuple[str, str]]:
    for table, column in candidates:
        info = next(
            row
            for row in connection.execute(f"PRAGMA table_info({quote(table)})")
            if str(row[1]) == column
        )
        declared = str(info[2] or "").upper()
        for variant in (1, 2):
            if any(token in declared for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
                expression = f"{quote(column)} + {1000003 * variant}"
                kind = "numeric_column_shift"
            else:
                suffix = f" [CF-{variant}]".replace("'", "''")
                expression = f"CAST({quote(column)} AS TEXT) || '{suffix}'"
                kind = "text_column_shift"
            sql = (
                f"UPDATE {quote(table)} SET {quote(column)} = {expression} "
                f"WHERE {quote(column)} IS NOT NULL"
            )
            yield kind, sql


def deletion_mutations(
    connection: sqlite3.Connection,
    tables: list[str],
) -> Iterable[tuple[str, str]]:
    for table in tables:
        count = int(connection.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0])
        if count <= 0:
            continue
        block = max(1, min(count, count // 10 or 1))
        offsets = sorted({0, max(0, (count - block) // 2), max(0, count - block)})
        for offset in offsets:
            sql = (
                f"DELETE FROM {quote(table)} WHERE rowid IN ("
                f"SELECT rowid FROM {quote(table)} ORDER BY rowid LIMIT {block} OFFSET {offset})"
            )
            yield "row_block_delete", sql
        if count >= 5:
            for remainder in (0, 1):
                yield (
                    "row_stride_delete",
                    f"DELETE FROM {quote(table)} WHERE ABS(rowid) % 5 = {remainder}",
                )


def value_group_deletion_mutations(
    connection: sqlite3.Connection,
    candidates: list[tuple[str, str]],
    *,
    limit_per_column: int = 32,
) -> Iterable[tuple[str, str]]:
    """Delete one frequent referenced-column value at a time.

    Row-id blocks and strides can preserve the winner of a grouped aggregate because they
    perturb every group nearly uniformly.  Removing a complete value group is still a
    deterministic, schema-preserving database perturbation, but it can expose alternate
    top-k/group-by denotations without consulting the gold result.
    """
    for table, column in candidates:
        rows = connection.execute(
            f"SELECT quote({quote(column)}), COUNT(*) AS frequency "
            f"FROM {quote(table)} WHERE {quote(column)} IS NOT NULL "
            f"GROUP BY {quote(column)} "
            f"ORDER BY frequency DESC, typeof({quote(column)}), "
            f"CAST({quote(column)} AS TEXT) LIMIT ?",
            (limit_per_column,),
        ).fetchall()
        literals = [literal for literal, _frequency in rows if literal is not None]
        for literal, _frequency in rows:
            if literal is None:
                continue
            yield (
                "referenced_value_group_delete",
                f"DELETE FROM {quote(table)} WHERE {quote(column)} IS {literal}",
            )
        for prefix_size in (2, 4, 8, 16, 32):
            if prefix_size > len(literals):
                continue
            yield (
                "referenced_value_group_prefix_delete",
                f"DELETE FROM {quote(table)} WHERE {quote(column)} IN "
                f"({', '.join(literals[:prefix_size])})",
            )


def _existing_counterfactuals(
    *,
    task_dir: Path,
    source_fingerprint: str,
    source_schema: str,
    gold_sql: str,
    prior_attempts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    outputs = [
        task_dir / "counterfactual_1.sqlite",
        task_dir / "counterfactual_2.sqlite",
    ]
    if outputs[1].exists() and not outputs[0].exists():
        raise ValueError("counterfactual_2.sqlite exists without counterfactual_1.sqlite")

    accepted_attempts = [row for row in prior_attempts if row.get("status") == "accepted"]
    accepted: list[dict[str, Any]] = []
    seen_fingerprints = {source_fingerprint}
    for index, path in enumerate(outputs):
        if not path.exists():
            break
        rows, fingerprint = execute_gold(path, gold_sql)
        schema = sqlite_schema_fingerprint(path)
        if not rows:
            raise ValueError(f"existing counterfactual has empty gold result: {path}")
        if schema != source_schema:
            raise ValueError(f"existing counterfactual changed schema: {path}")
        if fingerprint in seen_fingerprints:
            raise ValueError(f"existing counterfactual is not informative or unique: {path}")
        attempt = accepted_attempts[index] if index < len(accepted_attempts) else {}
        accepted.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "schema_sha256": schema,
                "gold_result_sha256": fingerprint,
                "gold_result_rows": len(rows),
                "mutation_kind": attempt.get("kind", "reused_existing"),
                "mutation_sql_sha256": attempt.get("mutation_sql_sha256"),
            }
        )
        seen_fingerprints.add(fingerprint)
    return accepted, seen_fingerprints


def build_task_suite(
    task: dict[str, Any],
    task_dir: Path,
) -> dict[str, Any]:
    source = Path(task["db_path"]).resolve()
    gold_sql = str(task.get("gold_sql") or task.get("query"))
    source_rows, source_fingerprint = execute_gold(source, gold_sql)
    if not source_rows:
        raise ValueError("source gold result is empty")
    source_schema = sqlite_schema_fingerprint(source)
    connection = sqlite3.connect(str(source))
    try:
        tables, columns = source_tables_and_columns(connection, gold_sql)
        mutations = list(column_mutations(connection, columns))
        mutations.extend(deletion_mutations(connection, tables))
        mutations.extend(value_group_deletion_mutations(connection, columns))
    finally:
        connection.close()
    if not tables or not mutations:
        raise ValueError("could not derive deterministic mutation candidates")

    task_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = task_dir / "generation_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    accepted, seen_fingerprints = _existing_counterfactuals(
        task_dir=task_dir,
        source_fingerprint=source_fingerprint,
        source_schema=source_schema,
        gold_sql=gold_sql,
        prior_attempts=attempts,
    )
    attempted_mutations = {
        str(row["mutation_sql_sha256"])
        for row in attempts
        if row.get("mutation_sql_sha256")
    }
    if len(accepted) == 2:
        return {
            "task_id": str(task.get("example_id") or task.get("instance_id")),
            "source_db_sha256": sha256_file(source),
            "source_schema_sha256": source_schema,
            "gold_sql_sha256": sha256_text(gold_sql),
            "source_gold_result_sha256": source_fingerprint,
            "source_gold_result_rows": len(source_rows),
            "min_informative_databases": 2,
            "databases": accepted,
        }
    for attempt_index, (kind, mutation_sql) in enumerate(mutations):
        mutation_sql_sha256 = sha256_text(mutation_sql)
        if mutation_sql_sha256 in attempted_mutations:
            continue
        candidate = task_dir / "candidate.sqlite"
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite unfinished candidate: {candidate}")
        shutil.copy2(source, candidate)
        connection = sqlite3.connect(str(candidate))
        try:
            changed = connection.execute(mutation_sql).rowcount
            connection.commit()
        except Exception as exc:
            connection.close()
            candidate.unlink(missing_ok=True)
            attempts.append({"kind": kind, "status": "mutation_error", "error": str(exc)})
            continue
        finally:
            if connection:
                connection.close()
        try:
            rows, fingerprint = execute_gold(candidate, gold_sql)
        except Exception as exc:
            candidate.unlink(missing_ok=True)
            attempts.append({"kind": kind, "status": "gold_execution_error", "error": str(exc)})
            continue
        schema = sqlite_schema_fingerprint(candidate)
        accepted_condition = (
            changed > 0
            and bool(rows)
            and fingerprint not in seen_fingerprints
            and schema == source_schema
        )
        attempts.append(
            {
                "kind": kind,
                "status": "accepted" if accepted_condition else "rejected",
                "changed_rows": changed,
                "result_rows": len(rows),
                "result_sha256": fingerprint,
                "schema_preserved": schema == source_schema,
                "mutation_sql_sha256": mutation_sql_sha256,
            }
        )
        if not accepted_condition:
            candidate.unlink(missing_ok=True)
            continue
        target = task_dir / f"counterfactual_{len(accepted) + 1}.sqlite"
        candidate.replace(target)
        seen_fingerprints.add(fingerprint)
        accepted.append(
            {
                "path": str(target.resolve()),
                "sha256": sha256_file(target),
                "schema_sha256": schema,
                "gold_result_sha256": fingerprint,
                "gold_result_rows": len(rows),
                "mutation_kind": kind,
                "mutation_sql_sha256": mutation_sql_sha256,
            }
        )
        if len(accepted) == 2:
            break
    attempts_path.write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2) + "\n"
    )
    if len(accepted) != 2:
        raise ValueError(f"only {len(accepted)}/2 informative counterfactuals were generated")
    return {
        "task_id": str(task.get("example_id") or task.get("instance_id")),
        "source_db_sha256": sha256_file(source),
        "source_schema_sha256": source_schema,
        "gold_sql_sha256": sha256_text(gold_sql),
        "source_gold_result_sha256": source_fingerprint,
        "source_gold_result_rows": len(source_rows),
        "min_informative_databases": 2,
        "databases": accepted,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prior-grounding-audit", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="independent task-level SQLite mutation workers",
    )
    parser.add_argument(
        "--allow-partial-screening",
        action="store_true",
        help=(
            "write a partial screening manifest when some tasks cannot form two "
            "informative databases; this never qualifies as a strict passed manifest"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    tasks = {
        str(row.get("example_id") or row.get("instance_id")): row
        for row in load_jsonl(args.tasks)
    }
    trajectory_rows = load_jsonl(args.trajectories)
    correct_task_ids = sorted(
        {
            str(row["environment"]["task_id"])
            for row in trajectory_rows
            if row["sample"]["correct"]
        }
    )
    if not correct_task_ids:
        raise SystemExit("fixed pool contains no correct trajectories")
    prior = json.loads(args.prior_grounding_audit.read_text())
    grounding = prior.get("independent_grounding_edge_precision_gate") or {}
    if grounding.get("status") != "passed" or grounding.get("decided_precision") != 1.0:
        raise SystemExit("prior independent grounding gate is not a strict pass")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    failures = {}

    def build_one(item: tuple[int, str]):
        position, task_id = item
        task = tasks[task_id]
        try:
            return (
                position,
                task_id,
                build_task_suite(task, args.output_dir / "databases" / task_id),
                None,
            )
        except Exception as exc:
            return position, task_id, None, f"{type(exc).__name__}: {exc}"

    work = list(enumerate(correct_task_ids, 1))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for position, task_id, result, error in executor.map(build_one, work):
            if error is None:
                generated[task_id] = result
                payload = {"task": task_id, "position": position, "status": "passed"}
            else:
                failures[task_id] = error
                payload = {
                    "task": task_id,
                    "position": position,
                    "status": "failed",
                    "error": error,
                }
            print(json.dumps(payload), flush=True)

    audit_path = args.output_dir / "quality_audit.json"
    audit = {
        "schema_version": "fixed-pool-counterfactual-quality-gate-v1",
        "status": "passed" if not failures else "failed",
        "denotation_comparison": "bird-set",
        "tasks_requiring_suites": len(correct_task_ids),
        "tasks_with_two_informative_databases": len(generated),
        "failed_tasks": failures,
        "deterministic_completeness_gate": {
            "status": "passed" if not failures else "failed",
            "requirements": [
                "two immutable full-source copies per correct-trajectory task",
                "unchanged SQLite table/column/FK schema",
                "nonempty gold denotation distinct from source and sibling",
            ],
        },
        "independent_grounding_edge_precision_gate": {
            "status": "passed",
            "inherited_audit_path": str(args.prior_grounding_audit.resolve()),
            "inherited_audit_sha256": sha256_file(args.prior_grounding_audit),
            "reviewer": grounding.get("reviewer"),
            "decided_precision": grounding.get("decided_precision"),
            "scope_note": "same frozen process-credit and grounding extractor; no extractor change",
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    quality_status = "passed" if not failures else "screening_partial"
    manifest = {
        "schema_version": "process-counterfactual-suite-v2",
        "denotation_comparison": "bird-set",
        "tasks": {
            task_id: {
                "source_db_sha256": row["source_db_sha256"],
                "gold_sql_sha256": row["gold_sql_sha256"],
                "min_informative_databases": 2,
                "databases": row["databases"],
            }
            for task_id, row in generated.items()
        },
        "generator": {
            "name": "build_counterfactual_suite_for_fixed_pool.py",
            "strategy": (
                "schema-preserving referenced-column shift, deterministic row deletion, "
                "or referenced-value-group deletion"
            ),
            "task_workers": args.workers,
        },
        "quality_gate": {
            "status": quality_status,
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "mandatory_gates": [
                "deterministic_completeness",
                "independent_grounding_edge_precision",
            ],
            "failed_task_ids": sorted(failures),
        },
    }
    manifest_path = args.output_dir / (
        "counterfactual_suite.passed.json"
        if not failures
        else "counterfactual_suite.screening.json"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "tasks": len(generated)}, indent=2))
    if failures and not args.allow_partial_screening:
        raise SystemExit(f"counterfactual generation failed for {len(failures)} tasks")


if __name__ == "__main__":
    main()
