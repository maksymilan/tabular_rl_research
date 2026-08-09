#!/usr/bin/env python3
"""Audit and optionally rewrite the released TRUST-SQL RL dataset.

The released JSONL embeds the authors' private database roots.  This script maps
those roots to local BIRD and Spider database trees while preserving every other
field.  Gold SQL and schema labels remain reward-only fields and are never copied
into the model-visible prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


EXPECTED_RECORDS = 11_642
EXPECTED_BIRD_RECORDS = 6_948
EXPECTED_SPIDER_RECORDS = 4_694
EXPECTED_BIRD_DATABASES = 69
EXPECTED_SPIDER_DATABASES = 146
SQL_TIMEOUT_ERROR = "OperationalError:interrupted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_family(record: dict[str, Any]) -> str:
    source_root = str(record["reward_model"]["database"]).lower()
    if "/bird/" in source_root:
        return "bird"
    if "/spider" in source_root:
        return "spider"
    raise ValueError(f"unrecognized released database root: {source_root!r}")


def local_database_path(root: Path, db_id: str) -> Path:
    return root / db_id / f"{db_id}.sqlite"


def actor_prompt_text(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("prompt must be a list of chat messages")
    parts: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError("every prompt message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("released actor prompt may contain only system/user text")
        parts.append(content)
    return "\n".join(parts)


def database_catalog(db_path: Path) -> dict[str, set[str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        actual_tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        ]
        catalog: dict[str, set[str]] = {}
        for table in actual_tables:
            quoted = str(table).replace('"', '""')
            catalog[table.lower()] = {
                str(row[1]).lower()
                for row in connection.execute(f'PRAGMA table_info("{quoted}")')
            }
        return catalog
    finally:
        connection.close()


def schema_issues(record: dict[str, Any], catalog: dict[str, set[str]]) -> list[str]:
    expected = record["reward_model"]["ground_truth"].get("schema") or {}
    tables = expected.get("tables") or []
    columns = expected.get("columns") or {}
    issues: list[str] = []
    for table in tables:
        if str(table).lower() not in catalog:
            issues.append(f"missing_table:{table}")
    for table, expected_columns in columns.items():
        actual_columns = catalog.get(str(table).lower(), set())
        for column in expected_columns:
            if str(column).lower() not in actual_columns:
                issues.append(f"missing_column:{table}.{column}")
    return issues


def execute_gold(record: dict[str, Any], db_path: Path, timeout: float) -> str | None:
    target = record["reward_model"]["ground_truth"].get("target")
    sql = target[0] if isinstance(target, list) and target else target
    if not isinstance(sql, str) or not sql.strip():
        return "missing_gold_sql"
    started = time.monotonic()
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")

        def interrupt_if_timed_out() -> int:
            return int(time.monotonic() - started > timeout)

        connection.set_progress_handler(interrupt_if_timed_out, 10_000)
        connection.execute(sql).fetchmany(31)
        return None
    except sqlite3.Error as exc:
        return f"{type(exc).__name__}:{exc}"
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bird-database-root", type=Path, required=True)
    parser.add_argument("--spider-database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute-gold", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sql-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and args.output.resolve() == args.input.resolve():
        raise ValueError("refusing to overwrite the released source JSONL")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    records: list[dict[str, Any]] = []
    families: Counter[str] = Counter()
    databases: dict[str, set[str]] = {"bird": set(), "spider": set()}
    ids: set[Any] = set()
    missing_databases: list[str] = []
    prompt_gold_leaks: list[Any] = []
    schema_failures: dict[str, list[str]] = {}
    db_paths: list[Path] = []

    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            record_id = record.get("id")
            if record_id in ids:
                raise ValueError(f"duplicate record id at line {line_number}: {record_id!r}")
            ids.add(record_id)
            family = dataset_family(record)
            root = (
                args.bird_database_root
                if family == "bird"
                else args.spider_database_root
            )
            db_id = str(record["reward_model"]["data_source"])
            db_path = local_database_path(root, db_id)
            families[family] += 1
            databases[family].add(db_id)
            db_paths.append(db_path)
            if not db_path.is_file() or db_path.stat().st_size == 0:
                missing_databases.append(f"{family}:{db_id}")
            prompt_text = actor_prompt_text(record)
            target = record["reward_model"]["ground_truth"].get("target")
            target_sqls = target if isinstance(target, list) else [target]
            if any(
                isinstance(sql, str) and sql.strip() and sql.strip() in prompt_text
                for sql in target_sqls
            ):
                prompt_gold_leaks.append(record_id)
            records.append(record)

    observed = {
        "records": len(records),
        "bird_records": families["bird"],
        "spider_records": families["spider"],
        "bird_databases": len(databases["bird"]),
        "spider_databases": len(databases["spider"]),
    }
    expected = {
        "records": EXPECTED_RECORDS,
        "bird_records": EXPECTED_BIRD_RECORDS,
        "spider_records": EXPECTED_SPIDER_RECORDS,
        "bird_databases": EXPECTED_BIRD_DATABASES,
        "spider_databases": EXPECTED_SPIDER_DATABASES,
    }
    if observed != expected:
        raise ValueError(f"released dataset counts changed: {observed} != {expected}")
    if missing_databases:
        raise FileNotFoundError(
            f"{len(missing_databases)} released database ids are unresolved; "
            f"first={missing_databases[:5]}"
        )
    if prompt_gold_leaks:
        raise ValueError(
            f"gold SQL appeared in {len(prompt_gold_leaks)} actor prompts; "
            f"first ids={prompt_gold_leaks[:5]}"
        )

    unique_db_paths = {
        (family, db_id): local_database_path(
            args.bird_database_root if family == "bird" else args.spider_database_root,
            db_id,
        )
        for family, ids_for_family in databases.items()
        for db_id in ids_for_family
    }
    catalogs = {
        key: database_catalog(db_path)
        for key, db_path in sorted(unique_db_paths.items())
    }
    for record in records:
        family = dataset_family(record)
        db_id = str(record["reward_model"]["data_source"])
        issues = schema_issues(record, catalogs[(family, db_id)])
        if issues:
            schema_failures[str(record.get("id"))] = issues

    gold_execution_failures: dict[str, str] = {}
    if args.execute_gold:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(execute_gold, record, db_path, args.sql_timeout_seconds): index
                for index, (record, db_path) in enumerate(zip(records, db_paths, strict=True))
            }
            for future in as_completed(futures):
                index = futures[future]
                error = future.result()
                if error is not None:
                    gold_execution_failures[str(records[index].get("id"))] = error

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for record in records:
                family = dataset_family(record)
                local_root = (
                    args.bird_database_root
                    if family == "bird"
                    else args.spider_database_root
                )
                rewritten = json.loads(json.dumps(record))
                rewritten["reward_model"]["database"] = str(local_root.resolve())
                stream.write(json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

    record_meta = {
        str(record.get("id")): {
            "family": dataset_family(record),
            "db_id": str(record["reward_model"]["data_source"]),
        }
        for record in records
    }
    schema_by_family = Counter(
        record_meta[record_id]["family"] for record_id in schema_failures
    )
    schema_by_database = Counter(
        f"{record_meta[record_id]['family']}:{record_meta[record_id]['db_id']}"
        for record_id in schema_failures
    )
    gold_by_family = Counter(
        record_meta[record_id]["family"] for record_id in gold_execution_failures
    )
    gold_by_database = Counter(
        f"{record_meta[record_id]['family']}:{record_meta[record_id]['db_id']}"
        for record_id in gold_execution_failures
    )
    gold_by_error_type = Counter(
        error.split(":", 1)[0] for error in gold_execution_failures.values()
    )
    gold_timeout_failures = {
        record_id: error
        for record_id, error in gold_execution_failures.items()
        if error == SQL_TIMEOUT_ERROR
    }
    gold_non_timeout_failures = {
        record_id: error
        for record_id, error in gold_execution_failures.items()
        if error != SQL_TIMEOUT_ERROR
    }
    timeout_by_family = Counter(
        record_meta[record_id]["family"] for record_id in gold_timeout_failures
    )
    timeout_by_database = Counter(
        f"{record_meta[record_id]['family']}:{record_meta[record_id]['db_id']}"
        for record_id in gold_timeout_failures
    )
    non_timeout_by_family = Counter(
        record_meta[record_id]["family"] for record_id in gold_non_timeout_failures
    )
    non_timeout_by_database = Counter(
        f"{record_meta[record_id]['family']}:{record_meta[record_id]['db_id']}"
        for record_id in gold_non_timeout_failures
    )

    audit = {
        "status": "passed" if not schema_failures and not gold_execution_failures else "completed_with_label_issues",
        "source_jsonl": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "counts": observed,
        "database_coverage": {
            "resolved": len(unique_db_paths),
            "expected": EXPECTED_BIRD_DATABASES + EXPECTED_SPIDER_DATABASES,
            "bird_root": str(args.bird_database_root.resolve()),
            "spider_root": str(args.spider_database_root.resolve()),
        },
        "actor_prompt_gold_sql_exact_match_leaks": 0,
        "schema_catalog_issue_count": len(schema_failures),
        "schema_catalog_issue_summary": {
            "by_family": dict(sorted(schema_by_family.items())),
            "affected_databases": len(schema_by_database),
            "by_database": dict(sorted(schema_by_database.items())),
        },
        "schema_catalog_issues": schema_failures,
        "gold_execution_checked": bool(args.execute_gold),
        "gold_execution_failure_count": len(gold_execution_failures),
        "gold_execution_timeout_count": len(gold_timeout_failures),
        "gold_execution_non_timeout_failure_count": len(gold_non_timeout_failures),
        "gold_execution_failure_summary": {
            "by_family": dict(sorted(gold_by_family.items())),
            "affected_databases": len(gold_by_database),
            "by_database": dict(sorted(gold_by_database.items())),
            "by_error_type": dict(sorted(gold_by_error_type.items())),
            "timeout": {
                "seconds": args.sql_timeout_seconds,
                "count": len(gold_timeout_failures),
                "by_family": dict(sorted(timeout_by_family.items())),
                "affected_databases": len(timeout_by_database),
                "by_database": dict(sorted(timeout_by_database.items())),
                "interpretation": (
                    "The read-only SQLite progress handler interrupted these queries. "
                    "Membership may vary with host load and worker count; they are not "
                    "classified as invalid SQL."
                ),
            },
            "non_timeout_failures": {
                "count": len(gold_non_timeout_failures),
                "by_family": dict(sorted(non_timeout_by_family.items())),
                "affected_databases": len(non_timeout_by_database),
                "by_database": dict(sorted(non_timeout_by_database.items())),
            },
            "overlap_with_schema_catalog_issues": len(
                set(gold_execution_failures) & set(schema_failures)
            ),
        },
        "gold_execution_failures": gold_execution_failures,
        "pass_rate_filter_audit": {
            "status": "not_recomputable_from_release",
            "reason": "the released retained JSONL contains no eight-rollout pass-rate sidecar",
        },
        "rewritten_jsonl": str(args.output.resolve()) if args.output else None,
        "rewritten_sha256": sha256_file(args.output) if args.output else None,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    display_audit = {
        key: value
        for key, value in audit.items()
        if key not in {"schema_catalog_issues", "gold_execution_failures"}
    }
    print(json.dumps(display_audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
