#!/usr/bin/env python3
"""Build and replay four frozen shortcut regressions for counterfactual-suite-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable

from trajectory_replay import (
    bird_set_fingerprint,
    evaluate_counterfactual_suite,
    sqlite_schema_fingerprint,
)


TARGETS = (
    "bird_train_00541",
    "bird_train_02918",
    "bird_train_03663",
    "bird_train_04696",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_target_trajectories(paths: list[Path]) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                trajectory_id = record.get("trajectory_id")
                if trajectory_id in TARGETS and trajectory_id not in trajectories:
                    trajectories[str(trajectory_id)] = record
    missing = sorted(set(TARGETS) - set(trajectories))
    if missing:
        raise ValueError(f"shortcut trajectories not found: {missing}")
    return trajectories


def source_gold_rows(trajectory: dict[str, Any], database: Path) -> list[list[Any]]:
    connection = sqlite3.connect(str(database))
    try:
        return [
            list(row)
            for row in connection.execute(
                str((trajectory.get("source") or {})["gold_sql"])
            ).fetchall()
        ]
    finally:
        connection.close()


def clone_row(
    connection: sqlite3.Connection,
    table: str,
    source_sql: str,
    overrides: dict[str, Any],
) -> None:
    quoted_table = '"' + table.replace('"', '""') + '"'
    columns = [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({quoted_table})")
    ]
    source = connection.execute(source_sql).fetchone()
    if source is None:
        raise RuntimeError(f"no source row available to clone in {table}")
    values = dict(zip(columns, source, strict=True))
    values.update(overrides)
    rendered_columns = ", ".join(
        '"' + column.replace('"', '""') + '"' for column in columns
    )
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO {quoted_table} ({rendered_columns}) VALUES ({placeholders})",
        tuple(values[column] for column in columns),
    )


def mutate_00541(connection: sqlite3.Connection) -> dict[str, Any]:
    cursor = connection.execute(
        """
        UPDATE Credit SET credited = 'true'
        WHERE person_id = 'nm0007064'
          AND lower(role) = 'narrator'
          AND credited = 'false'
        """
    )
    if cursor.rowcount < 1:
        raise RuntimeError("00541 credited mutation changed no rows")
    return {
        "shortcut": "omitted_role_and_credited_constraint",
        "changed_rows": cursor.rowcount,
    }


def mutate_02918(connection: sqlite3.Connection) -> dict[str, Any]:
    business_entity = connection.execute(
        """
        SELECT H.BusinessEntityID
        FROM EmployeeDepartmentHistory H
        JOIN Department D ON H.DepartmentID = D.DepartmentID
        WHERE D.Name = 'Engineering'
          AND H.StartDate >= '2007-01-01'
          AND H.StartDate <= '2007-12-31'
        ORDER BY H.BusinessEntityID LIMIT 1
        """
    ).fetchone()
    if business_entity is None:
        raise RuntimeError("02918 has no Engineering employee starting in 2007")
    business_entity_id = business_entity[0]
    new_credit_card_id = int(
        connection.execute("SELECT MAX(CreditCardID) FROM CreditCard").fetchone()[0]
    ) + 1000
    clone_row(
        connection,
        "CreditCard",
        "SELECT * FROM CreditCard ORDER BY CreditCardID LIMIT 1",
        {
            "CreditCardID": new_credit_card_id,
            "CardNumber": f"CF{new_credit_card_id:018d}",
            "ExpYear": 2008,
        },
    )
    clone_row(
        connection,
        "PersonCreditCard",
        "SELECT * FROM PersonCreditCard ORDER BY BusinessEntityID, CreditCardID LIMIT 1",
        {
            "BusinessEntityID": business_entity_id,
            "CreditCardID": new_credit_card_id,
        },
    )
    return {
        "shortcut": "omitted_credit_card_expiration_year",
        "business_entity_id": business_entity_id,
        "new_credit_card_id": new_credit_card_id,
        "new_exp_year": 2008,
    }


def mutate_03663(connection: sqlite3.Connection) -> dict[str, Any]:
    groups = connection.execute(
        """
        SELECT store_id, COUNT(*) AS n
        FROM customer WHERE active = 0
        GROUP BY store_id ORDER BY n DESC, store_id
        """
    ).fetchall()
    if len(groups) < 2:
        raise RuntimeError("03663 has fewer than two stores with inactive customers")
    winner, runner = groups[0][0], groups[1][0]
    cursor = connection.execute(
        "UPDATE customer SET active = 1 WHERE active = 0 AND store_id = ?",
        (winner,),
    )
    if cursor.rowcount < 1:
        raise RuntimeError("03663 argmax mutation changed no rows")
    return {
        "shortcut": "hardcoded_observed_argmax_store",
        "original_winner_store": winner,
        "counterfactual_winner_store": runner,
        "changed_rows": cursor.rowcount,
    }


def mutate_04696(connection: sqlite3.Connection) -> dict[str, Any]:
    word_ids = {
        word: int(wid)
        for wid, word in connection.execute(
            "SELECT wid, word FROM words WHERE word IN ('àbac', 'xinès', 'grec')"
        )
    }
    if set(word_ids) != {"àbac", "xinès", "grec"}:
        raise RuntimeError(f"04696 word IDs are incomplete: {word_ids}")
    rows = connection.execute(
        """
        SELECT w2nd, occurrences FROM biwords
        WHERE w1st = ? AND w2nd IN (?, ?)
        """,
        (word_ids["àbac"], word_ids["xinès"], word_ids["grec"]),
    ).fetchall()
    occurrences = {int(w2nd): value for w2nd, value in rows}
    if set(occurrences) != {word_ids["xinès"], word_ids["grec"]}:
        raise RuntimeError(f"04696 occurrence rows are incomplete: {occurrences}")
    connection.execute(
        "UPDATE biwords SET occurrences = ? WHERE w1st = ? AND w2nd = ?",
        (occurrences[word_ids["grec"]], word_ids["àbac"], word_ids["xinès"]),
    )
    connection.execute(
        "UPDATE biwords SET occurrences = ? WHERE w1st = ? AND w2nd = ?",
        (occurrences[word_ids["xinès"]], word_ids["àbac"], word_ids["grec"]),
    )
    return {
        "shortcut": "terminal_constant_after_argmax",
        "original_occurrences": {
            "àbac-xinès": occurrences[word_ids["xinès"]],
            "àbac-grec": occurrences[word_ids["grec"]],
        },
        "counterfactual_occurrences": {
            "àbac-xinès": occurrences[word_ids["grec"]],
            "àbac-grec": occurrences[word_ids["xinès"]],
        },
    }


MUTATIONS: dict[str, Callable[[sqlite3.Connection], dict[str, Any]]] = {
    "bird_train_00541": mutate_00541,
    "bird_train_02918": mutate_02918,
    "bird_train_03663": mutate_03663,
    "bird_train_04696": mutate_04696,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root-rewrite-from")
    parser.add_argument("--source-root-rewrite-to")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "shortcut_regression_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite {audit_path}")
    trajectories = read_target_trajectories(args.trajectory_jsonl)

    records = []
    for trajectory_id in TARGETS:
        trajectory = trajectories[trajectory_id]
        source = trajectory.get("source") or {}
        source_path_text = str(source["db_path"])
        if (
            args.source_root_rewrite_from
            and args.source_root_rewrite_to
            and source_path_text.startswith(args.source_root_rewrite_from)
        ):
            source_path_text = (
                args.source_root_rewrite_to
                + source_path_text[len(args.source_root_rewrite_from):]
            )
            source["db_path"] = source_path_text
        source_path = Path(source_path_text).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"source DB is missing for {trajectory_id}: {source_path}")
        target_path = output_dir / trajectory_id / f"{source['db_id']}.shortcut.sqlite"
        if target_path.exists():
            raise FileExistsError(f"refusing to overwrite {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        source_schema = sqlite_schema_fingerprint(source_path)
        source_rows = source_gold_rows(trajectory, source_path)
        connection = sqlite3.connect(str(target_path))
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN IMMEDIATE")
            mutation = MUTATIONS[trajectory_id](connection)
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if integrity != "ok":
            raise RuntimeError(f"integrity check failed for {target_path}: {integrity}")
        target_schema = sqlite_schema_fingerprint(target_path)
        if target_schema != source_schema:
            raise RuntimeError(f"schema changed for {target_path}")
        target_rows = source_gold_rows(trajectory, target_path)
        replay = evaluate_counterfactual_suite(
            trajectory,
            [target_path],
            min_informative_databases=1,
            denotation_comparison="bird-set",
        )
        rejected = not replay.passed and replay.reason == "counterexample_found"
        records.append(
            {
                "trajectory_id": trajectory_id,
                "source_db": str(source_path),
                "source_db_sha256": sha256_file(source_path),
                "counterfactual_db": str(target_path),
                "counterfactual_db_sha256": sha256_file(target_path),
                "schema_fingerprint": source_schema,
                "source_gold_fingerprint": bird_set_fingerprint(source_rows),
                "counterfactual_gold_fingerprint": bird_set_fingerprint(target_rows),
                "source_gold_sample": source_rows[:5],
                "counterfactual_gold_sample": target_rows[:5],
                "mutation": mutation,
                "replay": replay.to_dict(),
                "shortcut_rejected": rejected,
            }
        )
        print(
            json.dumps(
                {
                    "event": "shortcut_complete",
                    "trajectory_id": trajectory_id,
                    "shortcut_rejected": rejected,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    failed = [
        record["trajectory_id"]
        for record in records
        if not record["shortcut_rejected"]
    ]
    payload = {
        "schema_version": "process-counterfactual-shortcut-regression-audit-v1",
        "denotation_comparison": "bird-set",
        "expected_shortcuts": list(TARGETS),
        "rejected_shortcuts": sum(record["shortcut_rejected"] for record in records),
        "failed_to_reject": failed,
        "gate_passed": not failed,
        "records": records,
    }
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "gate_passed": not failed,
                "failed_to_reject": failed,
                "audit": str(audit_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
