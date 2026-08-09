#!/usr/bin/env python3
"""Create the path-remapped BIRD-dev input used by the all-NewGNN evaluator.

Only ``db_path`` is changed.  The output serialization and companion manifest are deliberately
deterministic so their hashes can be pinned before either evaluation arm is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "remote_eval_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def remap(
    source: Path,
    output: Path,
    manifest_path: Path,
    *,
    db_root: Path,
    lock_path: Path,
) -> dict[str, Any]:
    lock = load_object(lock_path)
    evaluation = lock["evaluation_input"]
    expected_databases = evaluation["databases"]

    source_hash = sha256(source)
    if source_hash != evaluation["source_sha256"]:
        raise ValueError(
            f"source evaluation hash mismatch: expected {evaluation['source_sha256']}, "
            f"got {source_hash}"
        )

    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source row {line_number} is not an object")
            rows.append(row)
    if len(rows) != evaluation["records"]:
        raise ValueError(
            f"source record count mismatch: expected {evaluation['records']}, got {len(rows)}"
        )

    db_counts: Counter[str] = Counter()
    remapped_rows: list[dict[str, Any]] = []
    remote_root = PurePosixPath(evaluation["remote_db_root"])
    for ordinal, row in enumerate(rows):
        db_id = row.get("db_id")
        if db_id not in expected_databases:
            raise ValueError(f"row {ordinal} has unexpected db_id={db_id!r}")
        if int(row.get("example_index", -1)) != ordinal:
            raise ValueError(
                f"row {ordinal} has non-canonical example_index={row.get('example_index')!r}"
            )
        original_path = row.get("db_path")
        if not isinstance(original_path, str) or Path(original_path).name != f"{db_id}.sqlite":
            raise ValueError(f"row {ordinal} has invalid source db_path={original_path!r}")
        mapped_path = str(remote_root / db_id / f"{db_id}.sqlite")
        remapped = dict(row)
        remapped["db_path"] = mapped_path
        if {key: value for key, value in remapped.items() if key != "db_path"} != {
            key: value for key, value in row.items() if key != "db_path"
        }:
            raise AssertionError(f"non-db_path field changed at row {ordinal}")
        remapped_rows.append(remapped)
        db_counts[db_id] += 1

    if set(db_counts) != set(expected_databases):
        raise ValueError(
            f"database set mismatch: expected {sorted(expected_databases)}, got {sorted(db_counts)}"
        )

    database_report: dict[str, dict[str, Any]] = {}
    for db_id, expected in sorted(expected_databases.items()):
        database_path = db_root / db_id / f"{db_id}.sqlite"
        if not database_path.is_file():
            raise ValueError(f"database file is missing: {database_path}")
        actual_hash = sha256(database_path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"database hash mismatch for {db_id}: expected {expected['sha256']}, "
                f"got {actual_hash}"
            )
        if db_counts[db_id] != expected["records"]:
            raise ValueError(
                f"database record count mismatch for {db_id}: expected {expected['records']}, "
                f"got {db_counts[db_id]}"
            )
        database_report[db_id] = {
            "relative_path": f"{db_id}/{db_id}.sqlite",
            "remote_path": str(remote_root / db_id / f"{db_id}.sqlite"),
            "records": db_counts[db_id],
            "sha256": actual_hash,
        }

    output_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in remapped_rows
    )
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    expected_derived_hash = evaluation["derived_sha256"]
    if expected_derived_hash and output_hash != expected_derived_hash:
        raise ValueError(
            f"derived evaluation hash mismatch: expected {expected_derived_hash}, got {output_hash}"
        )

    manifest: dict[str, Any] = {
        "schema_version": "qwen3-atomic-v26-remote-eval-input-v1",
        "transformation": "replace-db_path-only",
        "source_logical_name": evaluation["source_logical_name"],
        "source_sha256": source_hash,
        "derived_logical_name": evaluation["derived_logical_name"],
        "derived_sha256": output_hash,
        "records": len(remapped_rows),
        "preserved_fields": "all-except-db_path",
        "remote_db_root": str(remote_root),
        "databases": database_report,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_hash = evaluation.get("manifest_sha256")
    if expected_manifest_hash and manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"derived manifest hash mismatch: expected {expected_manifest_hash}, got {manifest_hash}"
        )

    if output.exists() and output.read_bytes() != output_bytes:
        raise ValueError(f"refusing to replace a different derived input: {output}")
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError(f"refusing to replace a different input manifest: {manifest_path}")
    atomic_write(output, output_bytes)
    atomic_write(manifest_path, manifest_bytes)
    return {**manifest, "manifest_sha256": manifest_hash}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        report = remap(
            args.source,
            args.output,
            args.manifest,
            db_root=args.db_root,
            lock_path=args.lock,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"remote evaluation input preparation failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
