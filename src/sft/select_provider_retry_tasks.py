#!/usr/bin/env python3
"""Select external-teacher tasks invalidated by provider-carrier response failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EMPTY_VISIBLE_CONTENT = "visible content was empty"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_retryable_provider_failure(record: dict[str, Any]) -> bool:
    legacy_protocol_failure = (
        not record.get("correct")
        and record.get("failure_type") == "protocol_error"
        and any(
            EMPTY_VISIBLE_CONTENT in str(event.get("message") or "")
            for event in record.get("error_events") or []
        )
    )
    explicit_carrier_failure = (
        not record.get("correct")
        and record.get("failure_type") == "provider_carrier_error"
    )
    explicit_api_failure = (
        not record.get("correct")
        and record.get("failure_type") == "api_error"
    )
    return legacy_protocol_failure or explicit_carrier_failure or explicit_api_failure


def source_identity(record: dict[str, Any]) -> str:
    value = record.get("example_id")
    if value is None and isinstance(record.get("task"), dict):
        value = record["task"].get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("source row is missing a non-empty example_id")
    return value


def attempt_identity(record: dict[str, Any]) -> str:
    value = record.get("example_id") or record.get("trajectory_id")
    if not isinstance(value, str) or not value:
        raise ValueError("attempt row is missing trajectory_id/example_id")
    return value


def build(source_path: Path, attempts_path: Path, out_path: Path) -> dict[str, Any]:
    source = read_jsonl(source_path)
    attempts = read_jsonl(attempts_path)
    source_by_id = {source_identity(row): row for row in source}
    if len(source_by_id) != len(source):
        raise ValueError("source tasks must have unique non-empty example_id values")

    latest_by_id: dict[str, dict[str, Any]] = {}
    for record in attempts:
        trajectory_id = attempt_identity(record)
        if trajectory_id not in source_by_id:
            raise ValueError(f"attempt references task outside source: {trajectory_id}")
        previous = latest_by_id.get(trajectory_id)
        if previous is None or int(record.get("attempt_index", 1)) > int(previous.get("attempt_index", 1)):
            latest_by_id[trajectory_id] = record

    retry_ids = {
        trajectory_id
        for trajectory_id, record in latest_by_id.items()
        if is_retryable_provider_failure(record)
    }
    selected = [row for row in source if source_identity(row) in retry_ids]
    write_jsonl_atomic(out_path, selected)
    manifest = {
        "selection": "provider_transport_or_empty_carrier_retry",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "attempts": str(attempts_path),
        "attempts_sha256": sha256(attempts_path),
        "attempted_tasks": len(latest_by_id),
        "retry_tasks": len(selected),
        "retry_example_ids": [source_identity(row) for row in selected],
        "output": str(out_path),
        "output_sha256": sha256(out_path),
        "semantic_interpretation": (
            "provider transport/carrier incomplete; excluded from policy accuracy and eligible "
            "for a fresh causal low-concurrency attempt"
        ),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source.resolve(), args.attempts.resolve(), args.out.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
