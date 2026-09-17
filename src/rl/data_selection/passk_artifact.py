"""Reduce a raw K-sample screening artifact into a compact, mergeable snapshot.

Raw pass-k result files embed every sampled transcript and grow to multiple
gigabytes, so they stay on the screening host.  The compact snapshot keeps the
per-task screening identity that ``screened_pool`` merges by stable
``example_id`` -- task id, question digest, correct/legal counts, protocol
identity, decode parameters -- and drops the transcripts.

Raw result rows carry no stable task id; they are paired with their immutable
input task file by ``(example_index, db_id, question)``.  That mapping is
verified in both directions so a partially written or reordered result file
fails loudly instead of silently shifting observations between tasks.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl.data_selection.screened_pool import jsonl_text, question_sha256, task_id
from rl.shared.io import iter_jsonl, sha256_file


SNAPSHOT_SCHEMA_VERSION = "atomic-v26-screened-passk-snapshot-v1"

COMPACT_FIELDS = (
    "task_id",
    "source",
    "example_index",
    "db_id",
    "question_sha256",
    "n_samples",
    "attempted_samples",
    "sample_correct_count",
    "sample_legal_count",
    "failure_type",
    "stop_on_success",
    "protocol_version",
    "protocol_hash",
    "assistant_carrier",
    "temperature",
    "top_p",
    "max_tokens",
    "max_steps",
    "elapsed_seconds",
    "recorded_at_utc",
)


def _pair_key(row: Mapping[str, Any]) -> tuple[Any, Any, str]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("screening task/result row is missing question text")
    return (row.get("example_index"), row.get("db_id"), question.strip())


def build_compact_rows(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    source_label: str,
    expected_group_size: int = 8,
    protocol_version: str | None = None,
    protocol_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Pair raw result rows with input tasks and return sorted compact rows."""
    by_key: dict[tuple[Any, Any, str], Mapping[str, Any]] = {}
    for row in input_rows:
        key = _pair_key(row)
        if key in by_key:
            raise ValueError(f"duplicate input task for key {key}")
        by_key[key] = row

    rows: list[dict[str, Any]] = []
    matched: set[tuple[Any, Any, str]] = set()
    for result in result_rows:
        key = _pair_key(result)
        task = by_key.get(key)
        if task is None:
            raise ValueError(f"result row does not match any input task: {key}")
        if key in matched:
            raise ValueError(f"duplicate result row for task: {key}")
        matched.add(key)
        identifier = task_id(task)
        n_samples = int(result.get("n_samples") or 0)
        attempted = int(result.get("attempted_samples") or 0)
        if n_samples != expected_group_size or attempted != expected_group_size:
            raise ValueError(
                f"{identifier}: incomplete K={expected_group_size} screen "
                f"(n={n_samples}, attempted={attempted})"
            )
        for field, expected in (
            ("protocol_version", protocol_version),
            ("protocol_hash", protocol_hash),
        ):
            actual = result.get(field)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{identifier}: {field} {actual!r} != expected {expected!r}"
                )
        row = {
            "task_id": identifier,
            "source": source_label,
            "example_index": task.get("example_index", task.get("index")),
            "db_id": task.get("db_id"),
            "question_sha256": question_sha256(task),
            "n_samples": n_samples,
            "attempted_samples": attempted,
            "sample_correct_count": int(result.get("sample_correct_count") or 0),
            "sample_legal_count": int(result.get("sample_legal_count") or 0),
            "failure_type": result.get("failure_type"),
            "stop_on_success": result.get("stop_on_success"),
            "protocol_version": result.get("protocol_version"),
            "protocol_hash": result.get("protocol_hash"),
            "assistant_carrier": result.get("assistant_carrier"),
            "temperature": result.get("temperature"),
            "top_p": result.get("top_p"),
            "max_tokens": result.get("max_tokens"),
            "max_steps": result.get("max_steps"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "recorded_at_utc": result.get("recorded_at_utc"),
        }
        missing = [field for field in COMPACT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{identifier}: compact row missing {missing}")
        rows.append(row)

    missing_keys = set(by_key) - matched
    if missing_keys:
        raise ValueError(f"{len(missing_keys)} input tasks have no result row")
    rows.sort(key=lambda row: str(row["task_id"]))
    return rows


def _task_id_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{row['task_id']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_label: str,
    expected_group_size: int,
    input_path: Path,
    input_records: int,
    result_path: Path,
    result_sha256: str,
    upstream_manifest: Mapping[str, Any] | None = None,
    exported_at_utc: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_label": source_label,
        "records": len(rows),
        "expected_group_size": expected_group_size,
        "task_id_sha256": _task_id_sha256(rows),
        "input": {
            "path": str(input_path),
            "records": input_records,
            "sha256": sha256_file(input_path),
        },
        "result": {
            "path": str(result_path),
            "records": len(rows),
            "sha256": result_sha256,
        },
        "correct_count_histogram": _histogram(rows, "sample_correct_count"),
        "legal_count_histogram": _histogram(rows, "sample_legal_count"),
        "incomplete_screen_rows": sum(
            1
            for row in rows
            if int(row["n_samples"]) != expected_group_size
            or int(row["attempted_samples"]) != expected_group_size
        ),
    }
    if upstream_manifest is not None:
        manifest["upstream_manifest"] = dict(upstream_manifest)
    if exported_at_utc is not None:
        manifest["exported_at_utc"] = exported_at_utc
    return manifest


def _histogram(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(int(row[field]) for row in rows)
    return {str(key): counts[key] for key in sorted(counts)}


def export_snapshot(
    *,
    input_path: Path,
    result_path: Path,
    source_label: str,
    expected_group_size: int = 8,
    protocol_version: str | None = None,
    protocol_hash: str | None = None,
    upstream_manifest: Mapping[str, Any] | None = None,
    exported_at_utc: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream ``result_path`` against ``input_path`` and return rows + manifest."""
    input_rows = list(iter_jsonl(input_path))
    rows = build_compact_rows(
        input_rows=input_rows,
        result_rows=iter_jsonl(result_path),
        source_label=source_label,
        expected_group_size=expected_group_size,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
    )
    manifest = snapshot_manifest(
        rows=rows,
        source_label=source_label,
        expected_group_size=expected_group_size,
        input_path=input_path,
        input_records=len(input_rows),
        result_path=result_path,
        result_sha256=sha256_file(result_path),
        upstream_manifest=upstream_manifest,
        exported_at_utc=exported_at_utc,
    )
    return rows, manifest


def compact_jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return jsonl_text(rows)


__all__ = [
    "COMPACT_FIELDS",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_compact_rows",
    "compact_jsonl_text",
    "export_snapshot",
    "snapshot_manifest",
]
