#!/usr/bin/env python3
"""Merge disjoint evaluation slices into one task-ordered, audited artifact.

The merger does not reinterpret records or rerun a model. It verifies that every
source manifest has the same semantic/runtime configuration, checks each source
against its declared task range, rejects duplicate examples, and writes records
in the exact order of the frozen task file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SLICE_FIELDS = {"task_start", "task_count", "config_sha256", "created_at_utc"}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_manifest(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key not in SLICE_FIELDS}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def summarize(records: list[dict], sources: list[dict]) -> dict:
    total = len(records)
    correct = sum(bool(record.get("correct")) for record in records)
    failure_types = Counter(
        record.get("failure_type") or record.get("fail") or "wrong_answer"
        for record in records
        if not record.get("correct")
    )
    return {
        "total": total,
        "correct": correct,
        "failed": total - correct,
        "accuracy": correct / total if total else 0.0,
        "completed_example_indices": [record["example_index"] for record in records],
        "failure_types": dict(sorted(failure_types.items())),
        "legal_answers": sum(bool(record.get("legal")) for record in records),
        "average_steps": (
            sum(int(record.get("steps", 0)) for record in records) / total
            if total
            else 0.0
        ),
        "total_tool_errors": sum(int(record.get("errors", 0)) for record in records),
        "composition_sources": sources,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--source-dir", action="append", required=True)
    args = parser.parse_args()

    tasks_path = Path(args.tasks_json)
    tasks = load_jsonl(tasks_path)
    tasks_sha256 = file_sha256(tasks_path)
    ordered_indices = [int(task["example_index"]) for task in tasks]
    if len(ordered_indices) != len(set(ordered_indices)):
        raise ValueError("task file contains duplicate example_index values")

    source_dirs = [Path(value) for value in args.source_dir]
    manifests: list[dict] = []
    records_by_index: dict[int, dict] = {}
    source_metadata: list[dict] = []
    reference_manifest: dict | None = None

    for source_dir in source_dirs:
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("tasks_sha256") != tasks_sha256:
            raise ValueError(
                f"{source_dir} task hash does not match {tasks_path}: "
                f"{manifest.get('tasks_sha256')} != {tasks_sha256}"
            )
        records = load_jsonl(source_dir / "all.jsonl")
        manifests.append(manifest)
        comparable = semantic_manifest(manifest)
        if reference_manifest is None:
            reference_manifest = comparable
        elif comparable != reference_manifest:
            differing = sorted(
                key
                for key in set(comparable) | set(reference_manifest)
                if comparable.get(key) != reference_manifest.get(key)
            )
            raise ValueError(f"source manifests differ in semantic fields: {differing}")

        start = int(manifest["task_start"])
        count = int(manifest["task_count"])
        expected = ordered_indices[start : start + count]
        actual = [int(record["example_index"]) for record in records]
        if len(actual) != count or set(actual) != set(expected):
            raise ValueError(
                f"{source_dir} does not exactly cover declared task range "
                f"[{start}, {start + count})"
            )
        for record in records:
            example_index = int(record["example_index"])
            if example_index in records_by_index:
                raise ValueError(f"duplicate example_index across sources: {example_index}")
            records_by_index[example_index] = record
        source_metadata.append(
            {
                "path": str(source_dir),
                "task_start": start,
                "task_count": count,
                "config_sha256": manifest.get("config_sha256"),
            }
        )

    missing = [index for index in ordered_indices if index not in records_by_index]
    extras = sorted(set(records_by_index) - set(ordered_indices))
    if missing or extras:
        raise ValueError(f"incomplete task coverage: missing={missing}, extras={extras}")
    ordered_records = [records_by_index[index] for index in ordered_indices]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(result_dir / "all.jsonl", ordered_records)
    write_jsonl(
        result_dir / "success.jsonl",
        [record for record in ordered_records if record.get("correct")],
    )
    write_jsonl(
        result_dir / "failure.jsonl",
        [record for record in ordered_records if not record.get("correct")],
    )

    assert reference_manifest is not None
    merged_manifest = {
        **reference_manifest,
        "task_start": 0,
        "task_count": len(tasks),
        "composition_policy": "strict-disjoint-task-order-v1",
        "composition_sources": source_metadata,
    }
    merged_manifest["config_sha256"] = canonical_hash(merged_manifest)
    merged_manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (result_dir / "manifest.json").write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_dir / "summary.json").write_text(
        json.dumps(summarize(ordered_records, source_metadata), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
