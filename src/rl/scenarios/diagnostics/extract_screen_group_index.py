#!/usr/bin/env python3
"""Extract compact, auditable outcome indexes from version26 K=8 screens.

The script never copies trajectory text.  It records only task identity and
group-level outcome counts, which is sufficient to assemble mixed/all-correct/
all-wrong task cohorts while keeping the original screen files immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "qwen3-v26-k8-screen-group-index-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id/instance_id")
    return value


def _task_maps(tasks_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_index: dict[int, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(tasks_path):
        identifier = _task_id(row)
        index = row.get("example_index")
        if type(index) is not int:
            raise ValueError(f"{identifier}: missing integer example_index")
        if index in by_index or identifier in by_id:
            raise ValueError(f"duplicate task identity: {identifier}/{index}")
        by_index[index] = row
        by_id[identifier] = row
    return by_index, by_id


def _label(correct_count: int, group_size: int) -> str:
    if correct_count == 0:
        return "all_wrong"
    if correct_count == group_size:
        return "all_correct"
    return "mixed"


def _structured_error(record: dict[str, Any]) -> bool:
    """Match the current four-level scorer using structured fields only."""
    if record.get("failure_type") == "generation_length":
        return True
    try:
        if int(record.get("errors", 0) or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    if record.get("error_events"):
        return True
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        try:
            if int(turn.get("errors", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        if turn.get("error_event") or turn.get("error_events"):
            return True
    return record.get("failure_type") == "timeout_error"


def _four_level_key(record: dict[str, Any]) -> str:
    correct = bool(record.get("correct"))
    has_errors = _structured_error(record)
    if correct:
        return "+1.0" if has_errors else "+1.5"
    return "-1.0" if has_errors else "-0.5"


def _compact(
    task: dict[str, Any],
    *,
    correct_count: int,
    legal_count: int,
    group_size: int,
    clean: bool,
    source_kind: str,
    source_path: Path,
    source_sha256: str | None = None,
    sample_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reward_histogram = {
        key: 0 for key in ("+1.5", "+1.0", "-0.5", "-1.0")
    }
    for record in sample_records or []:
        reward_histogram[_four_level_key(record)] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": _task_id(task),
        "example_index": task["example_index"],
        "db_id": task.get("db_id"),
        "group_size": group_size,
        "correct_count": correct_count,
        "legal_count": legal_count,
        "outcome_class": _label(correct_count, group_size) if clean else "contaminated",
        "clean": clean,
        "four_level_reward_histogram": reward_histogram,
        "four_level_nonconstant": sum(value > 0 for value in reward_histogram.values()) > 1,
        "structured_error_samples": sum(
            _structured_error(record) for record in (sample_records or [])
        ),
        "source_kind": source_kind,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256 or _sha256(source_path),
    }


def _passk_rows(screen_path: Path, tasks_path: Path) -> list[dict[str, Any]]:
    by_index, _ = _task_maps(tasks_path)
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    screen_sha = _sha256(screen_path)
    for row in _read_jsonl(screen_path):
        index = row.get("example_index")
        if type(index) is not int or index not in by_index or index in seen:
            raise ValueError(f"invalid/duplicate screen example_index: {index!r}")
        seen.add(index)
        attempted = row.get("attempted_samples")
        correct = row.get("sample_correct_count")
        legal = row.get("sample_legal_count")
        if not all(type(value) is int for value in (attempted, correct, legal)):
            raise ValueError(f"screen q{index}: invalid group counts")
        clean = attempted == 8 and 0 <= correct <= attempted and 0 <= legal <= attempted
        sample_records = row.get("samples")
        if not isinstance(sample_records, list) or not all(
            isinstance(sample, dict) for sample in sample_records
        ):
            raise ValueError(f"screen q{index}: missing sample records")
        output.append(
            _compact(
                by_index[index],
                correct_count=correct,
                legal_count=legal,
                group_size=attempted,
                clean=clean,
                source_kind="passk-jsonl",
                source_path=screen_path,
                source_sha256=screen_sha,
                sample_records=sample_records,
            )
        )
    return output


def _fixed_paths(plan_path: Path | None, group_dirs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    if plan_path is not None:
        plan = json.loads(plan_path.read_text())
        completed = plan.get("completed", {}).get("groups", {})
        if not isinstance(completed, dict):
            raise ValueError("fixed plan is missing completed.groups")
        paths.extend(Path(value["path"]) for value in completed.values())
    for directory in group_dirs:
        paths.extend(sorted(directory.glob("*.json")))
    identities = [path.stem for path in paths]
    if len(identities) != len(set(identities)):
        raise ValueError("fixed group sources contain duplicate task IDs")
    return paths


def _fixed_rows(
    tasks_path: Path,
    *,
    plan_path: Path | None,
    group_dirs: list[Path],
) -> list[dict[str, Any]]:
    _, by_id = _task_maps(tasks_path)
    output: list[dict[str, Any]] = []
    for path in _fixed_paths(plan_path, group_dirs):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: expected non-empty episode list")
        identifier = path.stem
        if identifier not in by_id:
            raise ValueError(f"{path}: task is absent from tasks file")
        clean = len(rows) == 8 and all(
            row.get("sample", {}).get("process_update") is True for row in rows
        )
        correct = sum(bool(row.get("sample", {}).get("correct")) for row in rows)
        legal = sum(row.get("sample", {}).get("failure_type") is None for row in rows)
        sample_records = [row.get("sample", {}).get("audit_record") for row in rows]
        if not all(isinstance(record, dict) for record in sample_records):
            raise ValueError(f"{path}: sample is missing audit_record")
        output.append(
            _compact(
                by_id[identifier],
                correct_count=correct,
                legal_count=legal,
                group_size=len(rows),
                clean=clean,
                source_kind="fixed-group-json",
                source_path=path,
                sample_records=sample_records,
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--passk-jsonl", type=Path)
    modes.add_argument("--fixed-plan", type=Path)
    parser.add_argument("--fixed-group-dir", type=Path, action="append", default=[])
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()

    if args.passk_jsonl is not None:
        if args.fixed_group_dir:
            parser.error("--fixed-group-dir cannot be used with --passk-jsonl")
        rows = _passk_rows(args.passk_jsonl, args.tasks)
    else:
        rows = _fixed_rows(
            args.tasks,
            plan_path=args.fixed_plan,
            group_dirs=args.fixed_group_dir,
        )
    if len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} records, found {len(rows)}")
    rows.sort(key=lambda row: (row["task_id"], row["example_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(rows),
                "sha256": _sha256(args.output),
                "class_counts": {
                    name: sum(row["outcome_class"] == name for row in rows)
                    for name in ("mixed", "all_correct", "all_wrong", "contaminated")
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
