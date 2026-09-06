#!/usr/bin/env python3
"""Freeze a deterministic mixed-outcome cohort after an operator-requested screen stop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "qwen3-v26-earlystop-mixed-grpo-cohort-v1"
SELECTION_SEED = "qwen3-v26-earlystop-mixed180-20260813"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object JSONL row: {path}")
    return rows


def task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id") or row.get("trajectory_id")
    if not isinstance(value, str) or not value:
        index = row.get("example_index")
        if type(index) is not int:
            raise ValueError("task has no stable identity")
        return f"bird_train_{index:05d}"
    return value


def validate_group(path: Path, expected_task: str, position: int) -> tuple[int, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError(f"group must contain exactly K=8 rows: {path}")
    sample_indices: set[int] = set()
    correct = 0
    for offset, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"group row is not an object: {path}")
        environment = row.get("environment") or {}
        sample = row.get("sample") or {}
        audit = sample.get("audit_record") or {}
        if environment.get("task_id") != expected_task:
            raise ValueError(f"group/task mismatch: {path}")
        if row.get("sequence") != position * 8 + offset:
            raise ValueError(f"global sequence mismatch: {path}")
        sample_index = audit.get("sample_index")
        if type(sample_index) is not int:
            raise ValueError(f"sample index missing: {path}")
        sample_indices.add(sample_index)
        if audit.get("protocol_version") != "version26" or audit.get("protocol_hash") != "4da19387399bd3a5":
            raise ValueError(f"protocol mismatch: {path}")
        if type(sample.get("process_update")) is not bool:
            raise ValueError(f"process_update must be explicit bool: {path}")
        if sample["process_update"] is not True:
            return -1, "runtime_contaminated"
        if type(sample.get("correct")) is not bool:
            raise ValueError(f"correct must be explicit bool: {path}")
        expected_reward = 1.0 if sample["correct"] else 0.0
        if float(sample.get("reward", -1.0)) != expected_reward:
            raise ValueError(f"binary reward mismatch: {path}")
        correct += int(sample["correct"])
    if sample_indices != set(range(8)):
        raise ValueError(f"sample indices are not exactly 0..7: {path}")
    return correct, "clean"


def atomic_publish(output_dir: Path, tasks_payload: bytes, manifest: dict[str, Any]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.next-", dir=output_dir.parent))
    try:
        tasks_path = temporary / "train180.jsonl"
        tasks_path.write_bytes(tasks_payload)
        manifest["outputs"] = {
            "train180": {
                "path": str(output_dir / "train180.jsonl"),
                "sha256": sha256_file(tasks_path),
                "records": 180,
            }
        }
        manifest_path = temporary / "earlystop_mixed180_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(tasks_path, 0o444)
        os.chmod(manifest_path, 0o444)
        temporary.rename(output_dir)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = args.tasks.resolve()
    plan_path = args.plan.resolve()
    tasks = load_jsonl(tasks_path)
    if len(tasks) != 600:
        raise ValueError("source task universe must contain exactly 600 rows")
    ordered_ids = [task_id(row) for row in tasks]
    if len(set(ordered_ids)) != 600:
        raise ValueError("source task identities are not unique")
    positions = {value: index for index, value in enumerate(ordered_ids)}
    task_rows = dict(zip(ordered_ids, tasks))
    plan = load_object(plan_path)
    if plan.get("status") != "frozen_continuation_plan":
        raise ValueError("continuation plan is not frozen")
    if plan.get("tasks", {}).get("sha256") != sha256_file(tasks_path):
        raise ValueError("continuation plan task SHA mismatch")

    paths: dict[str, Path] = {}
    completed = (plan.get("completed") or {}).get("groups") or {}
    if len(completed) != 69:
        raise ValueError("frozen plan must bind exactly 69 prior groups")
    for identity, record in completed.items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"prior group SHA mismatch: {path}")
        paths[identity] = path
    source_counts: dict[str, int] = {"frozen_prior": len(paths)}
    for label, directory in args.group_dir:
        directory = directory.resolve()
        if list(directory.glob("*.next")):
            raise ValueError(f"unfinished group exists in {label}: {directory}")
        found = sorted(directory.glob("*.json"))
        source_counts[label] = len(found)
        for path in found:
            identity = path.stem
            if identity in paths:
                raise ValueError(f"duplicate group identity: {identity}")
            paths[identity] = path
    if len(paths) != 568:
        raise ValueError(f"early-stop source must contain exactly 568 groups, got {len(paths)}")
    if not set(paths).issubset(task_rows):
        raise ValueError("screen group is outside frozen 600-task universe")

    histogram: Counter[int | str] = Counter()
    candidates: list[tuple[int, str, str, int]] = []
    group_hashes: dict[str, str] = {}
    for identity, path in sorted(paths.items()):
        count, status = validate_group(path, identity, positions[identity])
        group_hashes[identity] = sha256_file(path)
        if status != "clean":
            histogram["contaminated"] += 1
            continue
        histogram[count] += 1
        if 1 <= count <= 7:
            uncertainty = count * (8 - count)
            tie = hashlib.sha256(f"{SELECTION_SEED}\t{identity}".encode()).hexdigest()
            candidates.append((-uncertainty, tie, identity, count))
    if len(candidates) != 193:
        raise ValueError(f"expected frozen early-stop mixed count 193, got {len(candidates)}")
    candidates.sort()
    selected = candidates[:180]
    selected_ids = [item[2] for item in selected]
    selected_rows = [task_rows[identity] for identity in selected_ids]
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in selected_rows
    )
    selected_histogram = Counter(item[3] for item in selected)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_operator_requested_earlystop_mixed180",
        "operator_decision": {
            "stopped_before_full_600": True,
            "completed_groups": 568,
            "missing_groups": 32,
            "must_not_be_reported_as_preregistered_600_arm": True,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "rule": "all-clean mixed only; descending c*(8-c); SHA-256 tie break",
            "eligible_mixed": len(candidates),
            "selected": 180,
            "selected_task_ids": selected_ids,
            "selected_correct_count_histogram": dict(sorted(selected_histogram.items())),
            "held_out_mixed_task_ids": [item[2] for item in candidates[180:]],
        },
        "screen": {
            "source_counts": source_counts,
            "correct_count_histogram": {str(k): v for k, v in sorted(histogram.items(), key=lambda x: str(x[0]))},
            "group_sha256": group_hashes,
        },
        "inputs": {
            "tasks": {"path": str(tasks_path), "sha256": sha256_file(tasks_path), "records": 600},
            "continuation_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        },
        "training_contract": {
            "records": 180,
            "group_size": 8,
            "prompts_per_update": 30,
            "optimizer_steps": 12,
            "passes": 2,
            "fresh_online_trajectories": 2880,
            "reward": "binary-result-only",
            "kl_beta": 0.0,
        },
    }
    atomic_publish(args.output_dir.resolve(), payload, manifest)
    return manifest


def parse_group_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group dir must be LABEL=/absolute/path")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_absolute():
        raise argparse.ArgumentTypeError("group dir must be LABEL=/absolute/path")
    return label, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--group-dir", action="append", type=parse_group_dir, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build(args)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"early-stop cohort blocked: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "selected": 180}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
