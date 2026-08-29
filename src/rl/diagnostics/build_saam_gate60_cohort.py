#!/usr/bin/env python3
"""Build the frozen BIRD-train cohort for the online SAAM Gate60.

The source pool must contain exactly K=8 eligible rollouts per task and both
binary terminal outcomes.  Selection is based only on task identity and a
fixed hash seed; trajectory rewards are used only as an eligibility gate, not
to rank candidates.  Original training-task JSON lines are copied byte-for-
byte, preserving the existing hidden-label/runtime contract.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen3-v26-saam-gate60-cohort-v1"
DEFAULT_SEED = "qwen3-v26-saam-gate60-v1-20260828"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("training task lacks example_id/instance_id")
    return value


def _pool_task_id(row: dict[str, Any]) -> str:
    environment = row.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("pool row lacks environment object")
    value = environment.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("pool row lacks environment.task_id")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(
    *,
    pool: Path,
    source_tasks: Path,
    output: Path,
    manifest: Path,
    count: int,
    group_size: int,
    seed: str,
) -> dict[str, Any]:
    if count < 1 or group_size < 2 or not seed:
        raise ValueError("count, group_size, and seed must be positive/non-empty")
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite an existing cohort artifact")

    pool_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(pool, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample = row.get("sample")
            if not isinstance(sample, dict):
                raise ValueError(f"pool line {line_number} lacks sample object")
            pool_groups[_pool_task_id(row)].append(sample)

    reward_counts: dict[str, dict[str, int]] = {}
    candidates: list[str] = []
    for task_id, samples in pool_groups.items():
        if len(samples) != group_size:
            raise ValueError(
                f"{task_id}: expected K={group_size}, observed {len(samples)}"
            )
        if not all(bool(sample.get("process_update")) for sample in samples):
            raise ValueError(f"{task_id}: pool contains a process_update=false row")
        rewards = [float(sample.get("reward")) for sample in samples]
        if any(reward not in {0.0, 1.0} for reward in rewards):
            raise ValueError(f"{task_id}: non-binary reward in source pool")
        zeros = sum(reward == 0.0 for reward in rewards)
        ones = sum(reward == 1.0 for reward in rewards)
        if not zeros or not ones:
            raise ValueError(f"{task_id}: source group is not mixed reward")
        reward_counts[task_id] = {"zero": zeros, "one": ones}
        candidates.append(task_id)

    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible tasks for requested {count}")
    candidates.sort(
        key=lambda task_id: (
            hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest(),
            task_id,
        )
    )
    selected = candidates[:count]

    source_rows: dict[str, bytes] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    with source_tasks.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            task_id = _task_id(row)
            if task_id in source_rows:
                raise ValueError(f"duplicate source task id: {task_id}")
            source_rows[task_id] = raw_line.rstrip(b"\r\n")
            source_metadata[task_id] = {
                "example_index": int(row["example_index"]),
                "db_id": str(row["db_id"]),
                "split": str(row["split"]),
            }

    missing = [task_id for task_id in selected if task_id not in source_rows]
    if missing:
        raise ValueError(f"selected tasks missing from source task file: {missing}")
    if any(source_metadata[task_id]["split"] != "train" for task_id in selected):
        raise ValueError("Gate60 selection contains a non-training task")

    output_payload = b"".join(source_rows[task_id] + b"\n" for task_id in selected)
    _atomic_write(output, output_payload)

    selected_reward_histogram: dict[str, int] = defaultdict(int)
    for task_id in selected:
        counts = reward_counts[task_id]
        selected_reward_histogram[f"{counts['one']}_correct_of_{group_size}"] += 1

    output_sha256 = _sha256(output)
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_training_cohort",
        "dataset": "BIRD-train",
        "dev1534_used": False,
        "selection": {
            "method": "ascending_sha256(seed + NUL + task_id)",
            "seed": seed,
            "requested_records": count,
            "candidate_records": len(candidates),
            "source_group_size": group_size,
            "eligibility": "K8 process-update=true binary mixed terminal reward",
            "reward_used_for_ranking": False,
        },
        "sources": {
            "mixed_rollout_pool": {
                "path": _display_path(pool),
                "sha256": _sha256(pool),
            },
            "training_tasks": {
                "path": _display_path(source_tasks),
                "sha256": _sha256(source_tasks),
            },
        },
        "output": {
            "path": _display_path(output),
            "records": len(selected),
            "sha256": output_sha256,
            "task_ids_in_order": selected,
            "reward_count_histogram_in_source_pool": dict(
                sorted(selected_reward_histogram.items())
            ),
            "tasks": [
                {
                    "task_id": task_id,
                    **source_metadata[task_id],
                    "source_pool_rewards": reward_counts[task_id],
                }
                for task_id in selected
            ],
        },
    }
    _atomic_write(
        manifest,
        (json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    payload = build(
        pool=args.pool.resolve(),
        source_tasks=args.source_tasks.resolve(),
        output=args.output.resolve(),
        manifest=args.manifest.resolve(),
        count=args.count,
        group_size=args.group_size,
        seed=args.seed,
    )
    print(json.dumps(payload["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
