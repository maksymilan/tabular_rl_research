#!/usr/bin/env python3
"""Preflight and supervise the frozen Atomic-v24 9K rollout on one server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "deepseek-v4-flash"
OFFICIAL_BASE_URL = "https://api.deepseek.com"
PROFILE = "atomic-v24-frozen-v1"
GUIDANCE = "checkpoint-disabled-v1"
SHARD_SIZE = 50
SHARD_TOKEN_CAP = 3_600_000
TOTAL_TOKEN_CAP = 650_000_000
MAX_CONCURRENCY = 16
DEPLOYED_COMMIT = "84e1e341e37eea9f465b6ba6565a67c71770426b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _api_identity(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    base_url = values.get("BASE_URL", "").rstrip("/")
    key = values.get("API_KEY") or values.get("DEEPSEEK_API_KEY") or ""
    if base_url != OFFICIAL_BASE_URL:
        raise RuntimeError("api config must use the official DeepSeek base URL")
    if not key:
        raise RuntimeError("api config has no nonempty key")
    return {"base_url": base_url, "api_key_present": True}


def _preflight(root: Path, tasks_path: Path, manifest_path: Path, api_path: Path) -> dict[str, Any]:
    if SHARD_TOKEN_CAP * 180 > TOTAL_TOKEN_CAP:
        raise RuntimeError("reserved shard tokens exceed the global token cap")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    harness = manifest.get("outputs", {}).get("harness_tasks", {})
    actual_sha = _sha256(tasks_path)
    if harness.get("sha256") != actual_sha or harness.get("records") != 9000:
        raise RuntimeError("portable dataset does not match its manifest")
    if harness.get("path") != str(tasks_path.relative_to(root)):
        raise RuntimeError("portable manifest path is not rooted at the deployment")
    count = 0
    ids: set[str] = set()
    example_indexes: set[int] = set()
    datasets: dict[str, int] = {}
    missing: list[str] = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id or example_id in ids:
                raise RuntimeError("dataset example_id values must be unique and nonempty")
            ids.add(example_id)
            example_index = row.get("example_index")
            if (
                isinstance(example_index, bool)
                or not isinstance(example_index, int)
                or example_index != count - 1
                or example_index in example_indexes
                or row.get("index") != example_index
            ):
                raise RuntimeError(
                    "portable dataset requires unique position-aligned example_index/index"
                )
            example_indexes.add(example_index)
            dataset = str(row.get("dataset"))
            datasets[dataset] = datasets.get(dataset, 0) + 1
            db_path = Path(str(row.get("db_path")))
            resolved = db_path if db_path.is_absolute() else root / db_path
            if not resolved.is_file() and len(missing) < 20:
                missing.append(str(db_path))
    if count != 9000 or len(ids) != 9000 or len(example_indexes) != 9000:
        raise RuntimeError("dataset is not the frozen 9K question cohort")
    if missing:
        raise RuntimeError(f"missing database files: {missing}")
    api = _api_identity(api_path)
    return {
        "schema_version": "checkpoint-atomic-v24-frozen-9k-server-launch-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployed_git_commit": DEPLOYED_COMMIT,
        "supervisor_sha256": _sha256(Path(__file__).resolve()),
        "tasks_path": str(tasks_path),
        "tasks_sha256": actual_sha,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "records": count,
        "datasets": datasets,
        "model": MODEL,
        "provider": api,
        "mode": "atomic",
        "atomic_operator_profile": PROFILE,
        "checkpoint_guidance_profile": GUIDANCE,
        "max_checkpoints": 0,
        "max_restores": 0,
        "shard_size": SHARD_SIZE,
        "shard_count": 180,
        "max_concurrency": MAX_CONCURRENCY,
        "per_shard_provider_token_cap": SHARD_TOKEN_CAP,
        "total_provider_token_cap": TOTAL_TOKEN_CAP,
    }


def _completed_positions(
    roots: list[Path], tasks_path: Path
) -> tuple[set[int], dict[str, Any]]:
    task_ids: list[str] = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            task_ids.append(str(json.loads(line).get("example_id")))
    completed: set[int] = set()
    journals: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            raise RuntimeError(f"completed-result root does not exist: {root}")
        for journal in sorted(root.rglob("all.jsonl")):
            count = 0
            with journal.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    position = record.get("task_position")
                    if (
                        isinstance(position, bool)
                        or not isinstance(position, int)
                        or not 0 <= position < len(task_ids)
                    ):
                        raise RuntimeError(
                            f"invalid task_position in {journal}:{line_number}"
                        )
                    if record.get("example_id") != task_ids[position]:
                        raise RuntimeError(
                            f"example identity mismatch in {journal}:{line_number}"
                        )
                    if position in completed:
                        raise RuntimeError(f"duplicate completed task position {position}")
                    completed.add(position)
                    count += 1
            journals.append(
                {
                    "path": str(journal),
                    "sha256": _sha256(journal),
                    "records": count,
                }
            )
    encoded_positions = json.dumps(sorted(completed), separators=(",", ":")).encode()
    return completed, {
        "roots": [str(root) for root in roots],
        "journals": journals,
        "completed_records": len(completed),
        "completed_positions_sha256": hashlib.sha256(encoded_positions).hexdigest(),
    }


def _pending_ranges(completed: set[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for position in (value for value in range(9000) if value not in completed):
        if start is None:
            start = previous = position
            continue
        assert previous is not None
        if position != previous + 1 or position - start >= SHARD_SIZE:
            ranges.append((start, previous - start + 1))
            start = position
        previous = position
    if start is not None and previous is not None:
        ranges.append((start, previous - start + 1))
    return ranges


def _command(
    root: Path,
    tasks_path: Path,
    manifest_path: Path,
    api_path: Path,
    result_dir: Path,
    start: int,
    size: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tool_modules.checkpoint_relalg.runner",
        "--mode",
        "atomic",
        "--atomic-operator-profile",
        PROFILE,
        "--carrier",
        "text-json",
        "--experiment-arm",
        "A",
        "--checkpoint-guidance-profile",
        GUIDANCE,
        "--tasks-json",
        str(tasks_path),
        "--dataset-manifest",
        str(manifest_path),
        "--result-dir",
        str(result_dir),
        "--start",
        str(start),
        "--n",
        str(size),
        "--model",
        MODEL,
        "--api-config",
        str(api_path),
        "--api-timeout-seconds",
        "300",
        "--api-retries",
        "3",
        "--max-tokens",
        "32768",
        "--max-completion-tokens",
        "131072",
        "--max-model-turns",
        "30",
        "--max-primitive-calls",
        "50",
        "--max-checkpoints",
        "0",
        "--max-restores",
        "0",
        "--sql-timeout-seconds",
        "20",
        "--max-artifact-rows",
        "100000",
        "--max-artifact-bytes",
        "67108864",
        "--max-cell-bytes",
        "4194304",
        "--max-batch-provider-attempts",
        "600",
        "--max-batch-provider-tokens",
        str(SHARD_TOKEN_CAP),
        "--max-batch-wall-seconds",
        "14400",
        "--max-consecutive-provider-failures",
        "2",
        "--max-total-provider-failures",
        "3",
        "--max-consecutive-semantic-failures",
        "1000000",
        "--workers",
        "1",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument(
        "--exclude-completed-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    tasks_path = root / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.portable.tasks.jsonl"
    manifest_path = root / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.portable.manifest.json"
    api_path = (
        args.api_config.expanduser().resolve()
        if args.api_config is not None
        else root / "secrets/api.md"
    )
    result_root = (
        args.result_root.expanduser().resolve()
        if args.result_root is not None
        else root / "data/results/checkpoint_atomic_v24_frozen_sft9k_20260820"
    )
    result_root.mkdir(parents=True, exist_ok=True)
    launch_manifest = _preflight(root, tasks_path, manifest_path, api_path)
    completed_positions, exclusion_identity = _completed_positions(
        [path.expanduser().resolve() for path in args.exclude_completed_root],
        tasks_path,
    )
    pending_ranges = _pending_ranges(completed_positions)
    if len(pending_ranges) * SHARD_TOKEN_CAP + 1 > TOTAL_TOKEN_CAP:
        raise RuntimeError("reserved continuation tokens exceed the global token cap")
    launch_manifest["excluded_completed"] = exclusion_identity
    launch_manifest["pending_records"] = 9000 - len(completed_positions)
    launch_manifest["pending_ranges"] = [
        {"start": start, "size": size} for start, size in pending_ranges
    ]
    launch_manifest["result_root"] = str(result_root)
    launch_manifest["max_model_turns"] = 30
    launch_manifest["max_primitive_calls"] = 50
    launch_manifest["max_tokens"] = 32768
    launch_manifest["max_completion_tokens"] = 131072
    launch_manifest["task_local_provider_failures_do_not_stop_batch"] = True
    _atomic_json(result_root / "server_launch_manifest.json", launch_manifest)
    if not args.launch:
        print(json.dumps(launch_manifest, ensure_ascii=False, sort_keys=True))
        return 0

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{root / 'src/eval'}:{root / 'src'}"
    pending = list(pending_ranges)
    active: dict[int, tuple[subprocess.Popen[bytes], Any]] = {}
    completed: dict[str, int] = {}
    consecutive_failed_shards = 0
    status_path = result_root / "server_supervisor_status.json"

    while pending or active:
        while pending and len(active) < MAX_CONCURRENCY and consecutive_failed_shards < 3:
            start, size = pending.pop(0)
            shard = result_root / f"shard_{start:04d}_{start + size - 1:04d}"
            if shard.exists():
                raise RuntimeError(f"refusing to overwrite existing shard: {shard}")
            log_path = result_root / f"shard_{start:04d}_{start + size - 1:04d}.log"
            log_handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(
                _command(root, tasks_path, manifest_path, api_path, shard, start, size),
                cwd=root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[start] = (process, log_handle)

        finished: list[int] = []
        for start, (process, log_handle) in active.items():
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            completed[str(start)] = code
            consecutive_failed_shards = consecutive_failed_shards + 1 if code else 0
            finished.append(start)
        for start in finished:
            del active[start]
        _atomic_json(
            status_path,
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "state": "running" if pending or active else "completed",
                "pending_shards": len(pending),
                "active_shards": sorted(active),
                "completed_shards": completed,
                "consecutive_failed_shards": consecutive_failed_shards,
                "dispatch_halted": consecutive_failed_shards >= 3,
            },
        )
        if consecutive_failed_shards >= 3:
            if not active:
                return 2
        time.sleep(10)
    return 0 if all(code == 0 for code in completed.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
