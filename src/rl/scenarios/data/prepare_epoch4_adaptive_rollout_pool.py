#!/usr/bin/env python3
"""Prepare a smaller, structure-aware rollout pool for Epoch-4 result-only RL.

This is an *upstream* sampler, not a reward or trajectory rewriter.  It keeps a
small easy calibration slice, emphasizes medium/hard SQL structures, and
preserves a configurable source mix across BIRD/Spider/SynSQL.  The following
rollout stage should use a cheap K=2 pilot and reserve fresh K=8 sampling for
pilot-boundary tasks; no existing screen trajectory is reused for training.

Gold SQL is compiled only into content-free structural metadata.  The selected
task file still contains the original Harness fields required to execute a task,
but the manifest never stores SQL text or a model-authored action sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.sql_task_coverage_profile import profile_identity, profile_sql


SCHEMA_VERSION = "epoch4-adaptive-rollout-pool-v1"
MODEL_CHECKPOINT = "Epoch4-checkpoint-6380"
SOURCE_WEIGHTS = {"BIRD": 0.50, "Spider": 0.20, "SynSQL": 0.30}
SOURCE_PREFIXES = (("bird_", "BIRD"), ("spider_", "Spider"), ("synsql_", "SynSQL"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id/instance_id")
    return value


def _source(identifier: str) -> str:
    for prefix, name in SOURCE_PREFIXES:
        if identifier.startswith(prefix):
            return name
    return "Other"


def _rank(seed: str, namespace: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{identifier}".encode()).hexdigest()


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return payload, rows


def _profile(sql: Any) -> dict[str, Any]:
    raw = str(sql or "").strip()
    if not raw:
        return {"status": "missing", "difficulty": "unknown", "score": None}
    try:
        value = profile_sql(raw)
    except Exception as exc:
        return {"status": "error", "difficulty": "unknown", "score": None, "error": type(exc).__name__}
    data = value.to_json()
    counts = data["counts"]
    return {
        "status": "ok",
        "difficulty": value.difficulty,
        "score": value.difficulty_score,
        "atomic_support": value.atomic_support,
        "scope_count": counts.get("select_scopes", 0),
        "join_count": counts.get("joins", 0),
        "aggregate_count": counts.get("aggregate_metrics", 0),
        "predicate_count": counts.get("predicate_leaves", 0),
        "group_count": counts.get("group_by_columns", 0),
        "set_count": counts.get("set_operations", 0),
        "window_count": counts.get("windows", 0),
        "subquery_count": counts.get("subqueries", 0),
        "shape_hash": profile_identity(value),
    }


def _is_obviously_simple(profile: Mapping[str, Any]) -> bool:
    return (
        profile.get("difficulty") == "easy"
        and profile.get("scope_count", 0) <= 1
        and profile.get("join_count", 0) == 0
        and profile.get("aggregate_count", 0) == 0
        and profile.get("group_count", 0) == 0
        and profile.get("set_count", 0) == 0
        and profile.get("window_count", 0) == 0
        and profile.get("subquery_count", 0) == 0
    )


def _target_counts(total: int, available: Mapping[str, int]) -> dict[str, int]:
    requested = {source: int(round(total * SOURCE_WEIGHTS[source])) for source in SOURCE_WEIGHTS}
    requested["BIRD"] += total - sum(requested.values())
    counts = {source: min(requested[source], int(available.get(source, 0))) for source in requested}
    remainder = total - sum(counts.values())
    while remainder > 0:
        choices = sorted(
            (source for source in requested if counts[source] < int(available.get(source, 0))),
            key=lambda source: (requested[source] - counts[source], source),
            reverse=True,
        )
        if not choices:
            break
        counts[choices[0]] += 1
        remainder -= 1
    if sum(counts.values()) != total:
        raise ValueError(f"not enough BIRD/Spider/SynSQL tasks after filtering: need {total}, got {sum(counts.values())}")
    return counts


def _select_source(rows: Sequence[dict[str, Any]], count: int, *, seed: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"source has only {len(rows)} candidates; need {count}")
    # At most 8% of any source quota may be obviously simple.  Medium/hard rows
    # are ranked by compiler difficulty, while ties remain deterministic.
    simple = [row for row in rows if row["simple"]]
    complex_rows = [row for row in rows if not row["simple"]]
    easy_cap = min(len(simple), max(1, int(math.floor(count * 0.08))))
    simple.sort(key=lambda row: _rank(seed, "easy", row["task_id"]))
    complex_rows.sort(
        key=lambda row: (
            -int(row["profile"].get("score") or -1),
            row["profile"].get("difficulty") != "hard",
            _rank(seed, "complex", row["task_id"]),
        )
    )
    complex_needed = min(len(complex_rows), count)
    simple_needed = count - complex_needed
    # Prefer all available non-simple candidates.  Only use easy rows to fill a
    # source quota that cannot otherwise be met; with a sufficiently large
    # source this naturally caps easy rows at 8%.
    if simple_needed < easy_cap and len(complex_rows) >= count - easy_cap:
        complex_needed = count - easy_cap
        simple_needed = easy_cap
    selected = complex_rows[:complex_needed] + simple[:simple_needed]
    selected.sort(key=lambda row: _rank(seed, "output", row["task_id"]))
    return selected


def prepare_pool(*, tasks_path: Path, output_dir: Path, count: int, seed: str) -> dict[str, Any]:
    task_bytes, tasks = _read_jsonl(tasks_path)
    if not tasks:
        raise ValueError("tasks file is empty")
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    annotated: list[dict[str, Any]] = []
    for row in tasks:
        identifier = _task_id(row)
        index = row.get("example_index")
        if identifier in seen_ids or type(index) is not int or index in seen_indices:
            raise ValueError(f"duplicate/invalid task identity: {identifier}")
        seen_ids.add(identifier)
        seen_indices.add(index)
        source = _source(identifier)
        if source not in SOURCE_WEIGHTS:
            continue
        profile = _profile(row.get("gold_sql") or row.get("query"))
        annotated.append(
            {
                "task": row,
                "task_id": identifier,
                "example_index": index,
                "source": source,
                "profile": profile,
                "simple": _is_obviously_simple(profile),
            }
        )
    available = Counter(row["source"] for row in annotated if not row["simple"])
    # If a source has fewer non-simple tasks than its quota, simple rows can fill
    # the remainder but never become the dominant source of computation.
    source_counts = _target_counts(count, {source: sum(1 for row in annotated if row["source"] == source) for source in SOURCE_WEIGHTS})
    selected: list[dict[str, Any]] = []
    for source, quota in source_counts.items():
        selected.extend(_select_source([row for row in annotated if row["source"] == source], quota, seed=seed))
    if len(selected) != count:
        raise RuntimeError("selection count mismatch")
    selected.sort(key=lambda row: (int(row["example_index"]), row["task_id"]))
    selected_tasks = [row["task"] for row in selected]
    metadata = [
        {
            "example_index": row["example_index"],
            "task_id": row["task_id"],
            "source": row["source"],
            "difficulty": row["profile"].get("difficulty"),
            "score": row["profile"].get("score"),
            "atomic_support": row["profile"].get("atomic_support"),
            "simple": row["simple"],
            "shape_hash": row["profile"].get("shape_hash"),
        }
        for row in selected
    ]
    metadata_bytes = (
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in metadata
        ).encode()
    )
    task_payload = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in selected_tasks
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "rollout_tasks.jsonl").write_bytes(task_payload)
    (output_dir / "selection_metadata.jsonl").write_bytes(metadata_bytes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_pool_prepared_not_rl_admitted",
        "purpose": "upstream task reduction before Epoch4 result-only RL pilot",
        "model_checkpoint": MODEL_CHECKPOINT,
        "inputs": {"tasks": {"path": str(tasks_path.resolve()), "sha256": _sha(task_bytes), "records": len(tasks)}},
        "selection": {
            "seed": seed,
            "requested_count": count,
            "selected_count": len(selected),
            "source_weights": SOURCE_WEIGHTS,
            "source_counts": dict(Counter(row["source"] for row in selected)),
            "difficulty_counts": dict(Counter(row["profile"].get("difficulty") for row in selected)),
            "simple_count": sum(row["simple"] for row in selected),
            "simple_fraction": sum(row["simple"] for row in selected) / len(selected),
            "rule": "compiler medium/hard first; at most 8% obviously simple per source; deterministic source quotas",
            "gold_sql_used_for": "offline compiler structure only; never stored in this manifest or metadata",
        },
        "rollout_contract": {
            "screen_stage": "pilot K=2 then fresh K=8 only for mixed pilot groups",
            "screen_trajectories_reused": False,
            "reward": "binary result-only terminal denotation",
            "sft_tasks_must_be_excluded": True,
        },
        "outputs": {
            "rollout_tasks": {"path": str((output_dir / "rollout_tasks.jsonl").resolve()), "sha256": _sha(task_payload), "records": len(selected_tasks)},
            "selection_metadata": {"path": str((output_dir / "selection_metadata.jsonl").resolve()), "sha256": _sha(metadata_bytes), "records": len(metadata)},
        },
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", default="epoch4-adaptive-rollout-pool-v1")
    args = parser.parse_args()
    manifest = prepare_pool(tasks_path=args.tasks.resolve(), output_dir=args.output_dir.resolve(), count=args.count, seed=args.seed)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "records": manifest["outputs"]["rollout_tasks"]["records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
