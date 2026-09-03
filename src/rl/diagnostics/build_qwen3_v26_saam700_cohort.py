#!/usr/bin/env python3
"""Build the exact 700-task version26 SAAM/four-level online-RL cohort.

Screen trajectories are used only to select task identities.  The output is a
task JSONL for fresh online K=8 rollout; no sampled reasoning, action, or gold
tool path is copied into training data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from harness.sql_task_coverage_profile import (  # noqa: E402
    DIFFICULTY_VERSION,
    PROFILE_VERSION,
    profile_sql,
)


SCHEMA_VERSION = "qwen3-v26-saam-fourlevel-online-rl-cohort-v1"
SEED = "qwen3-v26-saam-fourlevel-700-20260902"
SOURCE_PREFIXES = ("bird", "spider", "synsql")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(namespace: str, task_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{namespace}\0{task_id}".encode()).hexdigest()


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


def _source(task_id: str) -> str:
    for prefix in SOURCE_PREFIXES:
        if task_id.startswith(f"{prefix}_train_"):
            return prefix
    raise ValueError(f"unsupported task source: {task_id}")


def _tasks(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        identifier = _task_id(row)
        if identifier in output:
            raise ValueError(f"{path}: duplicate task {identifier}")
        output[identifier] = row
    return output


def _outcomes(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        identifier = row.get("task_id")
        if not isinstance(identifier, str) or identifier in output:
            raise ValueError(f"{path}: invalid/duplicate outcome task {identifier!r}")
        output[identifier] = row
    return output


def _same_task(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("db_id", "question", "gold_sql", "external_knowledge")
    )


def _category(counts: dict[str, int]) -> str:
    if counts["set_operations"]:
        return "set"
    if counts["windows"]:
        return "window"
    if counts["subqueries"]:
        return "subquery"
    if counts["joins"] and counts["aggregate_metrics"]:
        return "join_group"
    if counts["joins"]:
        return "join"
    if counts["aggregate_metrics"] or counts["group_by_columns"]:
        return "group"
    if counts["order_keys"] and counts["limits"]:
        return "rank"
    return "filter_project"


def _profile(task: dict[str, Any]) -> dict[str, Any]:
    value = profile_sql(str(task.get("gold_sql") or task.get("query") or ""))
    counts = value.counts
    return {
        "difficulty": value.difficulty,
        "difficulty_score": value.difficulty_score,
        "category": _category(counts),
        "joins": counts["joins"],
        "aggregate_metrics": counts["aggregate_metrics"],
        "predicate_leaves": counts["predicate_leaves"],
        "select_scopes": counts["select_scopes"],
        "subqueries": counts["subqueries"],
        "set_operations": counts["set_operations"],
    }


def _remap_db_path(task: dict[str, Any], project_root: Path) -> str:
    raw = str(task.get("db_path") or "")
    if not raw:
        raise ValueError(f"{_task_id(task)}: missing db_path")
    normalized = raw.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
    elif normalized.startswith("data/"):
        suffix = normalized[len("data/") :]
    else:
        raise ValueError(f"{_task_id(task)}: cannot remap db_path {raw!r}")
    relative = Path(suffix)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{_task_id(task)}: unsafe db_path suffix {suffix!r}")
    return str(project_root / "data" / relative)


def _distribution(rows: Iterable[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    return {
        "records": len(selected),
        "source": dict(sorted(Counter(_source(_task_id(row)) for row in selected).items())),
        "difficulty": dict(
            sorted(Counter(annotations[_task_id(row)]["profile"]["difficulty"] for row in selected).items())
        ),
        "question_type": dict(
            sorted(Counter(annotations[_task_id(row)]["profile"]["category"] for row in selected).items())
        ),
        "unique_db_ids": len({str(row.get("db_id")) for row in selected}),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    old_tasks = _tasks(args.old_tasks)
    old_outcomes = _outcomes(args.old_index)
    if len(old_outcomes) != 568:
        raise ValueError(f"old S1 index must contain 568 groups, found {len(old_outcomes)}")

    all_tasks = dict(old_tasks)
    new_shards: list[dict[str, Any]] = []
    new_outcomes: dict[str, dict[str, Any]] = {}
    new_task_owner: dict[str, str] = {}
    for label, tasks_path, index_path in args.screen_shard:
        shard_tasks = _tasks(Path(tasks_path))
        shard_outcomes = _outcomes(Path(index_path))
        if not set(shard_outcomes) <= set(shard_tasks):
            raise ValueError(f"{label}: outcomes include tasks absent from task shard")
        for identifier, task in shard_tasks.items():
            if identifier in all_tasks and not _same_task(all_tasks[identifier], task):
                raise ValueError(f"{label}: conflicting task payload for {identifier}")
            all_tasks.setdefault(identifier, task)
        overlap = set(new_outcomes) & set(shard_outcomes)
        if overlap:
            raise ValueError(f"{label}: duplicate new-screen outcomes: {sorted(overlap)[:5]}")
        new_outcomes.update(shard_outcomes)
        new_task_owner.update({identifier: label for identifier in shard_outcomes})
        new_shards.append(
            {
                "label": label,
                "tasks": {
                    "file": Path(tasks_path).name,
                    "records": len(shard_tasks),
                    "sha256": _sha256(Path(tasks_path)),
                },
                "outcomes": {
                    "file": Path(index_path).name,
                    "records": len(shard_outcomes),
                    "sha256": _sha256(Path(index_path)),
                },
            }
        )

    manual = json.loads(args.manual_selection.read_text())
    correct_ids = list(manual.get("all_correct_task_ids") or [])
    wrong_ids = list(manual.get("all_wrong_task_ids") or [])
    if len(correct_ids) != 50 or len(wrong_ids) != 50:
        raise ValueError("manual selection must contain exactly 50 all-correct and 50 all-wrong IDs")
    manual_ids = correct_ids + wrong_ids
    if len(manual_ids) != len(set(manual_ids)):
        raise ValueError("manual selection contains duplicate task IDs")
    for identifier, expected_class in [
        *((identifier, "all_correct") for identifier in correct_ids),
        *((identifier, "all_wrong") for identifier in wrong_ids),
    ]:
        outcome = old_outcomes.get(identifier)
        if outcome is None or outcome.get("outcome_class") != expected_class:
            raise ValueError(f"manual task {identifier} is not old-S1 {expected_class}")
        if outcome.get("clean") is not True or outcome.get("four_level_nonconstant") is not True:
            raise ValueError(f"manual task {identifier} lacks a clean, nonconstant four-level K8 group")

    old_mixed = {
        identifier
        for identifier, row in old_outcomes.items()
        if row.get("clean") is True and row.get("outcome_class") == "mixed"
    }
    if len(old_mixed) != 193:
        raise ValueError(f"old S1 must contribute 193 clean mixed tasks, found {len(old_mixed)}")
    new_mixed = {
        identifier
        for identifier, row in new_outcomes.items()
        if row.get("clean") is True and row.get("outcome_class") == "mixed"
    }

    selected_mixed: set[str] = set(old_mixed)
    # Manual tasks remain in the final set exactly once even if a newer screen
    # later observed a mixed K8 outcome for the same identity.
    selected_mixed -= set(manual_ids)
    new_nonbird = {
        identifier
        for identifier in new_mixed
        if _source(identifier) != "bird" and identifier not in manual_ids
    }
    selected_mixed.update(new_nonbird)

    remaining = args.mixed_records - len(selected_mixed)
    if remaining < 0:
        raise ValueError("old mixed plus non-BIRD mixed exceed the requested mixed cohort")
    bird_candidates = [
        identifier
        for identifier in new_mixed
        if _source(identifier) == "bird"
        and identifier not in selected_mixed
        and identifier not in manual_ids
    ]
    profiles = {identifier: _profile(all_tasks[identifier]) for identifier in set(manual_ids) | old_mixed | new_mixed}

    high = sorted(
        (identifier for identifier in bird_candidates if profiles[identifier]["difficulty_score"] >= 2),
        key=lambda identifier: (
            -profiles[identifier]["difficulty_score"],
            -(new_outcomes[identifier]["correct_count"] * (8 - new_outcomes[identifier]["correct_count"])),
            _stable_rank("bird-high", identifier),
        ),
    )
    selected_bird: list[str] = high[:remaining]
    if len(selected_bird) < remaining:
        low = [identifier for identifier in bird_candidates if identifier not in selected_bird]
        db_counts = Counter(all_tasks[identifier]["db_id"] for identifier in selected_bird)
        while low and len(selected_bird) < remaining:
            choice = min(
                low,
                key=lambda identifier: (
                    db_counts[all_tasks[identifier]["db_id"]],
                    -new_outcomes[identifier]["correct_count"] * (8 - new_outcomes[identifier]["correct_count"]),
                    -new_outcomes[identifier]["legal_count"],
                    _stable_rank("bird-low", identifier),
                ),
            )
            low.remove(choice)
            selected_bird.append(choice)
            db_counts[all_tasks[choice]["db_id"]] += 1
    if len(selected_bird) != remaining:
        raise ValueError(f"only {len(selected_bird)} new BIRD mixed tasks available; need {remaining}")
    selected_mixed.update(selected_bird)
    if len(selected_mixed) != args.mixed_records:
        raise AssertionError("mixed selection cardinality drift")

    annotations: dict[str, dict[str, Any]] = {}
    for identifier in selected_mixed:
        if identifier in old_mixed:
            bucket = "mixed_old_s1"
            screen = old_outcomes[identifier]
        else:
            bucket = "mixed_new_nonbird" if _source(identifier) != "bird" else "mixed_new_bird_adaptive"
            screen = new_outcomes[identifier]
        annotations[identifier] = {"bucket": bucket, "screen": screen, "profile": profiles[identifier]}
    for identifier in correct_ids:
        annotations[identifier] = {
            "bucket": "manual_old_all_correct",
            "screen": old_outcomes[identifier],
            "profile": profiles[identifier],
        }
    for identifier in wrong_ids:
        annotations[identifier] = {
            "bucket": "manual_old_all_wrong",
            "screen": old_outcomes[identifier],
            "profile": profiles[identifier],
        }

    final_ids = set(selected_mixed) | set(manual_ids)
    if len(final_ids) != args.total_records:
        raise ValueError(f"final unique records {len(final_ids)} != {args.total_records}")
    ordered_ids = sorted(final_ids, key=lambda identifier: _stable_rank("output", identifier))
    rows: list[dict[str, Any]] = []
    for cohort_index, identifier in enumerate(ordered_ids):
        task = copy.deepcopy(all_tasks[identifier])
        original_index = task.get("example_index")
        task["db_path"] = _remap_db_path(task, args.a100_project_root)
        task["example_index"] = cohort_index
        task["index"] = cohort_index
        task["example_id"] = identifier
        task["instance_id"] = identifier
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        metadata = copy.deepcopy(metadata)
        metadata["saam700_selection"] = {
            "schema_version": SCHEMA_VERSION,
            "bucket": annotations[identifier]["bucket"],
            "source_example_index": original_index,
            "screen_outcome_class": annotations[identifier]["screen"]["outcome_class"],
            "screen_correct_count": annotations[identifier]["screen"]["correct_count"],
            "screen_group_size": annotations[identifier]["screen"]["group_size"],
        }
        task["metadata"] = metadata
        rows.append(task)
    if len({row["example_index"] for row in rows}) != args.total_records:
        raise AssertionError("cohort example_index values are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_output = args.output_dir / "qwen3_8b_atomic_v26_saam_fourlevel_train700_v1.jsonl"
    with tasks_output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manual_audit = args.output_dir / "manual_homogeneous100_audit.jsonl"
    with manual_audit.open("w") as handle:
        for identifier in correct_ids + wrong_ids:
            task = all_tasks[identifier]
            annotation = annotations[identifier]
            profile = annotation["profile"]
            note = (
                "medium_or_hard_structural_anchor"
                if profile["difficulty"] in {"medium", "hard"}
                else "nontrivial_easy_calibration_with_structured_reward_contrast"
            )
            handle.write(
                json.dumps(
                    {
                        "task_id": identifier,
                        "db_id": task["db_id"],
                        "question": task["question"],
                        "historical_outcome_class": annotation["screen"]["outcome_class"],
                        "four_level_reward_histogram": annotation["screen"]["four_level_reward_histogram"],
                        "profile": profile,
                        "manual_selection_note": note,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    bucket_counts = Counter(value["bucket"] for value in annotations.values())
    manual_new_mixed_overlap = sorted(set(manual_ids) & new_mixed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_training_input",
        "purpose": "fresh online K8 SAAM-asymmetric-error four-level span-balanced RL",
        "protocol_contract": {
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "assistant_carrier": "think-json-v1",
            "history_turns": 4,
            "max_agent_steps": 30,
            "initial_policy": "Qwen3-8B Atomic version26 4k-data 4-epoch checkpoint-6380",
        },
        "method_contract": {
            "source_thread_id": "01a0465e-0f83-7873-8e4c-f150f12f34d7",
            "credit_assignment": "saam-asymmetric-error",
            "error_penalty": 1.0,
            "result_reward_profile": "four-level",
            "terminal_rewards": {
                "correct_clean": 1.5,
                "correct_recovered_error": 1.0,
                "wrong_clean": -0.5,
                "wrong_error_or_policy_failure": -1.0,
            },
            "span_balance_alpha": 0.5,
            "reason_token_loss_weight": 0.5,
            "tool_token_loss_weight": 0.5,
            "pcgrad": False,
        },
        "selection_contract": {
            "total_records": args.total_records,
            "mixed_records": args.mixed_records,
            "manual_homogeneous_records": len(manual_ids),
            "manual_all_correct": len(correct_ids),
            "manual_all_wrong": len(wrong_ids),
            "old_s1_clean_mixed_available": len(old_mixed),
            "new_clean_mixed_available": len(new_mixed),
            "mixed_union_before_manual_overlap": len(old_mixed | new_mixed),
            "manual_tasks_also_mixed_in_new_screen": manual_new_mixed_overlap,
            "screen_trajectories_reused": False,
            "fresh_online_rollouts_required": True,
            "mixed_selection_rule": "keep all old-S1 mixed and all non-BIRD new mixed; fill exact 600 with nonoverlapping adaptive BIRD mixed, preferring SQL score>=2 then DB-balanced low-score calibration",
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "distribution": _distribution(rows, annotations),
        "profiling": {
            "profile_version": PROFILE_VERSION,
            "difficulty_version": DIFFICULTY_VERSION,
            "sampling_only": True,
            "gold_sql_model_visible": False,
        },
        "inputs": {
            "old_tasks": {"file": args.old_tasks.name, "records": len(old_tasks), "sha256": _sha256(args.old_tasks)},
            "old_index": {"file": args.old_index.name, "records": len(old_outcomes), "sha256": _sha256(args.old_index)},
            "manual_selection": {"file": args.manual_selection.name, "records": len(manual_ids), "sha256": _sha256(args.manual_selection)},
            "new_screen_shards": new_shards,
        },
        "outputs": {
            "tasks": {"file": tasks_output.name, "records": len(rows), "sha256": _sha256(tasks_output)},
            "manual_audit": {"file": manual_audit.name, "records": len(manual_ids), "sha256": _sha256(manual_audit)},
        },
        "a100_project_root": str(args.a100_project_root),
    }
    manifest_path = args.output_dir / "qwen3_8b_atomic_v26_saam_fourlevel_train700_v1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": str(tasks_output), "manifest": str(manifest_path), **manifest["outputs"]["tasks"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-tasks", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument(
        "--screen-shard",
        nargs=3,
        action="append",
        metavar=("LABEL", "TASKS", "INDEX"),
        required=True,
    )
    parser.add_argument("--manual-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixed-records", type=int, default=600)
    parser.add_argument("--total-records", type=int, default=700)
    parser.add_argument(
        "--a100-project-root",
        type=Path,
        default=Path("/home/dengyan/tabular_rl_project"),
    )
    args = parser.parse_args()
    if args.mixed_records != 600 or args.total_records != 700:
        parser.error("this frozen builder requires exactly 600 mixed + 100 manual = 700")
    build(args)


if __name__ == "__main__":
    main()
