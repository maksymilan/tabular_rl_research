#!/usr/bin/env python3
"""Promote a pending BIRD suite only after both mandatory process-RL gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SHORTCUTS = {
    "bird_train_00541",
    "bird_train_02918",
    "bird_train_03663",
    "bird_train_04696",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending-manifest", type=Path, required=True)
    parser.add_argument("--generation-audit", type=Path, required=True)
    parser.add_argument("--known-correct-replay-jsonl", type=Path, required=True)
    parser.add_argument("--known-correct-summary", type=Path, required=True)
    parser.add_argument("--shortcut-audit", type=Path, required=True)
    parser.add_argument("--grounding-summary", type=Path, required=True)
    parser.add_argument("--grounding-reviews", type=Path, required=True)
    parser.add_argument("--output-quality-audit", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.output_quality_audit, args.output_manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    manifest = read_json(args.pending_manifest)
    require(
        manifest.get("schema_version") == "process-counterfactual-suite-v2",
        "pending manifest has the wrong schema",
    )
    require(
        manifest.get("denotation_comparison") == "bird-set",
        "pending manifest must use bird-set",
    )
    require(
        (manifest.get("quality_gate") or {}).get("status") == "pending",
        "input manifest is not pending",
    )
    tasks = manifest.get("tasks")
    require(isinstance(tasks, dict) and len(tasks) == 23, "manifest must contain 23 tasks")
    for task_id, task in tasks.items():
        require(
            task_id.startswith("spider_train_"),
            f"trainer task ID is not loader-aligned: {task_id}",
        )
        require(
            isinstance(task.get("trajectory_task_id"), str)
            and task["trajectory_task_id"].startswith("bird_train_"),
            f"trajectory task ID is missing for {task_id}",
        )
        require(
            task.get("min_informative_databases") == 2,
            f"{task_id} must require both counterfactual databases",
        )
        require(
            isinstance(task.get("databases"), list) and len(task["databases"]) == 2,
            f"{task_id} must contain exactly two databases",
        )

    generation = read_json(args.generation_audit)
    require(
        generation.get("schema_version") == "process-counterfactual-generation-audit-v1",
        "generation audit has the wrong schema",
    )
    require(len(generation.get("tasks") or []) == 23, "generation audit must contain 23 tasks")
    require(
        (manifest.get("generator") or {}).get("generation_audit_sha256")
        == sha256_file(args.generation_audit),
        "manifest generator is not bound to the supplied generation audit",
    )

    database_hashes_verified = 0
    for task_id, task in tasks.items():
        for database in task["databases"]:
            path = Path(database["path"]).resolve()
            require(path.is_file(), f"counterfactual database is missing: {path}")
            require(
                sha256_file(path) == database["sha256"],
                f"counterfactual database hash mismatch: {path}",
            )
            database_hashes_verified += 1
    require(database_hashes_verified == 46, "exactly 46 database hashes must be verified")

    replay_records = read_jsonl(args.known_correct_replay_jsonl)
    replay_summary = read_json(args.known_correct_summary)
    require(len(replay_records) == 105, "known-correct replay audit must contain 105 records")
    replay_passed = sum(
        bool(record.get("replay") and record["replay"].get("passed"))
        for record in replay_records
    )
    require(replay_passed == 102, "frozen known-correct replay count must be 102/105")
    require(
        replay_summary.get("source_correct_trajectories") == 105
        and replay_summary.get("counterfactual_passed_trajectories") == 102,
        "known-correct replay summary disagrees with the JSONL audit",
    )
    require(
        replay_summary.get("provisional_known_correct_gate_passed") is True,
        "known-correct counterfactual gate did not pass",
    )
    require(
        replay_summary.get("tasks_without_passing_program") == [],
        "every task must have a known-correct passing program",
    )
    require(
        set(replay_summary.get("tasks") or {}) == set(tasks),
        "known-correct replay task IDs differ from the manifest",
    )
    require(
        all(
            int(task["counterfactual_passed_trajectories"]) >= 1
            and int(task["audit_errors"]) == 0
            for task in replay_summary["tasks"].values()
        ),
        "each task must have a passing trajectory and zero audit errors",
    )

    shortcut = read_json(args.shortcut_audit)
    require(
        shortcut.get("schema_version")
        == "process-counterfactual-shortcut-regression-audit-v1",
        "shortcut audit has the wrong schema",
    )
    require(shortcut.get("denotation_comparison") == "bird-set", "shortcut audit must use bird-set")
    require(shortcut.get("gate_passed") is True, "shortcut regression gate did not pass")
    require(shortcut.get("failed_to_reject") == [], "one or more shortcuts were not rejected")
    require(
        set(shortcut.get("expected_shortcuts") or []) == EXPECTED_SHORTCUTS,
        "shortcut regression set differs from the frozen four classes",
    )
    require(shortcut.get("rejected_shortcuts") == 4, "all four shortcuts must be rejected")

    grounding = read_json(args.grounding_summary)
    require(
        grounding.get("audit_kind") == "codex-grounding-change-audit-v1",
        "grounding summary has the wrong audit kind",
    )
    require(grounding.get("status") == "completed", "grounding audit is incomplete")
    require(
        grounding.get("reviewer") == "codex-primary-structured-audit",
        "grounding audit reviewer differs from the user-designated Codex audit",
    )
    require(
        grounding.get("grounding_edges") == 63
        and (grounding.get("grounding_edge_labels") or {}).get("valid") == 63
        and (grounding.get("grounding_edge_labels") or {}).get("invalid") == 0,
        "grounding edge labels must be 63 valid and 0 invalid",
    )
    require(
        grounding.get("grounding_edge_decided_precision") == 1.0
        and grounding.get("reported_missing_edges") == 0,
        "grounding precision/missing-edge gate did not pass",
    )
    require(
        (grounding.get("gate") or {}).get("changed_extractor_edge_precision_passed") is True,
        "changed-extractor grounding gate did not pass",
    )
    grounding_reviews_sha = sha256_file(args.grounding_reviews)
    require(
        grounding.get("reviews_output_sha256") == grounding_reviews_sha,
        "grounding review file hash differs from its frozen summary",
    )
    require(
        set(grounding.get("invalid_final_dependencies") or []) == {"bird_train_04696"},
        "frozen grounding audit must identify the 04696 terminal shortcut",
    )
    shortcut_04696 = next(
        record
        for record in shortcut["records"]
        if record["trajectory_id"] == "bird_train_04696"
    )
    require(
        shortcut_04696.get("shortcut_rejected") is True,
        "the additional 04696 terminal shortcut was not rejected",
    )

    failed_known_correct = sorted(
        record["trajectory_id"]
        for record in replay_records
        if not (record.get("replay") and record["replay"].get("passed"))
    )
    quality_audit = {
        "schema_version": "process-rl-mandatory-gates-v1",
        "status": "passed",
        "denotation_comparison": "bird-set",
        "pending_manifest": {
            "path": str(args.pending_manifest.resolve()),
            "sha256": sha256_file(args.pending_manifest),
        },
        "deterministic_completeness_gate": {
            "status": "passed",
            "tasks": 23,
            "counterfactual_databases": 46,
            "database_hashes_verified": database_hashes_verified,
            "source_correct_trajectories": 105,
            "counterfactual_passing_trajectories": 102,
            "counterfactual_pass_rate": 102 / 105,
            "tasks_with_known_correct_passing_program": 23,
            "expected_incomplete_source_correct_trajectories": failed_known_correct,
            "shortcut_regressions_rejected": 4,
            "generation_audit": {
                "path": str(args.generation_audit.resolve()),
                "sha256": sha256_file(args.generation_audit),
            },
            "known_correct_replay_jsonl": {
                "path": str(args.known_correct_replay_jsonl.resolve()),
                "sha256": sha256_file(args.known_correct_replay_jsonl),
            },
            "known_correct_summary": {
                "path": str(args.known_correct_summary.resolve()),
                "sha256": sha256_file(args.known_correct_summary),
            },
            "shortcut_audit": {
                "path": str(args.shortcut_audit.resolve()),
                "sha256": sha256_file(args.shortcut_audit),
            },
        },
        "independent_grounding_edge_precision_gate": {
            "status": "passed",
            "reviewer": grounding["reviewer"],
            "trajectories": grounding["trajectories"],
            "grounding_edges": grounding["grounding_edges"],
            "valid_edges": grounding["grounding_edge_labels"]["valid"],
            "invalid_edges": grounding["grounding_edge_labels"]["invalid"],
            "decided_precision": grounding["grounding_edge_decided_precision"],
            "reported_missing_edges": grounding["reported_missing_edges"],
            "grounding_summary": {
                "path": str(args.grounding_summary.resolve()),
                "sha256": sha256_file(args.grounding_summary),
            },
            "grounding_reviews": {
                "path": str(args.grounding_reviews.resolve()),
                "sha256": grounding_reviews_sha,
            },
        },
        "process_rl_admission": "allowed",
    }
    args.output_quality_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_quality_audit.write_text(
        json.dumps(quality_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quality_audit_sha = sha256_file(args.output_quality_audit)

    promoted = dict(manifest)
    promoted["quality_gate"] = {
        "status": "passed",
        "audit_path": str(args.output_quality_audit.resolve()),
        "audit_sha256": quality_audit_sha,
        "mandatory_gates": [
            "deterministic_completeness",
            "independent_grounding_edge_precision",
        ],
    }
    args.output_manifest.write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "promoted",
                "quality_audit": str(args.output_quality_audit),
                "quality_audit_sha256": quality_audit_sha,
                "manifest": str(args.output_manifest),
                "manifest_sha256": sha256_file(args.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
