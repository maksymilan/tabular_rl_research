#!/usr/bin/env python3
"""Freeze the 32 unscreened train tasks for the early-stop behavior gate.

Selection is the ordered set difference between the original, frozen 600-task
universe and the task identities whose *group files* are byte-bound by the
early-stop-568 manifest.  Group contents, rewards, correctness, and the
manifest's reward-derived selection fields are deliberately never consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "qwen3-v26-earlystop-missing32-behavior-cohort-v1"
STATUS = "frozen_reward_blind_ordered_complement"
EXPECTED_SOURCE_TASKS_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
EXPECTED_EARLYSTOP_MANIFEST_SHA256 = (
    "e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad"
)
EXPECTED_SOURCE_RECORDS = 600
EXPECTED_SCREENED_RECORDS = 568
EXPECTED_OUTPUT_RECORDS = 32
EXPECTED_MISSING32_TASKS_SHA256 = (
    "014e8ddc5cd6948510c9ef8dde0067915522b49d99e31472b75f4afaacde97bd"
)
EXPECTED_CHECKPOINT6_COHORT_MANIFEST_SHA256 = (
    "9e013ce2d0adb72de8c72e61ac7cd644e3e0b64c2b8d09915fd0496f29433340"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_jsonl(payload: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def task_id(row: Mapping[str, Any]) -> str:
    for field in ("example_id", "instance_id", "task_id", "trajectory_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    index = row.get("example_index")
    if type(index) is not int:
        raise ValueError("task has no stable identity")
    return f"bird_train_{index:05d}"


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _require_hex_digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def derive(
    source_rows: Sequence[dict[str, Any]],
    earlystop_manifest: Mapping[str, Any],
    *,
    source_tasks_sha256: str,
    earlystop_manifest_sha256: str,
    evaluation_arm: str = "checkpoint-6",
    evaluation_seed: int = 20260817,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evaluation_arm not in {"checkpoint-4", "checkpoint-6"}:
        raise ValueError("evaluation arm must be checkpoint-4 or checkpoint-6")
    expected_seed = 20260816 if evaluation_arm == "checkpoint-4" else 20260817
    if evaluation_seed != expected_seed:
        raise ValueError(f"{evaluation_arm} evaluation seed must be {expected_seed}")
    if source_tasks_sha256 != EXPECTED_SOURCE_TASKS_SHA256:
        raise ValueError("original 600-task SHA-256 mismatch")
    if earlystop_manifest_sha256 != EXPECTED_EARLYSTOP_MANIFEST_SHA256:
        raise ValueError("early-stop-568 manifest SHA-256 mismatch")
    if len(source_rows) != EXPECTED_SOURCE_RECORDS:
        raise ValueError("original task universe must contain exactly 600 rows")

    ordered_ids = [task_id(row) for row in source_rows]
    if len(set(ordered_ids)) != EXPECTED_SOURCE_RECORDS:
        raise ValueError("original task identities are not unique")
    for identity, row in zip(ordered_ids, source_rows, strict=True):
        if type(row.get("example_index")) is not int:
            raise ValueError(f"task {identity} has no integer example_index")
        if not all(
            isinstance(row.get(field), str) and bool(row[field].strip())
            for field in ("db_id", "db_path", "question", "gold_sql")
        ):
            raise ValueError(f"task {identity} lacks required execution fields")

    if earlystop_manifest.get("schema_version") != "qwen3-v26-earlystop-mixed-grpo-cohort-v1":
        raise ValueError("unsupported early-stop manifest schema")
    if earlystop_manifest.get("status") != "frozen_operator_requested_earlystop_mixed180":
        raise ValueError("early-stop manifest status is not frozen")
    decision = earlystop_manifest.get("operator_decision") or {}
    if (
        decision.get("completed_groups") != EXPECTED_SCREENED_RECORDS
        or decision.get("missing_groups") != EXPECTED_OUTPUT_RECORDS
        or decision.get("stopped_before_full_600") is not True
    ):
        raise ValueError("early-stop completed/missing group contract mismatch")
    inputs = earlystop_manifest.get("inputs") or {}
    declared_tasks = inputs.get("tasks") or {}
    if (
        declared_tasks.get("sha256") != EXPECTED_SOURCE_TASKS_SHA256
        or declared_tasks.get("records") != EXPECTED_SOURCE_RECORDS
    ):
        raise ValueError("early-stop manifest does not bind the original 600 tasks")

    # This is the only early-stop screening field used for selection.  Values
    # are checked only as opaque digests; no group file is opened and no reward,
    # correctness, histogram, candidate, selected, or held-out field is read.
    group_hashes = (earlystop_manifest.get("screen") or {}).get("group_sha256")
    if not isinstance(group_hashes, dict):
        raise ValueError("early-stop manifest lacks screen.group_sha256")
    screened_ids = set(group_hashes)
    if len(screened_ids) != EXPECTED_SCREENED_RECORDS:
        raise ValueError("early-stop manifest must bind exactly 568 group identities")
    unknown = screened_ids.difference(ordered_ids)
    if unknown:
        raise ValueError(f"screened identities are outside the 600-task universe: {sorted(unknown)[:3]}")
    for identity, digest in group_hashes.items():
        _require_hex_digest(digest, label=f"screen.group_sha256[{identity}]")

    missing_ids = [identity for identity in ordered_ids if identity not in screened_ids]
    if len(missing_ids) != EXPECTED_OUTPUT_RECORDS:
        raise ValueError("ordered complement must contain exactly 32 identities")
    missing_set = set(missing_ids)
    output_rows = [
        dict(row)
        for identity, row in zip(ordered_ids, source_rows, strict=True)
        if identity in missing_set
    ]
    output_bytes = canonical_jsonl(output_rows)
    db_counts: dict[str, int] = {}
    for row in output_rows:
        db_id = str(row["db_id"])
        db_counts[db_id] = db_counts.get(db_id, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scope": {
            "dataset_split": "BIRD-train",
            "diagnostic_only": True,
            "formal_dev_consumed": False,
            "formal_checkpoint_selection_authority": False,
        },
        "selection": {
            "rule": "ordered complement of screen.group_sha256 identities in frozen original600",
            "reward_blind": True,
            "group_files_opened": 0,
            "reward_or_correctness_fields_read": 0,
            "source_records": EXPECTED_SOURCE_RECORDS,
            "screened_records": EXPECTED_SCREENED_RECORDS,
            "missing_records": EXPECTED_OUTPUT_RECORDS,
            "ordered_task_ids": missing_ids,
            "database_cluster_counts": dict(sorted(db_counts.items())),
        },
        "inputs": {
            "source_tasks": {
                "sha256": source_tasks_sha256,
                "records": len(source_rows),
            },
            "earlystop_manifest": {
                "sha256": earlystop_manifest_sha256,
                "screen_group_identity_count": len(screened_ids),
            },
        },
        "output": {
            "records": len(output_rows),
            "sha256": sha256_bytes(output_bytes),
        },
        "evaluation_contract": {
            "arms": ["sft1", evaluation_arm],
            "group_size": 8,
            "trajectories_per_arm": 256,
            "seed": evaluation_seed,
            "temperature": 0.8,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
        },
    }
    return output_rows, manifest


def publish(output_dir: Path, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.next-", dir=output_dir.parent))
    try:
        tasks_path = temporary / "missing32.jsonl"
        tasks_path.write_bytes(canonical_jsonl(rows))
        manifest = json.loads(json.dumps(manifest))
        manifest["output"]["path"] = str((output_dir / "missing32.jsonl").resolve())
        manifest_path = temporary / "missing32_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(tasks_path, 0o444)
        os.chmod(manifest_path, 0o444)
        temporary.rename(output_dir)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise


def derive_checkpoint4_sibling(
    source_tasks_bytes: bytes,
    source_manifest_bytes: bytes,
    *,
    manifest_tasks_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Derive the seed-20260816 checkpoint-4 sibling from the frozen cp6 cohort.

    The exact cp6 task and manifest digests are the immutable input contract.
    Only the arm, seed, and model-host absolute output path may differ.  This
    lets a deployment stage the sibling without reopening reward-bearing group
    files or mutating the already frozen checkpoint-6 directory.
    """

    if not manifest_tasks_path.is_absolute():
        raise ValueError("checkpoint-4 manifest task path must be absolute")
    if sha256_bytes(source_tasks_bytes) != EXPECTED_MISSING32_TASKS_SHA256:
        raise ValueError("frozen missing32 task SHA-256 mismatch")
    if sha256_bytes(source_manifest_bytes) != EXPECTED_CHECKPOINT6_COHORT_MANIFEST_SHA256:
        raise ValueError("frozen checkpoint-6 cohort manifest SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_bytes)
    source_rows = parse_jsonl(source_tasks_bytes, path=Path("frozen-missing32.jsonl"))
    if len(source_rows) != EXPECTED_OUTPUT_RECORDS:
        raise ValueError("frozen missing32 cohort must contain exactly 32 rows")
    ordered_ids = [task_id(row) for row in source_rows]
    if source_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen missing32 cohort schema")
    if source_manifest.get("status") != STATUS:
        raise ValueError("frozen missing32 cohort status mismatch")
    if source_manifest.get("selection", {}).get("ordered_task_ids") != ordered_ids:
        raise ValueError("frozen missing32 cohort task ordering mismatch")
    if source_manifest.get("output", {}).get("records") != EXPECTED_OUTPUT_RECORDS:
        raise ValueError("frozen missing32 cohort record count mismatch")
    if source_manifest.get("output", {}).get("sha256") != EXPECTED_MISSING32_TASKS_SHA256:
        raise ValueError("frozen missing32 cohort output digest mismatch")
    source_contract = source_manifest.get("evaluation_contract") or {}
    if source_contract.get("arms") != ["sft1", "checkpoint-6"]:
        raise ValueError("sibling source must be the checkpoint-6 cohort")
    if source_contract.get("seed") != 20260817:
        raise ValueError("sibling source checkpoint-6 seed mismatch")

    sibling = json.loads(json.dumps(source_manifest))
    sibling["evaluation_contract"]["arms"] = ["sft1", "checkpoint-4"]
    sibling["evaluation_contract"]["seed"] = 20260816
    sibling["output"]["path"] = str(manifest_tasks_path)
    return source_tasks_bytes, sibling


def publish_checkpoint4_sibling(
    source_cohort_dir: Path,
    output_dir: Path,
    *,
    manifest_tasks_path: Path,
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing existing output directory: {output_dir}")
    source_tasks = source_cohort_dir / "missing32.jsonl"
    source_manifest = source_cohort_dir / "missing32_manifest.json"
    if not source_tasks.is_file() or source_tasks.is_symlink():
        raise ValueError("frozen checkpoint-6 tasks are absent or symlinked")
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise ValueError("frozen checkpoint-6 manifest is absent or symlinked")
    task_bytes, sibling = derive_checkpoint4_sibling(
        source_tasks.read_bytes(),
        source_manifest.read_bytes(),
        manifest_tasks_path=manifest_tasks_path,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.next-", dir=output_dir.parent))
    try:
        (temporary / "missing32.jsonl").write_bytes(task_bytes)
        (temporary / "missing32_manifest.json").write_text(
            json.dumps(sibling, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary / "missing32.jsonl", 0o444)
        os.chmod(temporary / "missing32_manifest.json", 0o444)
        temporary.rename(output_dir)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise


def verify_checkpoint4_sibling(
    source_cohort_dir: Path,
    output_dir: Path,
    *,
    manifest_tasks_path: Path,
) -> dict[str, Any]:
    source_tasks = source_cohort_dir / "missing32.jsonl"
    source_manifest = source_cohort_dir / "missing32_manifest.json"
    observed_tasks = output_dir / "missing32.jsonl"
    observed_manifest = output_dir / "missing32_manifest.json"
    for path, label in (
        (source_tasks, "frozen checkpoint-6 tasks"),
        (source_manifest, "frozen checkpoint-6 manifest"),
        (observed_tasks, "checkpoint-4 sibling tasks"),
        (observed_manifest, "checkpoint-4 sibling manifest"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} are absent or symlinked")
    expected_tasks, expected_manifest = derive_checkpoint4_sibling(
        source_tasks.read_bytes(),
        source_manifest.read_bytes(),
        manifest_tasks_path=manifest_tasks_path,
    )
    if observed_tasks.read_bytes() != expected_tasks:
        raise ValueError("checkpoint-4 sibling task bytes differ from frozen missing32")
    if json.loads(observed_manifest.read_bytes()) != expected_manifest:
        raise ValueError("checkpoint-4 sibling manifest differs from canonical contract")
    return expected_manifest


def verify(
    output_dir: Path,
    source_tasks: Path,
    earlystop_manifest_path: Path,
    *,
    evaluation_arm: str,
    evaluation_seed: int,
) -> dict[str, Any]:
    tasks_path = output_dir / "missing32.jsonl"
    manifest_path = output_dir / "missing32_manifest.json"
    if not tasks_path.is_file() or tasks_path.is_symlink():
        raise ValueError("missing32 tasks are absent or symlinked")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("missing32 manifest is absent or symlinked")
    source_bytes = source_tasks.read_bytes()
    earlystop_bytes = earlystop_manifest_path.read_bytes()
    expected_rows, expected_manifest = derive(
        parse_jsonl(source_bytes, path=source_tasks),
        json.loads(earlystop_bytes),
        source_tasks_sha256=sha256_bytes(source_bytes),
        earlystop_manifest_sha256=sha256_bytes(earlystop_bytes),
        evaluation_arm=evaluation_arm,
        evaluation_seed=evaluation_seed,
    )
    observed_rows = parse_jsonl(tasks_path.read_bytes(), path=tasks_path)
    if canonical_jsonl(observed_rows) != canonical_jsonl(expected_rows):
        raise ValueError("missing32 task bytes differ from reward-blind complement")
    observed_manifest = json.loads(manifest_path.read_bytes())
    expected_manifest["output"]["path"] = str(tasks_path.resolve())
    if observed_manifest != expected_manifest:
        raise ValueError("missing32 manifest differs from canonical contract")
    return observed_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path)
    parser.add_argument("--earlystop-manifest", type=Path)
    parser.add_argument("--source-cohort-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-tasks-path", type=Path)
    parser.add_argument("--checkpoint4-sibling", action="store_true")
    parser.add_argument(
        "--evaluation-arm", choices=("checkpoint-4", "checkpoint-6"), default="checkpoint-6"
    )
    parser.add_argument("--evaluation-seed", type=int, default=20260817)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.checkpoint4_sibling:
            if args.source_cohort_dir is None or args.manifest_tasks_path is None:
                raise ValueError(
                    "checkpoint-4 sibling mode requires --source-cohort-dir and --manifest-tasks-path"
                )
            if args.source_tasks is not None or args.earlystop_manifest is not None:
                raise ValueError("checkpoint-4 sibling mode does not accept original input flags")
            if args.evaluation_arm != "checkpoint-6" or args.evaluation_seed != 20260817:
                raise ValueError("checkpoint-4 sibling mode has an intrinsic arm/seed contract")
            if args.verify:
                result = verify_checkpoint4_sibling(
                    args.source_cohort_dir.resolve(),
                    args.output_dir.resolve(),
                    manifest_tasks_path=args.manifest_tasks_path,
                )
            else:
                publish_checkpoint4_sibling(
                    args.source_cohort_dir.resolve(),
                    args.output_dir.resolve(),
                    manifest_tasks_path=args.manifest_tasks_path,
                )
                result = {"status": STATUS}
            print(json.dumps({"status": result["status"], "records": 32}, sort_keys=True))
            return 0
        if args.source_tasks is None or args.earlystop_manifest is None:
            raise ValueError("standard mode requires --source-tasks and --earlystop-manifest")
        if args.source_cohort_dir is not None or args.manifest_tasks_path is not None:
            raise ValueError("standard mode does not accept checkpoint-4 sibling input flags")
        if args.verify:
            result = verify(
                args.output_dir.resolve(), args.source_tasks.resolve(), args.earlystop_manifest.resolve(),
                evaluation_arm=args.evaluation_arm, evaluation_seed=args.evaluation_seed,
            )
        else:
            source_bytes = args.source_tasks.read_bytes()
            earlystop_bytes = args.earlystop_manifest.read_bytes()
            rows, result = derive(
                parse_jsonl(source_bytes, path=args.source_tasks),
                json.loads(earlystop_bytes),
                source_tasks_sha256=sha256_bytes(source_bytes),
                earlystop_manifest_sha256=sha256_bytes(earlystop_bytes),
                evaluation_arm=args.evaluation_arm,
                evaluation_seed=args.evaluation_seed,
            )
            publish(args.output_dir.resolve(), rows, result)
        print(json.dumps({"status": result["status"], "records": 32}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"missing32 cohort blocked: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
