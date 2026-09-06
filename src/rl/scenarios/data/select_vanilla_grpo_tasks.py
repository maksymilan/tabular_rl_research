#!/usr/bin/env python3
"""Freeze a representative, leakage-safe BIRD-train cohort for vanilla GRPO.

The selector deliberately reuses the public-field-only allocation used by the
active BIRD-train evaluation cohort, but produces a separately identified
*training* cohort.  Gold SQL remains in each record for the local Harness and
is never used for selection or ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO_ROOT / "src" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from select_bird_train_baseline import (  # noqa: E402
    FORBIDDEN_SELECTION_FIELDS,
    PUBLIC_SELECTION_FIELDS,
    distribution_audit,
    length_cutpoints,
    read_jsonl,
    select_rows,
    sha256_file,
    task_id,
)


SCHEMA_VERSION = "bird-train-vanilla-grpo-cohort-v1"
EXCLUSION_ID_FIELDS = ("example_id", "instance_id", "source_episode_id")


def exclusion_task_id(row: Mapping[str, Any], *, source: str) -> str:
    for field in EXCLUSION_ID_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        f"{source}: exclusion row has none of {EXCLUSION_ID_FIELDS!r}"
    )


def exclusion_ids(paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    combined: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        ids = {
            exclusion_task_id(row, source=f"{path}:{index}")
            for index, row in enumerate(rows, start=1)
        }
        combined.update(ids)
        artifacts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(rows),
                "unique_task_ids": len(ids),
                # Keep the leakage boundary self-contained even when an input
                # exclusion artifact lives only on a remote runtime or in /tmp.
                # The source path and hash remain provenance; these ids are the
                # exact semantic exclusion contract used by the selector.
                "task_ids": sorted(ids),
            }
        )
    return combined, artifacts


def remap_database_paths(
    rows: Sequence[dict[str, Any]], remote_db_root: Path | None
) -> list[dict[str, Any]]:
    if remote_db_root is None:
        return [dict(row) for row in rows]
    remapped: list[dict[str, Any]] = []
    for row in rows:
        source = Path(str(row.get("db_path") or ""))
        db_id = str(row.get("db_id") or "")
        if not db_id or not source.name:
            raise ValueError(f"{task_id(row)}: missing db_id/db_path")
        if source.parent.name != db_id:
            raise ValueError(
                f"{task_id(row)}: db_path parent {source.parent.name!r} "
                f"does not match db_id {db_id!r}"
            )
        retained = dict(row)
        retained["db_path"] = str(remote_db_root / db_id / source.name)
        metadata = dict(retained.get("metadata") or {})
        metadata["rl_training_cohort"] = SCHEMA_VERSION
        retained["metadata"] = metadata
        remapped.append(retained)
    return remapped


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_cohort(
    *,
    reference_path: Path,
    eligible_path: Path,
    exclusion_paths: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    count: int,
    seed: str,
    remote_db_root: Path | None,
) -> dict[str, Any]:
    reference = read_jsonl(reference_path)
    eligible = read_jsonl(eligible_path)
    invalid_eligible = [
        task_id(row)
        for row in eligible
        if (row.get("metadata") or {}).get("tool_round_trip") != "verified"
    ]
    if invalid_eligible:
        raise ValueError(
            "eligible tasks must have verified tool round-trip status: "
            f"{invalid_eligible[:5]}"
        )

    excluded, exclusion_artifacts = exclusion_ids(exclusion_paths)
    selected, details = select_rows(
        reference,
        eligible,
        excluded,
        count=count,
        seed=seed,
    )
    selected_ids = [task_id(row) for row in selected]
    selected_id_set = set(selected_ids)
    cutpoints = length_cutpoints(reference)
    audit = distribution_audit(reference, selected, cutpoints=cutpoints)
    remapped = remap_database_paths(selected, remote_db_root)
    rendered = jsonl_bytes(remapped)

    gates = {
        "record_count_exact": len(selected) == count,
        "task_ids_unique": len(selected_id_set) == count,
        "overlap_with_all_exclusions_zero": not bool(selected_id_set & excluded),
        "all_reference_databases_covered": (
            {str(row.get("db_id") or "") for row in selected}
            == {str(row.get("db_id") or "") for row in reference}
        ),
        "database_tv_at_most_0_04": (
            audit["axes"]["database"]["total_variation_distance"] <= 0.04
        ),
        "length_quintile_tv_at_most_0_03": (
            audit["axes"]["question_length_quintile"]["total_variation_distance"]
            <= 0.03
        ),
        "knowledge_rate_delta_at_most_0_02": (
            audit["axes"]["external_knowledge"]["total_variation_distance"]
            <= 0.02
        ),
        "joint_knowledge_length_tv_at_most_0_04": (
            audit["axes"]["knowledge_x_length"]["total_variation_distance"]
            <= 0.04
        ),
        "question_length_mean_delta_at_most_0_05": (
            audit["question_length_characters"]["relative_mean_difference"] <= 0.05
        ),
        "remote_paths_bound_when_requested": (
            remote_db_root is None
            or all(
                Path(str(row["db_path"])).is_relative_to(remote_db_root)
                for row in remapped
            )
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"training cohort acceptance gates failed: {failed}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_training_cohort",
        "purpose": (
            "representative BIRD-train on-policy cohort for the Qwen3-8B "
            "atomic vanilla-GRPO baseline"
        ),
        "reference_population": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "records": len(reference),
        },
        "eligibility_population": {
            "path": str(eligible_path),
            "sha256": sha256_file(eligible_path),
            "records": len(eligible),
            "rule": "tool_round_trip=verified and frozen non-empty BIRD-train input",
        },
        "exclusions": exclusion_artifacts,
        "selection": {
            "count": count,
            "seed": seed,
            "public_fields_used": list(PUBLIC_SELECTION_FIELDS),
            "forbidden_fields_not_used": list(FORBIDDEN_SELECTION_FIELDS),
            "gold_used_for_selection_or_order": False,
            "algorithm": (
                "public-field proportional database allocation with knowledge x "
                "question-length stratification and deterministic representative order"
            ),
            "details": details,
        },
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(rendered).hexdigest(),
            "records": len(remapped),
            "unique_databases": len({row["db_id"] for row in remapped}),
            "remote_db_root": str(remote_db_root) if remote_db_root else None,
            "task_ids_in_frozen_order": selected_ids,
        },
        "distribution_audit": audit,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": True,
        "gold_visibility": (
            "gold SQL remains Harness-only and is never rendered to the actor"
        ),
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    write_atomic(output_path, rendered)
    write_atomic(manifest_path, manifest_payload)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument(
        "--seed", default="qwen3-8b-atomic-v26-vanilla-grpo-train600-v1-20260812"
    )
    parser.add_argument("--remote-db-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = freeze_cohort(
        reference_path=args.reference,
        eligible_path=args.eligible,
        exclusion_paths=args.exclude,
        output_path=args.output,
        manifest_path=args.manifest,
        count=args.count,
        seed=args.seed,
        remote_db_root=args.remote_db_root,
    )
    print(
        json.dumps(
            {
                "output": manifest["output"],
                "acceptance_gates": manifest["acceptance_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
