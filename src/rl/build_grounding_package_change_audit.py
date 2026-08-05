#!/usr/bin/env python3
"""Select changed grounding packages and deterministic controls for fresh review."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


LOCATOR_FIELDS = {
    "argument_path",
    "binding_ambiguous",
    "replay_binding",
    "source_column_index",
    "source_row_index",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_locator_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_locator_fields(child)
            for key, child in value.items()
            if key not in LOCATOR_FIELDS
        }
    if isinstance(value, list):
        return [strip_locator_fields(child) for child in value]
    return value


def semantic_edge_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    target = strip_locator_fields(edge.get("target"))
    return (
        str(edge.get("from_step")),
        str(edge.get("to_step")),
        str(edge.get("role")),
        json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def counter_diff(
    old_edges: list[dict[str, Any]],
    new_edges: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    old_counter = collections.Counter(semantic_edge_key(edge) for edge in old_edges)
    new_counter = collections.Counter(semantic_edge_key(edge) for edge in new_edges)
    return list((new_counter - old_counter).elements()), list(
        (old_counter - new_counter).elements()
    )


def stable_order(trajectory_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{trajectory_id}".encode()).hexdigest()


def edge_summary(edge_key: tuple[str, str, str, str]) -> dict[str, Any]:
    from_step, to_step, role, target = edge_key
    return {
        "from_step": from_step,
        "to_step": to_step,
        "role": role,
        "target": json.loads(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-packages", type=Path, required=True)
    parser.add_argument("--new-packages", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--row-controls", type=int, default=6)
    parser.add_argument("--schema-controls", type=int, default=6)
    parser.add_argument("--seed", default="process-grounding-change-audit-v1")
    args = parser.parse_args()

    old_rows = read_jsonl(args.old_packages)
    new_rows = read_jsonl(args.new_packages)
    old = {str(row["trajectory_id"]): row for row in old_rows}
    new = {str(row["trajectory_id"]): row for row in new_rows}
    if len(old) != len(old_rows) or len(new) != len(new_rows):
        raise ValueError("trajectory_id must be unique in both package files")
    if old.keys() != new.keys():
        raise ValueError("old and new package files must contain the same trajectory ids")

    changes: list[dict[str, Any]] = []
    unchanged_row: list[str] = []
    unchanged_schema: list[str] = []
    for trajectory_id in sorted(old):
        old_edges = old[trajectory_id].get("grounding_edges") or []
        new_edges = new[trajectory_id].get("grounding_edges") or []
        additions, removals = counter_diff(old_edges, new_edges)
        if additions or removals:
            changes.append(
                {
                    "trajectory_id": trajectory_id,
                    "old_edge_count": len(old_edges),
                    "new_edge_count": len(new_edges),
                    "semantic_additions": [edge_summary(edge) for edge in additions],
                    "semantic_removals": [edge_summary(edge) for edge in removals],
                }
            )
            continue
        roles = {str(edge.get("role")) for edge in new_edges}
        if "row_observation" in roles:
            unchanged_row.append(trajectory_id)
        elif "schema_observation" in roles:
            unchanged_schema.append(trajectory_id)

    unchanged_row.sort(key=lambda item: stable_order(item, args.seed))
    unchanged_schema.sort(key=lambda item: stable_order(item, args.seed))
    row_controls = unchanged_row[: args.row_controls]
    schema_controls = unchanged_schema[: args.schema_controls]
    changed_ids = [row["trajectory_id"] for row in changes]
    selected_ids = changed_ids + row_controls + schema_controls

    args.ids_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.write_text("".join(f"{item}\n" for item in selected_ids), encoding="utf-8")
    manifest = {
        "audit_kind": "grounding-package-change-audit-v1",
        "status": "external_dual_review_pending",
        "normalization": {
            "ignored_locator_fields": sorted(LOCATOR_FIELDS),
            "rationale": (
                "These fields identify the exact visible-cell/argument binding without changing "
                "the semantic source step, consumer step, role, copied value, or table/column pair."
            ),
        },
        "inputs": {
            "old_packages": str(args.old_packages),
            "old_packages_sha256": sha256_file(args.old_packages),
            "new_packages": str(args.new_packages),
            "new_packages_sha256": sha256_file(args.new_packages),
            "old_trajectory_count": len(old_rows),
            "new_trajectory_count": len(new_rows),
        },
        "selection": {
            "seed": args.seed,
            "changed_count": len(changed_ids),
            "changed_trajectory_ids": changed_ids,
            "row_control_count": len(row_controls),
            "row_control_trajectory_ids": row_controls,
            "schema_control_count": len(schema_controls),
            "schema_control_trajectory_ids": schema_controls,
            "total_selected": len(selected_ids),
            "ids_output": str(args.ids_output),
            "ids_output_sha256": sha256_file(args.ids_output),
        },
        "changes": changes,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
