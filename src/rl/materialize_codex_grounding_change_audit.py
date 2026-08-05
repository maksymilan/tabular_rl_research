#!/usr/bin/env python3
"""Materialize the human-directed Codex audit of changed grounding packages."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


AUDIT_PROTOCOL = "grounding-review-v4-visible-literal-copy"
REVIEWER = "codex-primary-structured-audit"

FINAL_REASONS = {
    "bird_train_04677": "Filters shipment 1233, copies its truck id, and returns that truck's model year.",
    "bird_train_03216": "Filters the three named systems, counts criteria per system, then averages those counts.",
    "bird_train_05873": "Maps both solution ids to repository ids, reads both star counts, then computes percent change.",
    "bird_train_05552": "Filters games id 13 and returns its games_name column.",
    "bird_train_00262": "Maps Mark Hammel to person_id, filters crew rows, and returns job.",
    "bird_train_05443": "Maps the named author to papers, orders them by year, and returns the latest title.",
    "bird_train_05741": "Reads Australia's capital, filters organizations by that city, and returns both requested fields.",
    "bird_train_00739": "Filters the exact ball, maps its striker id to Player, and returns Player_Name.",
    "bird_train_04109": "Filters season 2012, maps Man_of_the_Series to Player, and joins the country name.",
    "bird_train_02779": "Maps Quebec to StateProvinceID and counts distinct TaxType values in its tax rows.",
    "bird_train_02265": "Maps Oak Park to city id, shipment to customer id, and returns the customer name.",
    "bird_train_03361": "Excludes capital-city rows and selects the remaining city with maximum population.",
    "bird_train_03877": "Filters the repository statistics, maps its id to solutions, and averages ProcessedTime.",
    "bird_train_02600": "Maps the named supplier to partsupp rows, joins parts, and returns every part name.",
    "bird_train_02869": "Filters 2013 SellStartDate rows, joins ProductVendor and Vendor, and projects both names.",
    "bird_train_03534": "Computes the global minimum inspection score, joins its business, and returns the name.",
    "bird_train_06577": "Selects the maximum-duration trip, copies start_station_id, and returns station coordinates.",
    "bird_train_00429": "Counts teachers per course, filters counts above four, then counts the retained courses.",
    "bird_train_05875": "Selects the repository with maximum forks and sums ProcessedTime over its solutions.",
    "bird_train_04696": (
        "Invalid: the final project hardcodes 'àbac-xinès' instead of deriving the selected pair "
        "from top_003; swapping pair occurrences would not change the output."
    ),
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


def flatten(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in flatten(child)]
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def resolve_path(value: Any, path: list[Any]) -> Any:
    current = value
    for component in path:
        current = current[component]
    return current


def validate_schema_edge(edge: dict[str, Any]) -> str:
    if edge.get("from_tool") != "describe_table":
        raise ValueError(f"{edge['edge_id']}: schema edge source is not describe_table")
    target = edge.get("target") or {}
    table_name = target.get("table")
    observed = next(
        (
            table
            for table in (edge.get("from_output") or {}).get("tables") or []
            if table.get("table_name") == table_name
        ),
        None,
    )
    if observed is None:
        raise ValueError(f"{edge['edge_id']}: target table is absent from source schema")
    observed_columns = {column.get("name") for column in observed.get("columns") or []}
    target_columns = set(target.get("columns") or [])
    if not target_columns <= observed_columns:
        raise ValueError(f"{edge['edge_id']}: target columns are absent from source schema")
    columns = ", ".join(sorted(target_columns)) if target_columns else "base-table schema"
    return f"Source schema contains {table_name}.{columns} consumed by {edge.get('to_tool')}."


def validate_row_edge(edge: dict[str, Any]) -> str:
    target = edge.get("target") or {}
    matches = target.get("column_matches") or []
    if not matches:
        raise ValueError(f"{edge['edge_id']}: row edge has no column_matches")
    visible_values = flatten(edge.get("from_output") or {})
    arguments = edge.get("to_arguments") or {}
    checked: list[str] = []
    for match in matches:
        value = match.get("value")
        if value not in visible_values:
            raise ValueError(f"{edge['edge_id']}: copied value {value!r} is not source-visible")
        argument_path = match.get("argument_path")
        if not isinstance(argument_path, list):
            raise ValueError(f"{edge['edge_id']}: copied value has no exact argument_path")
        if resolve_path(arguments, argument_path) != value:
            raise ValueError(f"{edge['edge_id']}: argument_path does not resolve to copied value")
        replay_binding = match.get("replay_binding")
        if replay_binding is not None:
            rows = (edge.get("from_output") or {}).get("rows") or []
            row = rows[int(replay_binding["row_index"])]
            if row[int(replay_binding["column_index"])] != value:
                raise ValueError(f"{edge['edge_id']}: replay_binding does not resolve to copied value")
        checked.append(
            f"{match.get('source_column')}={value!r}->{match.get('target_column')}"
        )
    return "Exact source-visible copy verified: " + ", ".join(checked) + "."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--reviews-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    packages = read_jsonl(args.packages)
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    expected_ids = set(manifest["selection"]["changed_trajectory_ids"])
    expected_ids.update(manifest["selection"]["row_control_trajectory_ids"])
    expected_ids.update(manifest["selection"]["schema_control_trajectory_ids"])
    observed_ids = {str(package["trajectory_id"]) for package in packages}
    if observed_ids != expected_ids:
        raise ValueError("review packages do not exactly match the frozen selection")
    if observed_ids != FINAL_REASONS.keys():
        raise ValueError("manual final-dependency decisions do not cover the frozen selection")

    reviews: list[dict[str, Any]] = []
    role_counts: collections.Counter[str] = collections.Counter()
    final_counts: collections.Counter[str] = collections.Counter()
    for package in packages:
        trajectory_id = str(package["trajectory_id"])
        edge_reviews = []
        for edge in package.get("grounding_edges") or []:
            role = str(edge.get("role"))
            if role == "schema_observation":
                reason = validate_schema_edge(edge)
            elif role == "row_observation":
                reason = validate_row_edge(edge)
            else:
                raise ValueError(f"{edge['edge_id']}: unsupported grounding role {role!r}")
            role_counts[role] += 1
            edge_reviews.append(
                {"edge_id": edge["edge_id"], "label": "valid", "reason": reason}
            )
        final_label = "invalid" if trajectory_id == "bird_train_04696" else "valid"
        final_counts[final_label] += 1
        reviews.append(
            {
                "trajectory_id": trajectory_id,
                "reviewer": REVIEWER,
                "review_protocol_version": AUDIT_PROTOCOL,
                "review": {
                    "trajectory_id": trajectory_id,
                    "final_dependency": {
                        "label": final_label,
                        "reason": FINAL_REASONS[trajectory_id],
                    },
                    "edges": edge_reviews,
                    "missing_edges": [],
                    "overall": "pass" if final_label == "valid" else "fail",
                },
            }
        )

    args.reviews_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.reviews_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviews),
        encoding="utf-8",
    )
    summary = {
        "audit_kind": "codex-grounding-change-audit-v1",
        "status": "completed",
        "reviewer": REVIEWER,
        "review_protocol_version": AUDIT_PROTOCOL,
        "inputs": {
            "packages": str(args.packages),
            "packages_sha256": sha256_file(args.packages),
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
        "trajectories": len(packages),
        "changed_trajectories": len(manifest["selection"]["changed_trajectory_ids"]),
        "control_trajectories": (
            len(manifest["selection"]["row_control_trajectory_ids"])
            + len(manifest["selection"]["schema_control_trajectory_ids"])
        ),
        "grounding_edges": sum(role_counts.values()),
        "grounding_edge_roles": dict(sorted(role_counts.items())),
        "grounding_edge_labels": {"valid": sum(role_counts.values()), "invalid": 0},
        "grounding_edge_decided_precision": 1.0,
        "reported_missing_edges": 0,
        "final_dependency_labels": dict(sorted(final_counts.items())),
        "invalid_final_dependencies": ["bird_train_04696"],
        "reviews_output": str(args.reviews_output),
        "reviews_output_sha256": sha256_file(args.reviews_output),
        "gate": {
            "changed_extractor_edge_precision_passed": True,
            "process_reward_ready": False,
            "reason": (
                "All selected current grounding edges have exact structural support, but "
                "bird_train_04696 exposes an additional terminal constant-output shortcut and "
                "the required content-hashed counterfactual suites are not yet complete."
            ),
        },
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
