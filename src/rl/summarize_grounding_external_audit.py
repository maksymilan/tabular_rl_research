#!/usr/bin/env python3
"""Summarize independent Flash/Pro reviews of harness grounding evidence."""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            raise ValueError(f"{label} row is missing trajectory_id")
        if trajectory_id in indexed:
            raise ValueError(f"{label} contains duplicate trajectory_id {trajectory_id}")
        indexed[trajectory_id] = row
    return indexed


def overall(result: dict[str, Any] | None) -> str:
    if not result or not result.get("review"):
        return "error"
    return str(result["review"].get("overall"))


def label_map(result: dict[str, Any] | None) -> dict[str, str]:
    if not result or not result.get("review"):
        return {}
    return {
        edge["edge_id"]: edge["label"]
        for edge in result["review"].get("edges") or []
    }


def component_counts(
    reviews: dict[str, dict[str, Any]],
    trajectory_ids: set[str],
) -> dict[str, Any]:
    final_labels = collections.Counter()
    edge_labels = collections.Counter()
    overall_labels = collections.Counter()
    missing_dependencies = 0
    errors = 0
    for trajectory_id in trajectory_ids:
        result = reviews.get(trajectory_id)
        if not result or not result.get("review"):
            errors += 1
            continue
        review = result["review"]
        final_labels[(review.get("final_dependency") or {}).get("label")] += 1
        overall_labels[review.get("overall")] += 1
        missing_dependencies += len(review.get("missing_edges") or [])
        edge_labels.update(edge.get("label") for edge in review.get("edges") or [])
    decided_edges = edge_labels["valid"] + edge_labels["invalid"]
    return {
        "trajectories": len(trajectory_ids),
        "errors": errors,
        "final_dependency_labels": dict(sorted(final_labels.items())),
        "edge_labels": dict(sorted(edge_labels.items())),
        "overall_labels": dict(sorted(overall_labels.items())),
        "reported_missing_dependencies": missing_dependencies,
        "edge_decided_precision": (
            edge_labels["valid"] / decided_edges if decided_edges else None
        ),
    }


def project_review_edges(
    packages: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    *,
    label: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    projected: dict[str, dict[str, Any]] = {}
    removed_edges = 0
    for trajectory_id, result in reviews.items():
        if trajectory_id not in packages:
            raise ValueError(f"{label} contains unknown trajectory_id {trajectory_id}")
        if not result.get("review"):
            projected[trajectory_id] = result
            continue
        expected = {
            edge["edge_id"]
            for edge in packages[trajectory_id].get("grounding_edges") or []
        }
        observed = {
            edge.get("edge_id")
            for edge in result["review"].get("edges") or []
        }
        if not expected <= observed:
            raise ValueError(
                f"{label} is missing current package edges for {trajectory_id}: "
                f"expected {len(expected)}, observed {len(observed)}"
            )
        normalized = copy.deepcopy(result)
        review = normalized["review"]
        review["edges"] = [
            edge
            for edge in review.get("edges") or []
            if edge.get("edge_id") in expected
        ]
        removed_edges += len(observed - expected)
        component_labels = [
            (review.get("final_dependency") or {}).get("label"),
            *(edge.get("label") for edge in review["edges"]),
        ]
        if "invalid" in component_labels or review.get("missing_edges"):
            review["overall"] = "fail"
        elif "ambiguous" in component_labels:
            review["overall"] = "ambiguous"
        else:
            review["overall"] = "pass"
        projected[trajectory_id] = normalized
    return projected, removed_edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--flash-reviews", type=Path, required=True)
    parser.add_argument("--flash-reviews-override", type=Path, action="append", default=[])
    parser.add_argument("--pro-reviews", type=Path, required=True)
    parser.add_argument("--pro-reviews-override", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    args = parser.parse_args()

    packages = index_unique(read_jsonl(args.packages), "packages")
    flash_base = index_unique(read_jsonl(args.flash_reviews), "flash reviews")
    flash_override: dict[str, dict[str, Any]] = {}
    for path in args.flash_reviews_override:
        flash_override.update(
            index_unique(read_jsonl(path), f"Flash review override {path}")
        )
    pro = index_unique(read_jsonl(args.pro_reviews), "pro reviews")
    for path in args.pro_reviews_override:
        pro.update(index_unique(read_jsonl(path), f"Pro review override {path}"))
    selection = {
        row["trajectory_id"]: row["reasons"]
        for row in read_jsonl(args.selection)
    }
    package_ids = set(packages)
    if set(flash_base) != package_ids:
        raise ValueError(
            "base Flash reviews must cover every package exactly: "
            f"packages={len(package_ids)}, reviews={len(flash_base)}"
        )
    selected_ids = set(selection)
    if not selected_ids <= package_ids:
        raise ValueError("selection contains trajectory ids absent from packages")
    if flash_override and set(flash_override) != selected_ids:
        raise ValueError(
            "Combined Flash override reviews must cover the selected trajectories exactly"
        )
    if set(pro) != selected_ids:
        raise ValueError("Pro reviews must cover the selected trajectories exactly")
    flash = dict(flash_base)
    flash.update(flash_override)
    flash, flash_projected_edges = project_review_edges(
        packages, flash, label="Flash reviews"
    )
    pro, pro_projected_edges = project_review_edges(
        packages, pro, label="Pro reviews"
    )

    confusion = collections.Counter()
    edge_confusion = collections.Counter()
    final_confusion = collections.Counter()
    status = collections.Counter()
    control_pro = collections.Counter()
    control_consensus = collections.Counter()
    reason_outcomes = collections.Counter()
    joint_missing_dependency_trajectories = 0
    cases = []
    for trajectory_id, package in packages.items():
        flash_result = flash.get(trajectory_id)
        pro_result = pro.get(trajectory_id)
        flash_label = overall(flash_result)
        if trajectory_id not in selection:
            if flash_label != "pass":
                raise ValueError(
                    f"unselected trajectory {trajectory_id} has Flash label {flash_label}"
                )
            status["flash_pass_unreviewed"] += 1
            continue
        pro_label = overall(pro_result)
        confusion[f"{flash_label}->{pro_label}"] += 1
        for reason in selection[trajectory_id]:
            reason_outcomes[f"{reason}:{flash_label}->{pro_label}"] += 1
        if "flash_pass_control" in selection[trajectory_id]:
            control_pro[pro_label] += 1
            if flash_label == pro_label == "pass":
                control_consensus["pass"] += 1
            elif flash_label == pro_label == "fail":
                control_consensus["fail"] += 1
            else:
                control_consensus["disputed"] += 1

        if flash_label == pro_label == "pass":
            status["strong_pass"] += 1
        elif flash_label == pro_label == "fail":
            status["confirmed_fail"] += 1
        elif "error" in {flash_label, pro_label}:
            status["unresolved_reviewer_error"] += 1
        elif "ambiguous" in {flash_label, pro_label}:
            status["ambiguous_or_disputed"] += 1
        else:
            status["pass_fail_disagreement"] += 1

        flash_edges = label_map(flash_result)
        pro_edges = label_map(pro_result)
        for edge_id in sorted(set(flash_edges) & set(pro_edges)):
            edge_confusion[f"{flash_edges[edge_id]}->{pro_edges[edge_id]}"] += 1
        if flash_result and flash_result.get("review") and pro_result and pro_result.get("review"):
            flash_final = flash_result["review"]["final_dependency"]["label"]
            pro_final = pro_result["review"]["final_dependency"]["label"]
            final_confusion[f"{flash_final}->{pro_final}"] += 1
            if (
                flash_result["review"].get("missing_edges")
                and pro_result["review"].get("missing_edges")
            ):
                joint_missing_dependency_trajectories += 1

        if flash_label == pro_label == "fail":
            cases.append({
                "trajectory_id": trajectory_id,
                "difficulty": package.get("difficulty"),
                "question": package.get("question"),
                "selection_reasons": selection[trajectory_id],
                "final_dependency": package.get("final_dependency"),
                "flash_final": flash_result["review"].get("final_dependency"),
                "pro_final": pro_result["review"].get("final_dependency"),
                "flash_missing_edges": flash_result["review"].get("missing_edges"),
                "pro_missing_edges": pro_result["review"].get("missing_edges"),
            })

    control_decided = control_pro["pass"] + control_pro["fail"]
    confirmed_invalid_edges = edge_confusion["invalid->invalid"]
    confirmed_invalid_final_dependencies = final_confusion["invalid->invalid"]
    reviewer_errors = status["unresolved_reviewer_error"]
    confirmed_failures = status["confirmed_fail"]
    summary = {
        "total_trajectories": len(packages),
        "single_review_screen_pass": status["flash_pass_unreviewed"],
        "selected_for_dual_review": len(selection),
        "pro_completed": sum(1 for trajectory_id in selection if overall(pro.get(trajectory_id)) != "error"),
        "reviewer_errors": reviewer_errors,
        "review_edges_removed_by_package_projection": {
            "flash": flash_projected_edges,
            "pro": pro_projected_edges,
        },
        "flash_base_protocol_versions": dict(sorted(collections.Counter(
            row.get("review_protocol_version") or "unversioned"
            for row in flash_base.values()
        ).items())),
        "flash_override_protocol_versions": dict(sorted(collections.Counter(
            row.get("review_protocol_version") or "unversioned"
            for row in flash_override.values()
        ).items())),
        "pro_protocol_versions": dict(sorted(collections.Counter(
            row.get("review_protocol_version") or "unversioned"
            for row in pro.values()
        ).items())),
        "effective_flash_components": component_counts(flash, package_ids),
        "effective_pro_components": component_counts(pro, selected_ids),
        "status": dict(sorted(status.items())),
        "flash_to_pro_confusion": dict(sorted(confusion.items())),
        "final_dependency_label_confusion": dict(sorted(final_confusion.items())),
        "edge_label_confusion": dict(sorted(edge_confusion.items())),
        "selection_reason_outcomes": dict(sorted(reason_outcomes.items())),
        "flash_pass_control_pro_labels": dict(sorted(control_pro.items())),
        "flash_pass_control_consensus": dict(sorted(control_consensus.items())),
        "flash_pass_control_decided_failure_rate": (
            control_pro["fail"] / control_decided if control_decided else None
        ),
        "confirmed_invalid_edges": confirmed_invalid_edges,
        "confirmed_invalid_final_dependencies": confirmed_invalid_final_dependencies,
        "joint_reported_missing_dependency_trajectories": (
            joint_missing_dependency_trajectories
        ),
        "confirmed_fail_examples": len(cases),
        "gate": {
            "independent_grounding_edge_precision_passed": (
                confirmed_invalid_edges == 0 and reviewer_errors == 0
            ),
            "independent_dependency_completeness_passed": (
                confirmed_failures == 0 and reviewer_errors == 0
            ),
            "process_reward_ready": (
                confirmed_invalid_edges == 0
                and confirmed_failures == 0
                and reviewer_errors == 0
            ),
            "reason": (
                "External labels are audit evidence, not reward targets. Process credit remains "
                "gated when any jointly confirmed invalid/missing dependency or reviewer error "
                "remains."
            ),
        },
    }
    for path in (args.summary_output, args.cases_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.cases_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
