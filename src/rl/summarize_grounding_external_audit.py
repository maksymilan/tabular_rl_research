#!/usr/bin/env python3
"""Summarize independent Flash/Pro reviews of harness grounding evidence."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--flash-reviews", type=Path, required=True)
    parser.add_argument("--pro-reviews", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    args = parser.parse_args()

    packages = {row["trajectory_id"]: row for row in read_jsonl(args.packages)}
    flash = {row["trajectory_id"]: row for row in read_jsonl(args.flash_reviews)}
    pro = {row["trajectory_id"]: row for row in read_jsonl(args.pro_reviews)}
    selection = {row["trajectory_id"]: row["reasons"] for row in read_jsonl(args.selection)}

    confusion = collections.Counter()
    edge_confusion = collections.Counter()
    final_confusion = collections.Counter()
    status = collections.Counter()
    control_pro = collections.Counter()
    cases = []
    for trajectory_id, package in packages.items():
        flash_result = flash.get(trajectory_id)
        pro_result = pro.get(trajectory_id)
        flash_label = overall(flash_result)
        if trajectory_id not in selection:
            status["flash_pass_unreviewed"] += 1
            continue
        pro_label = overall(pro_result)
        confusion[f"{flash_label}->{pro_label}"] += 1
        if "flash_pass_control" in selection[trajectory_id]:
            control_pro[pro_label] += 1

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
    summary = {
        "total_trajectories": len(packages),
        "selected_for_pro": len(selection),
        "pro_completed": sum(1 for trajectory_id in selection if overall(pro.get(trajectory_id)) != "error"),
        "status": dict(sorted(status.items())),
        "flash_to_pro_confusion": dict(sorted(confusion.items())),
        "final_dependency_label_confusion": dict(sorted(final_confusion.items())),
        "edge_label_confusion": dict(sorted(edge_confusion.items())),
        "flash_pass_control_pro_labels": dict(sorted(control_pro.items())),
        "flash_pass_control_decided_failure_rate": (
            control_pro["fail"] / control_decided if control_decided else None
        ),
        "confirmed_fail_examples": len(cases),
        "gate": {
            "process_reward_ready": False,
            "reason": (
                "Confirmed missing/invalid dependencies and non-zero reviewer disagreement remain; "
                "external labels are audit evidence, not reward targets."
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
