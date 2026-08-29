#!/usr/bin/env python3
"""Fail-closed promotion gate for a matched vanilla-GRPO evaluation.

The unified evaluation analyzer owns metrics and paired statistics.  This
module adds a preregistered decision boundary: one primary checkpoint may be
promoted, while intermediate checkpoints remain diagnostic and cannot be
selected post hoc because they happened to score best.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.rl.diagnostics.analyze_evaluation_results import exact_mcnemar_p


SCHEMA_VERSION = "vanilla-grpo-matched-gate-v1"
DEFAULT_REQUIRED_MATCHED_IDENTITY_FIELDS = (
    "base_model",
    "base_model_revision",
    "base_model_identity_sha256",
    "adapter_config_semantic_sha256",
    "dataset",
    "input_sha256",
    "task_identity_sha256_without_db_path",
    "protocol_version",
    "protocol_hash",
    "runtime",
    "runtime_sha256",
    "serving",
    "concurrency",
    "decode",
    "agent",
    "tool_execution_timeout_seconds",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"analysis must be a JSON object: {path}")
    return value


def _candidate_gate(
    analysis: dict[str, Any],
    *,
    candidate: str,
    baseline: str,
    expected_count: int,
    min_accuracy_gain_pp: float,
    alpha: float,
) -> dict[str, Any]:
    arms = analysis.get("arms") or {}
    if candidate not in arms or baseline not in arms:
        raise ValueError(f"unknown candidate/baseline: {candidate}/{baseline}")
    comparison_name = f"{candidate}_vs_{baseline}"
    comparison = (analysis.get("comparisons") or {}).get(comparison_name)
    if not comparison:
        raise ValueError(f"missing paired comparison: {comparison_name}")

    candidate_arm = arms[candidate]
    baseline_arm = arms[baseline]
    accuracy = comparison.get("accuracy") or {}
    legal = comparison.get("legal") or {}
    gains = int(accuracy.get("gains") or 0)
    regressions = int(accuracy.get("regressions") or 0)
    net = gains - regressions
    candidate_correct = int(candidate_arm.get("correct") or 0)
    baseline_correct = int(baseline_arm.get("correct") or 0)
    if candidate_correct - baseline_correct != net:
        raise ValueError(
            f"paired accuracy counts are inconsistent for {comparison_name}"
        )
    observed_p = float(accuracy.get("exact_mcnemar_p"))
    recomputed_p = exact_mcnemar_p(gains, regressions)
    if not math.isclose(observed_p, recomputed_p, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"McNemar p-value is inconsistent for {comparison_name}: "
            f"observed={observed_p} recomputed={recomputed_p}"
        )

    legal_gains = int(legal.get("gains") or 0)
    legal_regressions = int(legal.get("regressions") or 0)
    legal_net = legal_gains - legal_regressions
    candidate_legal = int(candidate_arm.get("legal") or 0)
    baseline_legal = int(baseline_arm.get("legal") or 0)
    if candidate_legal - baseline_legal != legal_net:
        raise ValueError(
            f"paired legal counts are inconsistent for {comparison_name}"
        )

    complete = (
        int(candidate_arm.get("total") or 0) == expected_count
        and int(baseline_arm.get("total") or 0) == expected_count
    )
    net_percentage_points = 100.0 * net / expected_count
    checks = {
        "complete_exact_cohort": complete,
        "minimum_accuracy_gain": net_percentage_points >= min_accuracy_gain_pp,
        "exact_mcnemar_significant": recomputed_p < alpha,
        "legal_nonregression": legal_net >= 0,
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "correct": candidate_correct,
        "baseline_correct": baseline_correct,
        "accuracy_gains": gains,
        "accuracy_regressions": regressions,
        "accuracy_net": net,
        "accuracy_net_percentage_points": net_percentage_points,
        "accuracy_exact_mcnemar_p": recomputed_p,
        "legal": candidate_legal,
        "baseline_legal": baseline_legal,
        "legal_gains": legal_gains,
        "legal_regressions": legal_regressions,
        "legal_net": legal_net,
        "checks": checks,
        "candidate_passes": all(checks.values()),
    }


def audit(
    analysis: dict[str, Any],
    *,
    baseline: str,
    primary_candidate: str,
    candidates: Sequence[str],
    expected_count: int = 1534,
    min_accuracy_gain_pp: float = 1.0,
    alpha: float = 0.05,
    required_matched_identity_fields: Sequence[
        str
    ] = DEFAULT_REQUIRED_MATCHED_IDENTITY_FIELDS,
) -> dict[str, Any]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if min_accuracy_gain_pp <= 0.0:
        raise ValueError("min_accuracy_gain_pp must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    candidate_names = list(dict.fromkeys(candidates))
    if primary_candidate not in candidate_names:
        raise ValueError("primary_candidate must be included in candidates")
    if baseline in candidate_names:
        raise ValueError("baseline cannot also be a candidate")

    identity = analysis.get("identity_contract") or {}
    matched = identity.get("matched_fields") or {}
    adapter_hashes = identity.get("adapter_sha256") or {}
    required_fields = set(required_matched_identity_fields)
    missing_fields = sorted(required_fields - set(matched))
    evaluated_names = [baseline, *candidate_names]
    bound_hashes = [adapter_hashes.get(name) for name in evaluated_names]
    identity_checks = {
        "identities_required": identity.get("required") is True,
        "identities_all_present": identity.get("all_present") is True,
        "required_fields_matched": not missing_fields,
        "adapter_hashes_present": all(bound_hashes),
        "adapter_hashes_distinct": (
            bool(bound_hashes) and len(bound_hashes) == len(set(bound_hashes))
        ),
        "analyzer_distinct_adapter_guard_enabled": (
            identity.get("distinct_adapters_required") is True
        ),
    }
    cohort = analysis.get("cohort") or {}
    cohort_check = int(cohort.get("total") or 0) == expected_count
    candidate_results = {
        candidate: _candidate_gate(
            analysis,
            candidate=candidate,
            baseline=baseline,
            expected_count=expected_count,
            min_accuracy_gain_pp=min_accuracy_gain_pp,
            alpha=alpha,
        )
        for candidate in candidate_names
    }
    matched_evaluation_valid = cohort_check and all(identity_checks.values())
    primary_passes = (
        matched_evaluation_valid
        and candidate_results[primary_candidate]["candidate_passes"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_contract": {
            "baseline": baseline,
            "primary_candidate": primary_candidate,
            "secondary_candidates": [
                name for name in candidate_names if name != primary_candidate
            ],
            "expected_count": expected_count,
            "minimum_accuracy_gain_percentage_points": min_accuracy_gain_pp,
            "exact_mcnemar_alpha": alpha,
            "legal_nonregression_required": True,
            "checkpoint_selection": (
                "primary-only; secondary checkpoints are diagnostic and cannot "
                "trigger promotion post hoc"
            ),
        },
        "identity_guard": {
            "required_matched_fields": list(required_matched_identity_fields),
            "missing_matched_fields": missing_fields,
            "checks": identity_checks,
        },
        "cohort": {
            "observed_count": cohort.get("total"),
            "expected_count": expected_count,
            "passes": cohort_check,
        },
        "candidates": candidate_results,
        "status": {
            "matched_evaluation_valid": matched_evaluation_valid,
            "primary_candidate_passes": primary_passes,
            "promote": primary_passes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a strict, identity-matched vanilla-GRPO promotion gate."
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--expected-count", type=int, default=1534)
    parser.add_argument("--min-accuracy-gain-pp", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output}")
    result = audit(
        load_json(args.analysis),
        baseline=args.baseline,
        primary_candidate=args.primary_candidate,
        candidates=args.candidate,
        expected_count=args.expected_count,
        min_accuracy_gain_pp=args.min_accuracy_gain_pp,
        alpha=args.alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **result["status"]}))


if __name__ == "__main__":
    main()
