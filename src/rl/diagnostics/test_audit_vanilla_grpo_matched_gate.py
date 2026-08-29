from __future__ import annotations

import copy

import pytest

from src.rl.diagnostics.analyze_evaluation_results import exact_mcnemar_p
from src.rl.diagnostics.audit_vanilla_grpo_matched_gate import (
    DEFAULT_REQUIRED_MATCHED_IDENTITY_FIELDS,
    audit,
)


def _analysis() -> dict:
    gains = 12
    regressions = 2
    legal_gains = 3
    legal_regressions = 1
    matched = {
        field: f"matched-{field}"
        for field in DEFAULT_REQUIRED_MATCHED_IDENTITY_FIELDS
    }
    return {
        "cohort": {"total": 100},
        "identity_contract": {
            "required": True,
            "all_present": True,
            "matched_fields": matched,
            "adapter_sha256": {
                "sft1": "sft-sha",
                "checkpoint2": "cp2-sha",
                "final": "final-sha",
            },
            "distinct_adapters_required": True,
        },
        "arms": {
            "sft1": {"correct": 50, "legal": 90, "total": 100},
            "checkpoint2": {"correct": 49, "legal": 89, "total": 100},
            "final": {"correct": 60, "legal": 92, "total": 100},
        },
        "comparisons": {
            "checkpoint2_vs_sft1": {
                "accuracy": {
                    "gains": 6,
                    "regressions": 7,
                    "exact_mcnemar_p": exact_mcnemar_p(6, 7),
                },
                "legal": {"gains": 2, "regressions": 3},
            },
            "final_vs_sft1": {
                "accuracy": {
                    "gains": gains,
                    "regressions": regressions,
                    "exact_mcnemar_p": exact_mcnemar_p(gains, regressions),
                },
                "legal": {
                    "gains": legal_gains,
                    "regressions": legal_regressions,
                },
            },
        },
    }


def _audit(analysis: dict) -> dict:
    return audit(
        analysis,
        baseline="sft1",
        primary_candidate="final",
        candidates=["checkpoint2", "final"],
        expected_count=100,
    )


def test_primary_checkpoint_passes_strict_gate() -> None:
    result = _audit(_analysis())
    assert result["status"]["matched_evaluation_valid"]
    assert result["candidates"]["final"]["candidate_passes"]
    assert result["status"]["promote"]
    assert not result["candidates"]["checkpoint2"]["candidate_passes"]


def test_secondary_checkpoint_cannot_trigger_posthoc_promotion() -> None:
    analysis = _analysis()
    analysis["arms"]["checkpoint2"].update(correct=60, legal=92)
    analysis["comparisons"]["checkpoint2_vs_sft1"] = copy.deepcopy(
        analysis["comparisons"]["final_vs_sft1"]
    )
    analysis["arms"]["final"].update(correct=49, legal=89)
    analysis["comparisons"]["final_vs_sft1"] = {
        "accuracy": {
            "gains": 6,
            "regressions": 7,
            "exact_mcnemar_p": exact_mcnemar_p(6, 7),
        },
        "legal": {"gains": 2, "regressions": 3},
    }
    result = _audit(analysis)
    assert result["candidates"]["checkpoint2"]["candidate_passes"]
    assert not result["candidates"]["final"]["candidate_passes"]
    assert not result["status"]["promote"]


def test_missing_runtime_digest_invalidates_matched_gate() -> None:
    analysis = _analysis()
    del analysis["identity_contract"]["matched_fields"]["runtime_sha256"]
    result = _audit(analysis)
    assert not result["identity_guard"]["checks"]["required_fields_matched"]
    assert result["identity_guard"]["missing_matched_fields"] == [
        "runtime_sha256"
    ]
    assert not result["status"]["matched_evaluation_valid"]
    assert not result["status"]["promote"]


def test_legal_regression_blocks_otherwise_positive_primary() -> None:
    analysis = _analysis()
    analysis["arms"]["final"]["legal"] = 89
    analysis["comparisons"]["final_vs_sft1"]["legal"] = {
        "gains": 1,
        "regressions": 2,
    }
    result = _audit(analysis)
    assert result["candidates"]["final"]["checks"][
        "exact_mcnemar_significant"
    ]
    assert not result["candidates"]["final"]["checks"]["legal_nonregression"]
    assert not result["status"]["promote"]


def test_inconsistent_paired_counts_are_rejected() -> None:
    analysis = _analysis()
    analysis["arms"]["final"]["correct"] = 59
    with pytest.raises(ValueError, match="paired accuracy counts are inconsistent"):
        _audit(analysis)
