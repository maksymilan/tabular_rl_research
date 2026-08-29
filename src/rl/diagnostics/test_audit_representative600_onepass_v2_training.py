from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.rl.diagnostics import audit_representative600_onepass_v2_training as audit


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "src/rl/configs/experiments/"
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2.yaml"
)


def _eligible() -> dict:
    return {
        "trajectory_id": "eligible",
        "process_update": True,
        "failure_type": None,
        "optimization_exclusion": None,
        "rollout_tokenization_warning": False,
        "turns": [],
        "error_events": [],
    }


def _context_overflow() -> dict:
    row = _eligible()
    row.update(
        {
            "trajectory_id": "context",
            "process_update": False,
            "failure_type": "context_overflow",
            "optimization_exclusion": "nonsemantic_runtime_failure",
            "context_overflow": {
                "schema_version": "vllm-context-overflow-v1",
                "kind": "context_overflow",
                "detection": "prompt_plus_max_new_tokens_exceeds_context",
                "turn_index": 1,
                "prompt_tokens": 18000,
                "max_new_tokens": 3072,
                "max_context_tokens": 20480,
                "required_tokens": 21072,
            },
        }
    )
    return row


def _generation_length() -> dict:
    row = _eligible()
    row.update(
        {
            "trajectory_id": "length",
            "process_update": False,
            "failure_type": "generation_length",
            "optimization_exclusion": "nonsemantic_runtime_failure",
            "generation_truncation": {
                "schema_version": "vllm-generation-truncation-v2",
                "kind": "length",
                "detection": "max_new_tokens_reached",
                "turn_index": 0,
                "prompt_tokens": 512,
                "completion_tokens": 3072,
                "max_new_tokens": 3072,
                "response_token_ids_sha256": "a" * 64,
                "finish_reason": None,
                "finish_reason_field": None,
                "stop_reason": None,
                "stop_reason_field": None,
            },
        }
    )
    return row


def _timeout() -> dict:
    row = _eligible()
    row.update(
        {
            "trajectory_id": "timeout",
            "process_update": False,
            "optimization_exclusion": "nonsemantic_runtime_failure",
            "error_events": [
                {
                    "error_type": "timeout_error",
                    "error_code": "tool_execution_timeout",
                    "state_before_hash": "same-state",
                    "state_after_hash": "same-state",
                    "details": {"state_preserved": True},
                }
            ],
        }
    )
    return row


def _full_rows() -> list[dict]:
    rows = []
    for index in range(4800):
        row = _eligible()
        row["trajectory_id"] = f"row-{index}"
        rows.append(row)
    return rows


def _distributed_indices(count: int) -> list[int]:
    return [
        step * 240 + offset
        for offset in range(240)
        for step in range(20)
    ][:count]


def _replace_runtime(rows: list[dict], indices: list[int], template: dict) -> None:
    for index in indices:
        trajectory_id = rows[index]["trajectory_id"]
        rows[index] = copy.deepcopy(template)
        rows[index]["trajectory_id"] = trajectory_id


def test_v2_completion_config_is_exactly_3072_over_20480() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    audit._validate_experiment_config(payload)
    assert audit._engine.common.sha256_file(CONFIG) == audit.EXPECTED_CONFIG_SHA256

    bad = copy.deepcopy(payload)
    bad["rollout"]["max_context_tokens"] = 16384
    with pytest.raises(ValueError, match="max_context_tokens"):
        audit._validate_experiment_config(bad)


def test_full_run_classifier_accepts_only_known_mutually_exclusive_exclusions() -> None:
    summary = audit._runtime_exclusion_summary(
        [_eligible(), _generation_length(), _context_overflow()]
    )
    assert summary["counts"] == {
        "eligible": 1,
        "generation_length": 1,
        "structured_timeout": 0,
        "context_overflow": 1,
    }
    assert summary["unknown_or_contradictory"] == 0


def test_full_run_thresholds_accept_exact_clean_4800() -> None:
    summary = audit._runtime_exclusion_summary(_full_rows())
    assert all(summary["checks"].values())
    assert summary["eligible_by_step"] == [240] * 20
    assert summary["eligible_fraction_by_step"] == [1.0] * 20


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("overall", "overall_eligible_at_least_95_percent"),
        ("step", "each_step_eligible_at_least_90_percent"),
        ("length", "generation_length_exclusions_at_most_5_percent"),
        ("timeout", "timeout_trajectories_at_most_2_percent"),
        ("context", "context_overflow_at_most_0_25_percent"),
        ("count", "exact_4800_runtime_classifications"),
    ],
)
def test_full_run_five_thresholds_fail_closed(
    case: str, failed_check: str
) -> None:
    rows = _full_rows()
    if case == "overall":
        indices = _distributed_indices(241)
        _replace_runtime(rows, indices[:200], _generation_length())
        _replace_runtime(rows, indices[200:230], _timeout())
        _replace_runtime(rows, indices[230:], _context_overflow())
    elif case == "step":
        _replace_runtime(rows, list(range(25)), _generation_length())
    elif case == "length":
        _replace_runtime(rows, _distributed_indices(241), _generation_length())
    elif case == "timeout":
        _replace_runtime(rows, _distributed_indices(97), _timeout())
    elif case == "context":
        _replace_runtime(rows, _distributed_indices(13), _context_overflow())
    else:
        rows.pop()
    summary = audit._runtime_exclusion_summary(rows)
    assert summary["checks"][failed_check] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown", "lacks a known runtime cause"),
        ("unexcluded", "known runtime failure"),
        ("contradictory", "conflict"),
        ("oom", "generation OOM"),
        ("warning", "tokenization warning"),
    ],
)
def test_full_run_classifier_fails_closed(
    mutation: str, message: str
) -> None:
    row = _context_overflow()
    if mutation == "unknown":
        row = _eligible()
        row["optimization_exclusion"] = "mystery"
    elif mutation == "unexcluded":
        row["optimization_exclusion"] = None
    elif mutation == "contradictory":
        row.update(_generation_length())
        row["failure_type"] = "context_overflow"
    elif mutation == "oom":
        row["failure_type"] = "generation_oom"
    else:
        row["rollout_tokenization_warning"] = True
    with pytest.raises(ValueError, match=message):
        audit._runtime_exclusion_summary([row])


def test_completion_auditor_declares_final_as_only_evaluation_candidate() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"evaluation_candidates": ["final"]' in source
    assert '"posthoc_checkpoint_selection_allowed": False' in source
    assert "checkpoint inventory is not exact steps 1..20" in source
