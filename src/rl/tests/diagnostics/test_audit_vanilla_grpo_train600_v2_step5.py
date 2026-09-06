from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import rl.scenarios.diagnostics.audit_vanilla_grpo_train600_v2_step5 as v2
from rl.tests.diagnostics.test_audit_vanilla_grpo_train600_step5 import (
    _fixture as _v1_fixture,
    _write_json,
    _write_jsonl,
)


SYNTHETIC_MAX_NEW_TOKENS = 4096
SYNTHETIC_MAX_CONTEXT_TOKENS = 16384


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    paths = _v1_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(v2, "EXPECTED_EXPERIMENT_CONFIG_SHA256", "c" * 64)
    for row in paths[4]:
        row["process_update"] = True
    _write_jsonl(paths[0] / "rollouts.jsonl", paths[4])

    manifest_path = paths[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_config"]["experiment_name"] = v2.EXPECTED_EXPERIMENT_NAME
    manifest["experiment_config"]["rollout"]["max_new_tokens"] = (
        SYNTHETIC_MAX_NEW_TOKENS
    )
    manifest["experiment_config"]["rollout"]["max_context_tokens"] = (
        SYNTHETIC_MAX_CONTEXT_TOKENS
    )
    checkpoint_gate = v2._expected_checkpoint_gate(paths[0], paths[2])
    manifest["checkpoint_gate"] = checkpoint_gate
    _write_json(manifest_path, manifest)

    lock_path = paths[0] / "implementation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["checkpoint_gate"] = checkpoint_gate
    _write_json(lock_path, lock)
    return paths


def _audit(
    paths: tuple[Path, Path, Path, Path, list[dict[str, Any]]]
) -> dict[str, Any]:
    return v2.audit(paths[0], paths[1], paths[2], paths[3])


def _rewrite(
    paths: tuple[Path, Path, Path, Path, list[dict[str, Any]]]
) -> None:
    _write_jsonl(paths[0] / "rollouts.jsonl", paths[4])


def _exclude_generation_length(row: dict[str, Any]) -> None:
    row["failure_type"] = "generation_length"
    row["generation_truncation"] = {
        "schema_version": "vllm-generation-truncation-v2",
        "kind": "length",
        "detection": "max_new_tokens_reached",
        "turn_index": 2,
        "completion_tokens": SYNTHETIC_MAX_NEW_TOKENS,
        "max_new_tokens": SYNTHETIC_MAX_NEW_TOKENS,
        "finish_reason": None,
        "finish_reason_field": None,
        "stop_reason": None,
        "stop_reason_field": None,
        "prompt_tokens": 1234,
        "response_token_ids_sha256": "a" * 64,
    }
    row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    row["process_update"] = False
    row["result_reward"]["value"] = 0.0


def _exclude_context_overflow(row: dict[str, Any]) -> None:
    row["failure_type"] = "context_overflow"
    prompt_tokens = SYNTHETIC_MAX_CONTEXT_TOKENS
    row["context_overflow"] = {
        "schema_version": "vllm-context-overflow-v1",
        "kind": "context_overflow",
        "detection": "prompt_plus_max_new_tokens_exceeds_context",
        "turn_index": 3,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": SYNTHETIC_MAX_NEW_TOKENS,
        "max_context_tokens": SYNTHETIC_MAX_CONTEXT_TOKENS,
        "required_tokens": prompt_tokens + SYNTHETIC_MAX_NEW_TOKENS,
    }
    row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    row["process_update"] = False
    row["result_reward"]["value"] = 0.0


def _exclude_timeout(row: dict[str, Any]) -> None:
    row["error_events"] = [
        {
            "error_type": "timeout_error",
            "error_code": "tool_execution_timeout",
            "state_before_hash": "same-state",
            "state_after_hash": "same-state",
            "details": {"state_preserved": True},
        }
    ]
    row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    row["process_update"] = False
    row["result_reward"]["value"] = 0.0


def test_clean_v2_fixture_passes_full_inherited_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _audit(paths)

    assert result["schema_version"] == v2.SCHEMA_VERSION
    assert result["status"] == {
        "outcome": "pass",
        "passes": True,
        "continuation_admitted": True,
        "exit_code": 0,
    }
    assert result["checks"]["five_blocks_are_30_by_k8"]
    assert result["checks"]["binary_terminal_reward_only"]
    assert result["checks"]["at_least_30_mixed_groups"]
    assert result["checks"]["finite_loss_and_gradients"]
    assert result["checks"]["all_252_qlora_layers_synced"]
    assert result["checks"]["checkpoint_fp32_trainables_and_adam"]
    assert result["checks"]["no_generation_oom"]
    assert "eligible_at_least_97_percent" not in result["checks"]
    assert "generation_length_exclusions_at_most_3_percent" not in result["checks"]


def test_44_exact_generation_length_exclusions_pass_within_frozen_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    groups = [
        step * 30 + offset
        for step, count in enumerate((9, 9, 9, 9, 8))
        for offset in range(count)
    ]
    assert len(groups) == 44
    for group in groups:
        _exclude_generation_length(paths[4][group * 8 + 7])
    _rewrite(paths)

    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["eligible"] == 1156
    assert result["observed"]["known_runtime_exclusions"] == 44
    assert result["observed"]["generation_length_exclusions"] == 44
    assert result["checks"][
        "generation_length_exclusions_have_exact_evidence"
    ]


def test_context_and_structured_timeout_are_excluded_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _exclude_context_overflow(paths[4][7])
    _exclude_timeout(paths[4][15])
    _rewrite(paths)

    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["context_overflow_exclusions"] == 1
    assert result["observed"]["structured_timeout_trajectories"] == 1
    assert result["observed"]["structured_timeout_events"] == 1
    assert result["observed"]["known_runtime_exclusions"] == 2


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("missing_process_update", "explicit_process_update_boolean"),
        (
            "false_without_evidence",
            "no_unknown_or_contradictory_optimization_exclusion",
        ),
        (
            "unknown_exclusion",
            "no_unknown_or_contradictory_optimization_exclusion",
        ),
        (
            "malformed_length",
            "generation_length_exclusions_have_exact_evidence",
        ),
        (
            "length_process_true",
            "no_unknown_or_contradictory_optimization_exclusion",
        ),
        (
            "timeout_not_state_preserving",
            "structured_timeouts_are_excluded_and_state_preserving",
        ),
        ("overlapping_bad_timeout", "context_overflow_exclusions_are_explicit"),
        ("bad_context_arithmetic", "context_overflow_exclusions_are_explicit"),
    ],
)
def test_unknown_or_contradictory_exclusions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failed_check: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][7]
    if mutation == "missing_process_update":
        del row["process_update"]
    elif mutation == "false_without_evidence":
        row["process_update"] = False
    elif mutation == "unknown_exclusion":
        row["optimization_exclusion"] = "mystery"
    elif mutation in {"malformed_length", "length_process_true"}:
        _exclude_generation_length(row)
        if mutation == "malformed_length":
            row["generation_truncation"]["completion_tokens"] = 3
        else:
            row["process_update"] = True
    elif mutation == "timeout_not_state_preserving":
        _exclude_timeout(row)
        row["error_events"][0]["details"]["state_preserved"] = False
    elif mutation == "overlapping_bad_timeout":
        _exclude_context_overflow(row)
        row["error_events"] = [
            {
                "error_type": "timeout_error",
                "error_code": "tool_execution_timeout",
                "state_before_hash": "same",
                "state_after_hash": "same",
                "details": {"state_preserved": False},
            }
        ]
    else:
        _exclude_context_overflow(row)
        row["context_overflow"]["required_tokens"] -= 1
    _rewrite(paths)

    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"][failed_check]
    assert not result["checks"][
        "no_unknown_or_contradictory_optimization_exclusion"
    ]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("generation_oom", "no_generation_oom"),
        ("tokenization_warning", "no_tokenization_warning"),
    ],
)
def test_oom_and_tokenization_warning_remain_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failed_check: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][7]
    if mutation == "generation_oom":
        row["failure_type"] = "generation_oom"
        row["process_update"] = False
        row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    else:
        row["rollout_tokenization_warning"] = True
    _rewrite(paths)

    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"][failed_check]


def test_optimizer_and_credit_integrity_gates_are_still_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    state_path = paths[0] / "checkpoint-5/trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_history"][0]["rollout/qlora_layers_synced"] = 251.0
    _write_json(state_path, state)
    paths[4][0]["process_reward"] = {"total_reward": 1.0}
    _rewrite(paths)

    result = _audit(paths)
    assert not result["checks"]["all_252_qlora_layers_synced"]
    assert not result["checks"]["binary_terminal_reward_only"]


def test_cli_receipt_is_immutable_and_fully_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = paths[0] / v2.GATE_RECEIPT_NAME
    args = [
        "--run-dir",
        str(paths[0]),
        "--tasks",
        str(paths[1]),
        "--tasks-manifest",
        str(paths[2]),
        "--initial-adapter",
        str(paths[3]),
        "--output",
        str(output),
    ]

    assert v2.main(args) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == v2.SCHEMA_VERSION
    assert v2.main(args + ["--verify-existing"]) == 0
    assert v2.main(args) == 2


def test_stable_runtime_projection_uses_terminal_failure_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    eligible = paths[4][0]
    assert (
        v2._runtime_exclusion_kind(
            eligible,
            expected_max_tokens=SYNTHETIC_MAX_NEW_TOKENS,
            expected_max_context_tokens=SYNTHETIC_MAX_CONTEXT_TOKENS,
        )
        is None
    )
    length = paths[4][7]
    _exclude_generation_length(length)
    assert (
        v2._runtime_exclusion_kind(
            length,
            expected_max_tokens=SYNTHETIC_MAX_NEW_TOKENS,
            expected_max_context_tokens=SYNTHETIC_MAX_CONTEXT_TOKENS,
        )
        == "generation_length"
    )
    context = paths[4][15]
    _exclude_context_overflow(context)
    _exclude_timeout(context)
    assert (
        v2._runtime_exclusion_kind(
            context,
            expected_max_tokens=SYNTHETIC_MAX_NEW_TOKENS,
            expected_max_context_tokens=SYNTHETIC_MAX_CONTEXT_TOKENS,
        )
        == "context_overflow"
    )


def test_mirrored_timeout_event_is_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][7]
    _exclude_timeout(row)
    row["turns"] = [{"error_event": dict(row["error_events"][0])}]
    _rewrite(paths)

    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["structured_timeout_trajectories"] == 1
    assert result["observed"]["structured_timeout_events"] == 1


def test_frozen_overall_and_generation_length_caps_are_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    groups = [
        step * 30 + offset
        for step in range(5)
        for offset in range(12)
    ]
    assert len(groups) == 60
    for group in groups:
        _exclude_generation_length(paths[4][group * 8 + 7])
    _rewrite(paths)
    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["eligible_fraction"] == 0.95
    assert result["observed"]["generation_length_exclusion_fraction"] == 0.05

    _exclude_generation_length(paths[4][12 * 8 + 7])
    _rewrite(paths)
    result = _audit(paths)
    assert not result["checks"]["overall_eligible_at_least_95_percent"]
    assert not result["checks"][
        "generation_length_exclusions_at_most_5_percent"
    ]


def test_per_step_timeout_and_context_caps_fail_systemic_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    # 25 exclusions in step 0 leave 215/240 eligible (89.58%) while every
    # global category cap still passes.
    for group in range(25):
        _exclude_generation_length(paths[4][group * 8 + 7])
    _rewrite(paths)
    result = _audit(paths)
    assert not result["checks"]["each_step_eligible_at_least_90_percent"]
    assert result["checks"]["overall_eligible_at_least_95_percent"]

    paths = _fixture(tmp_path / "timeout", monkeypatch)
    for group in range(25):
        _exclude_timeout(paths[4][group * 8 + 7])
    _rewrite(paths)
    result = _audit(paths)
    assert not result["checks"]["timeout_trajectories_at_most_2_percent"]

    paths = _fixture(tmp_path / "context", monkeypatch)
    for group in range(4):
        _exclude_context_overflow(paths[4][group * 8 + 7])
    _rewrite(paths)
    result = _audit(paths)
    assert not result["checks"]["context_overflow_at_most_0_25_percent"]


def test_group_with_fewer_than_two_eligible_is_report_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    for sample in range(7):
        _exclude_generation_length(paths[4][sample])
    _rewrite(paths)

    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["groups_with_fewer_than_two_eligible"] == 1


def test_excluded_nonzero_reward_fails_binary_terminal_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][7]
    _exclude_generation_length(row)
    row["result_reward"]["value"] = 1.0
    _rewrite(paths)

    result = _audit(paths)
    assert not result["checks"]["binary_terminal_reward_only"]
