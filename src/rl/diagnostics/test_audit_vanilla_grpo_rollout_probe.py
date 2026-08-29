from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.rl.diagnostics.audit_vanilla_grpo_rollout_probe import (
    EXPECTED_GROUP_SIZE,
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_STUDENT_PROMPT_SHA256,
    EXPECTED_TASKS,
    TRAJECTORY_SCHEMA,
    audit,
    main,
)


def _rows(*, mixed_groups: int = 20) -> list[dict]:
    rows = []
    for task_index in range(EXPECTED_TASKS):
        for sample_index in range(EXPECTED_GROUP_SIZE):
            correct = (
                sample_index < EXPECTED_GROUP_SIZE // 2
                if task_index < mixed_groups
                else True
            )
            sequence = task_index * EXPECTED_GROUP_SIZE + sample_index
            rows.append(
                {
                    "schema_version": TRAJECTORY_SCHEMA,
                    "sequence": sequence,
                    "environment": {
                        "task_id": f"bird_train_{task_index:05d}",
                        "example_index": task_index,
                    },
                    "sample": {
                        "reward": float(correct),
                        "correct": correct,
                        "failure_type": None,
                        "audit_record": {
                            "example_index": task_index,
                            "sample_index": sample_index,
                            "protocol_version": EXPECTED_PROTOCOL_VERSION,
                            "protocol_hash": EXPECTED_PROTOCOL_HASH,
                            "failure_type": None,
                            "turns": [],
                            "error_events": [],
                            "result_reward": {
                                "profile": "binary",
                                "correct": correct,
                                "executable_terminal": correct,
                                "value": float(correct),
                            },
                        },
                        "step_rewards": None,
                        "process_update": True,
                    },
                    "policy_turns": [
                        {
                            "prompt_ids": [1, 2],
                            "response_ids": [3, 4],
                            "sampling_logprobs": [-0.1, -0.2],
                        }
                    ],
                }
            )
    return rows


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()


def _manifest(rows: list[dict], digest: str | None = None) -> dict:
    return {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": (
            EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256
        ),
        "tasks": EXPECTED_TASKS,
        "group_size": EXPECTED_GROUP_SIZE,
        "trajectories": len(rows),
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "trajectories_sha256": digest
        or hashlib.sha256(_jsonl_bytes(rows)).hexdigest(),
        "correct_trajectories": sum(row["sample"]["correct"] for row in rows),
    }


def _audit(rows: list[dict], manifest: dict | None = None) -> dict:
    digest = hashlib.sha256(_jsonl_bytes(rows)).hexdigest()
    return audit(
        manifest or _manifest(rows, digest),
        rows,
        trajectories_sha256=digest,
    )


def test_valid_no_update_probe_passes_all_frozen_gates() -> None:
    result = _audit(_rows())
    assert result["status"] == {"passes": True, "probe_admitted": True}
    assert all(result["checks"].values())
    assert result["observed"]["eligible_trajectories"] == 256
    assert result["observed"]["mixed_outcome_groups"] == 20
    assert result["contract"]["optimizer_updates"] == 0


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("generation_length", "no_generation_length"),
        ("context_overflow", "no_context_overflow"),
        ("timeout", "no_timeout"),
        ("optimization_exclusion", "no_optimization_exclusion"),
    ],
)
def test_forbidden_runtime_failures_fail_closed(
    mutation: str, failed_check: str
) -> None:
    rows = _rows()
    sample = rows[0]["sample"]
    audit_record = sample["audit_record"]
    if mutation in {"generation_length", "context_overflow"}:
        sample["failure_type"] = mutation
        audit_record["failure_type"] = mutation
    elif mutation == "timeout":
        audit_record["turns"] = [
            {"execution_error_type": "timeout_error"}
        ]
    else:
        audit_record["optimization_exclusion"] = "max_token_completion"
    result = _audit(rows)
    assert not result["checks"][failed_check]
    assert not result["status"]["passes"]


def test_below_95_percent_eligibility_fails_closed() -> None:
    rows = _rows()
    for row in rows[:13]:
        row["sample"]["process_update"] = False
    result = _audit(rows)
    assert result["observed"]["eligible_trajectories"] == 243
    assert not result["checks"]["eligible_at_least_95_percent"]
    assert not result["status"]["passes"]


def test_fewer_than_20_mixed_groups_fails_closed() -> None:
    result = _audit(_rows(mixed_groups=19))
    assert result["observed"]["mixed_outcome_groups"] == 19
    assert not result["checks"]["at_least_20_mixed_outcome_groups"]


def test_nonbinary_reward_or_wrong_protocol_identity_fails_closed() -> None:
    rows = _rows()
    rows[0]["sample"]["reward"] = 0.2
    rows[1]["sample"]["audit_record"]["protocol_hash"] = "wrong"
    result = _audit(rows)
    assert not result["checks"]["binary_result_reward_only"]
    assert not result["checks"]["trajectory_protocol_identity"]
    assert not result["status"]["passes"]


def test_nonzero_optimizer_step_metadata_fails_no_update_contract() -> None:
    rows = _rows()
    rows[0]["sample"]["audit_record"]["policy_global_step"] = 1
    result = _audit(rows)
    assert not result["checks"]["no_optimizer_update_evidence"]
    assert not result["status"]["passes"]


def test_group_must_have_exact_sample_indices_zero_through_seven() -> None:
    rows = _rows()
    rows[0]["sample"]["audit_record"]["sample_index"] = 7
    result = _audit(rows)
    assert not result["checks"]["exact_k8_groups"]
    assert not result["status"]["passes"]


def test_environment_and_audit_identity_must_match() -> None:
    rows = _rows()
    rows[0]["environment"]["example_index"] = 999
    result = _audit(rows)
    assert not result["checks"]["trajectory_rows_valid"]
    assert not result["status"]["passes"]


def _write_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_bytes(_jsonl_bytes(rows))
    manifest = tmp_path / "manifest.pending.json"
    manifest.write_text(
        json.dumps(
            _manifest(rows, hashlib.sha256(trajectories.read_bytes()).hexdigest())
        ),
        encoding="utf-8",
    )
    return manifest, trajectories


def test_cli_writes_success_json_only_after_all_checks_pass(tmp_path: Path) -> None:
    manifest, trajectories = _write_inputs(tmp_path, _rows())
    output = tmp_path / "audit.json"
    return_code = main(
        [
            "--manifest",
            str(manifest),
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
        ]
    )
    assert return_code == 0
    assert json.loads(output.read_text())["status"]["passes"] is True


def test_cli_failure_is_nonzero_and_does_not_write_output(tmp_path: Path) -> None:
    rows = _rows(mixed_groups=19)
    manifest, trajectories = _write_inputs(tmp_path, rows)
    output = tmp_path / "audit.json"
    return_code = main(
        [
            "--manifest",
            str(manifest),
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
        ]
    )
    assert return_code == 1
    assert not output.exists()


def test_failed_overwrite_revokes_stale_success_output(tmp_path: Path) -> None:
    rows = _rows(mixed_groups=19)
    manifest, trajectories = _write_inputs(tmp_path, rows)
    output = tmp_path / "audit.json"
    output.write_text('{"status":{"passes":true}}', encoding="utf-8")
    return_code = main(
        [
            "--manifest",
            str(manifest),
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
            "--overwrite",
        ]
    )
    assert return_code == 1
    assert not output.exists()


def test_manifest_and_file_digest_are_both_frozen() -> None:
    rows = _rows()
    manifest = _manifest(rows)
    manifest["protocol_hash"] = "wrong"
    manifest["trajectories_sha256"] = "0" * 64
    result = _audit(rows, manifest)
    assert not result["checks"]["manifest_contract"]
    assert not result["checks"]["trajectory_digest_bound"]
    assert not result["status"]["passes"]
