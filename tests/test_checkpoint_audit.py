from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import tool_modules.checkpoint_relalg.audit as audit_module

from tool_modules.checkpoint_relalg.audit import (
    BATCH_CONTROL_VERSION,
    PREFIX_BATCH_RUNNER_VERSION,
    _canonical_sha256,
    _dataset_source_identity,
    _manifest_config_sha256,
    _selection_identity,
    audit_record,
    audit_result_dir,
    fresh_replay_record,
    no_leak_issues,
    provider_history_issues,
)
from tool_modules.checkpoint_relalg.protocol import (
    BACKEND,
    CHECKPOINT_POLICY_VERSION,
    DIALECT,
    ENVIRONMENT_RENDERER_VERSION,
    EXECUTOR_VERSION,
    PROTOCOL_VERSION,
    SCHEME,
    capability_manifest,
    get_system_prompt,
    prompt_hash,
    tool_schema_hash,
)
from tool_modules.checkpoint_relalg.runtime import CheckpointRelalgRuntime
from tool_modules.checkpoint_relalg.provider import tool_result_message


def _model_input(mode: str):
    return [
        {"role": "system", "content": get_system_prompt(mode, teacher=True)},
        {"role": "user", "content": "current facts"},
    ]


def _request_options():
    return {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "fake",
        "tool_choice": "auto",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }


def _runtime_identity(mode: str):
    return {
        "capability_manifest": capability_manifest(mode),
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "executor_version": EXECUTOR_VERSION,
    }


def _messages():
    return [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "current facts"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "inspect",
            "tool_calls": [
                {
                    "id": "call_1",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "describe_table",
                        "arguments": '{"tables":["items"]}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"success"}'},
        {"role": "user", "content": "updated facts"},
    ]


def test_no_leak_gate_recurses_through_provider_messages():
    assert not no_leak_issues(_messages())
    leaking = _messages()
    leaking[-1]["content"] = {"gold_sql": "SELECT secret"}
    assert "gold_sql" in no_leak_issues(leaking)[0]


def test_provider_history_requires_ordered_tool_results():
    assert not provider_history_issues(_messages(), root="messages")
    broken = _messages()
    broken[3]["tool_call_id"] = "other"
    assert provider_history_issues(broken, root="messages")


def test_record_audit_checks_hashes_and_state_preservation():
    mode = "direct"
    assistant = _messages()[2]
    assistant["tool_calls"][0]["function"]["arguments"] = '{"tables":["missing"]}'
    result = {
        "step_id": "step_001",
        "status": "error",
        "error": {"code": "unknown_table"},
    }
    record = {
        **_runtime_identity(mode),
        "tool_scheme": SCHEME,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "tool_schema_sha256": tool_schema_hash(mode),
        "teacher_prompt_sha256": prompt_hash(mode, teacher=True),
        "admission_status": "diagnostic-only",
        "provider_request_options": _request_options(),
        "turns": [
            {
                "model_input": _model_input(mode),
                "assistant_message": assistant,
                "provider_response_metadata": {"model": "fake"},
                "action": {
                    "tool": "describe_table",
                    "arguments": {"tables": ["missing"]},
                    "tool_call_id": "call_1",
                },
                "tool_result_messages": [tool_result_message("call_1", result)],
                "result": result,
                "environment_state_hash_before": "same",
                "environment_state_hash_after": "same",
            }
        ],
    }
    assert audit_record(record)["passed"]
    record["turns"][0]["environment_state_hash_after"] = "changed"
    assert not audit_record(record)["passed"]


def test_record_audit_requires_multi_call_rejection_and_ordered_results():
    mode = "direct"
    assistant = _messages()[2]
    assistant["tool_calls"].append(
        {
            "id": "call_2",
            "index": 1,
            "type": "function",
            "function": {"name": "read_rows", "arguments": '{"table":"items"}'},
        }
    )
    result = {
        "step_id": "step_001",
        "status": "error",
        "error": {
            "type": "protocol_error",
            "code": "invalid_tool_call_count",
            "message": "each assistant turn requires exactly one native tool call; received 2",
            "details": {"argument_path": "$.tool_calls", "call_count": 2},
        },
        "attempted_action": {
            "tool": ["describe_table", "read_rows"],
            "arguments": ['{"tables":["items"]}', '{"table":"items"}'],
        },
    }
    record = {
        **_runtime_identity(mode),
        "tool_scheme": SCHEME,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "tool_schema_sha256": tool_schema_hash(mode),
        "teacher_prompt_sha256": prompt_hash(mode, teacher=True),
        "admission_status": "diagnostic-only",
        "provider_request_options": _request_options(),
        "turns": [
            {
                "model_input": _model_input(mode),
                "assistant_message": assistant,
                "provider_response_metadata": {"model": "fake"},
                "tool_result_messages": [
                    tool_result_message("call_1", result),
                    tool_result_message("call_2", result),
                ],
                "native_rejection": {
                    "code": "invalid_tool_call_count",
                    "message": "each assistant turn requires exactly one native tool call; received 2",
                    "argument_path": "$.tool_calls",
                    "call_count": 2,
                },
                "result": result,
                "environment_state_hash_before": "same",
                "environment_state_hash_after": "same",
            }
        ],
    }
    assert audit_record(record)["passed"]
    record["turns"][0]["result"]["error"]["code"] = "different"
    assert not audit_record(record)["passed"]


def test_runtime_identity_is_bound_by_structure_and_fresh_replay(tmp_path):
    mode = "direct"
    db_path = tmp_path / "identity.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY)")
    runtime = CheckpointRelalgRuntime(connection, mode=mode)
    final_runtime = runtime.audit_state()
    connection.close()
    task = {
        "db_path": str(db_path),
        "db_id": "identity",
        "question": "Return the item identifiers.",
        "external_knowledge": None,
    }
    record = {
        **_runtime_identity(mode),
        "tool_scheme": SCHEME,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "tool_schema_sha256": tool_schema_hash(mode),
        "teacher_prompt_sha256": prompt_hash(mode, teacher=True),
        "admission_status": "diagnostic-only",
        "provider_request_options": _request_options(),
        "db_id": task["db_id"],
        "question": task["question"],
        "external_knowledge": None,
        "runtime_config": {},
        "turns": [],
        "final_runtime": final_runtime,
        "legal": False,
    }
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]

    for field, forged in {
        "backend": "postgres",
        "dialect": "duckdb",
        "environment_renderer_version": "forged-renderer",
        "checkpoint_policy_version": "forged-checkpoint-policy",
        "executor_version": "forged-executor",
        "capability_manifest": {"tools": ["answer"]},
    }.items():
        tampered = deepcopy(record)
        tampered[field] = forged
        assert not audit_record(tampered)["passed"], field
        assert not fresh_replay_record(tampered, task)["passed"], field


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_v2_result_fixture(tmp_path: Path, *, count: int = 3) -> tuple[Path, dict, list[dict]]:
    tasks = [
        {
            "example_id": f"public-{index}",
            "example_index": 100 + index,
            "db_id": "synthetic",
            "db_path": str(tmp_path / "not-opened.sqlite"),
            "question": f"Synthetic public question {index}",
            "external_knowledge": None,
        }
        for index in range(count)
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )
    source_manifest = {
        "schema_version": "synthetic-frozen-cohort-v1",
        "status": "frozen_teacher_rollout_candidate",
        "outputs": {
            "harness_tasks": {
                "path": str(tasks_path.resolve()),
                "sha256": _sha256_file(tasks_path),
                "records": count,
            }
        },
        "selection": {
            "algorithm": "preserve certified order",
            "preserved_cohort": {"task_ids_and_order_preserved": True},
        },
    }
    source_manifest_path = tmp_path / "source.manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="utf-8"
    )
    batch_limits = {
        "max_provider_attempts": 100,
        "max_provider_tokens": 1000,
        "max_wall_seconds": 100.0,
        "max_consecutive_provider_failures": 2,
        "max_total_provider_failures": 3,
        "max_consecutive_semantic_failures": 5,
    }
    source_identity = _dataset_source_identity(
        source_manifest,
        project_root=tmp_path,
    )
    selection_identity = _selection_identity(tasks, start=0, requested_size=count)
    manifest = {
        "runner": PREFIX_BATCH_RUNNER_VERSION,
        "dataset": str(tasks_path.resolve()),
        "dataset_sha256": _sha256_file(tasks_path),
        "dataset_manifest": str(source_manifest_path.resolve()),
        "dataset_manifest_sha256": _sha256_file(source_manifest_path),
        "dataset_source_identity": source_identity,
        "selection_identity": selection_identity,
        "start": 0,
        "requested_size": count,
        "max_tokens": 64,
        "max_completion_tokens": 256,
        "api_retries": 4,
        "batch_control_version": BATCH_CONTROL_VERSION,
        "batch_limits": batch_limits,
    }
    manifest["config_sha256"] = _manifest_config_sha256(manifest)
    records = [
        {
            "runner": PREFIX_BATCH_RUNNER_VERSION,
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "dataset_source_identity_sha256": source_identity[
                "dataset_source_identity_sha256"
            ],
            "selection_identity_sha256": selection_identity[
                "selection_identity_sha256"
            ],
            "selection_start": 0,
            "selection_requested_size": count,
            "batch_control_version": BATCH_CONTROL_VERSION,
            "batch_limits": batch_limits,
            "task_position": index,
            "example_id": task["example_id"],
            "correct": True,
            "failure_type": None,
            "elapsed_seconds": 0.25,
            "provider_usage": {"total_tokens": 0},
            "turns": [],
        }
        for index, task in enumerate(tasks)
    ]
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    manifest_path = result_dir / "manifest.json"
    records_path = result_dir / "all.jsonl"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status = {
        "batch_control_version": BATCH_CONTROL_VERSION,
        "state": "completed",
        "stop_code": None,
        "stop_detail": None,
        "requested_size": count,
        "completed_records": count,
        "counters": {
            "provider_attempts": 0,
            "provider_tokens": 0,
            "total_provider_failures": 0,
            "consecutive_provider_failures": 0,
            "consecutive_semantic_failures": 0,
            "semantic_failures": 0,
            "total_wall_seconds": count * 0.25,
        },
        "limits": batch_limits,
        "manifest_config_sha256": manifest["config_sha256"],
        "all_jsonl_sha256": _sha256_file(records_path),
        "recorded_at_utc": "2026-08-09T00:00:00+00:00",
    }
    (result_dir / "batch_status.json").write_text(
        json.dumps(status, sort_keys=True), encoding="utf-8"
    )
    return result_dir, manifest, records


@pytest.fixture
def _stub_record_and_replay_audits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        audit_module,
        "audit_records",
        lambda records: {
            "schema_version": "test",
            "records": len(records),
            "passed_records": len(records),
            "failed_records": 0,
            "issue_count": 0,
            "issue_types": {},
            "passed": True,
            "records_detail": [],
        },
    )
    monkeypatch.setattr(
        audit_module,
        "fresh_replay_records",
        lambda records, tasks: {
            "records": len(records),
            "passed_records": len(records),
            "failed_records": 0,
            "issue_count": 0,
            "passed": True,
            "records_detail": [],
        },
    )


def test_v2_result_dir_binds_cohort_completion_config_and_batch_status(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, _, _ = _write_v2_result_fixture(tmp_path)
    report = audit_result_dir(result_dir)
    assert report["passed"] is True
    assert report["cohort_identity"]["complete"] is True
    assert report["batch_control"]["complete"] is True
    assert set(report["artifact_sha256"]) == {
        "manifest.json",
        "all.jsonl",
        "batch_status.json",
    }


def test_v2_result_dir_rejects_synchronized_selection_identity_forgery(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, manifest, records = _write_v2_result_fixture(tmp_path)
    forged = "f" * 64
    manifest["selection_identity"]["selection_identity_sha256"] = forged
    for record in records:
        record["selection_identity_sha256"] = forged
    manifest["config_sha256"] = _manifest_config_sha256(manifest)
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    records_path = result_dir / "all.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_path = result_dir / "batch_status.json"
    status = json.loads(status_path.read_text())
    status["manifest_config_sha256"] = manifest["config_sha256"]
    status["all_jsonl_sha256"] = _sha256_file(records_path)
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["passed"] is False
    assert any(
        "selection_identity differs" in issue
        for issue in report["cohort_identity"]["issues"]
    )


def test_v2_result_dir_rejects_synchronized_dataset_source_forgery(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, manifest, records = _write_v2_result_fixture(tmp_path)
    source_path = Path(manifest["dataset_manifest"])
    source_manifest = json.loads(source_path.read_text())
    forged = "e" * 64
    source_manifest["outputs"]["harness_tasks"]["sha256"] = forged
    source_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")
    source_identity = _dataset_source_identity(source_manifest, project_root=tmp_path)
    manifest.update({
        "dataset_sha256": forged,
        "dataset_manifest_sha256": _sha256_file(source_path),
        "dataset_source_identity": source_identity,
    })
    for record in records:
        record["dataset_manifest_sha256"] = manifest["dataset_manifest_sha256"]
        record["dataset_source_identity_sha256"] = source_identity[
            "dataset_source_identity_sha256"
        ]
    manifest["config_sha256"] = _manifest_config_sha256(manifest)
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    records_path = result_dir / "all.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_path = result_dir / "batch_status.json"
    status = json.loads(status_path.read_text())
    status["manifest_config_sha256"] = manifest["config_sha256"]
    status["all_jsonl_sha256"] = _sha256_file(records_path)
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["passed"] is False
    assert any(
        "actual dataset" in issue or "audited dataset" in issue
        for issue in report["cohort_identity"]["issues"]
    )


def test_v2_result_dir_rejects_incomplete_or_stopped_batch(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, _, records = _write_v2_result_fixture(tmp_path)
    records.pop()
    records_path = result_dir / "all.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_path = result_dir / "batch_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        "state": "stopped",
        "stop_code": "max_consecutive_semantic_failures",
        "stop_detail": {},
        "completed_records": len(records),
        "all_jsonl_sha256": _sha256_file(records_path),
    })
    status["counters"]["total_wall_seconds"] = len(records) * 0.25
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["passed"] is False
    assert report["cohort_identity"]["complete"] is False
    assert report["batch_control"]["complete"] is False
    assert any("incomplete" in issue for issue in report["batch_control"]["issues"])


def test_v2_result_dir_rejects_attempt_budget_and_manifest_hash_tampering(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, manifest, records = _write_v2_result_fixture(tmp_path)
    records[0]["turns"] = [{
        "provider_attempt_events": [{
            "attempt_index": 1,
            "max_tokens": 999999,
            "shape_category": "transport",
        }]
    }]
    records_path = result_dir / "all.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_path = result_dir / "batch_status.json"
    status = json.loads(status_path.read_text())
    status["counters"]["provider_attempts"] = 1
    status["all_jsonl_sha256"] = _sha256_file(records_path)
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["passed"] is False
    assert any(
        "bounded retry progression" in issue
        for issue in report["provider_budget"]["issues"]
    )

    manifest["max_tokens"] = 65
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    report = audit_result_dir(result_dir)
    assert report["manifest_binding"]["passed"] is False
    assert any("config_sha256" in issue for issue in report["manifest_binding"]["issues"])


def test_batch_semantic_streak_excludes_explicit_infrastructure_failures(
    tmp_path: Path,
    _stub_record_and_replay_audits,
):
    result_dir, _, records = _write_v2_result_fixture(tmp_path)
    records[0].update({"correct": False, "failure_type": "hidden_verifier_error"})
    records[1].update({"correct": False, "failure_type": "batch_limit_reached"})
    records[2].update({"correct": False, "failure_type": "wrong_answer"})
    records_path = result_dir / "all.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    status_path = result_dir / "batch_status.json"
    status = json.loads(status_path.read_text())
    status["counters"].update({
        "semantic_failures": 1,
        "consecutive_semantic_failures": 1,
    })
    status["all_jsonl_sha256"] = _sha256_file(records_path)
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    report = audit_result_dir(result_dir)
    assert report["batch_control"]["passed"] is True
    assert report["batch_control"]["recomputed_counters"]["semantic_failures"] == 1
