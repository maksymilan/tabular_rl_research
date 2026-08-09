from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tool_modules.checkpoint_relalg.analyze_carrier_ab as analyzer_module
from tool_modules.checkpoint_relalg.analyze_carrier_ab import (
    CarrierAnalysisError,
    analyze_result_dirs,
    render_markdown,
)
from tool_modules.checkpoint_relalg.protocol import (
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
    assistant_carrier_protocol,
    capability_manifest,
    carrier_experiment_arm,
    prompt_hash,
    tool_schema_hash,
)
from tool_modules.checkpoint_relalg.provider import (
    NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION,
)
from tool_modules.registry import build_checkpoint_relalg_tool_scheme


@pytest.fixture(autouse=True)
def _stub_current_fresh_replay(monkeypatch: pytest.MonkeyPatch):
    # Synthetic records intentionally omit real DB/question payloads. Production
    # analysis calls the current structural auditor and fresh replay here.
    monkeypatch.setattr(
        analyzer_module,
        "_run_current_audit",
        lambda directory, records, manifest: "current-audit-hash",
    )


def _manifest(carrier: str, *, start: int, size: int) -> dict:
    native = carrier == CARRIER_NATIVE_TOOL_CALLS
    scheme_fields = build_checkpoint_relalg_tool_scheme(
        mode="hybrid",
        carrier=carrier,
    ).manifest_fields()
    request_options = {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-v4-flash",
        "carrier": carrier,
        "assistant_carrier": assistant_carrier_protocol(carrier),
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        **(
            {"tool_choice": "auto"}
            if native
            else {"response_format": {"type": "json_object"}}
        ),
    }
    return {
        **scheme_fields,
        "dataset_sha256": "dataset-hash",
        "mode": "hybrid",
        "model": "deepseek-v4-flash",
        "backend": "sqlite",
        "dialect": "sqlite",
        "environment_renderer_version": "renderer-v1",
        "checkpoint_policy_version": "checkpoint-v1",
        "executor_version": "executor-v1",
        "runtime_config": {
            "max_model_turns": 20,
            "max_primitive_calls": 30,
            "max_checkpoints": 8,
            "max_restores": 3,
            "sql_timeout_seconds": 20.0,
            "max_artifact_rows": 100_000,
            "max_artifact_bytes": 1024,
            "max_cell_bytes": 256,
        },
        "max_tokens": 4096,
        "max_completion_tokens": 8192,
        "api_retries": 4,
        "api_timeout_seconds": 300,
        "denotation_comparison": "bird-set",
        "strict_artifact_audit_version": "strict-v1",
        "carrier_ablation_protocol_version": "carrier-ab-v1",
        "carrier_policy_version": "carrier-policy-v1",
        "tool_schema_hash": tool_schema_hash("hybrid"),
        "capability_manifest": capability_manifest("hybrid", carrier),
        "carrier": carrier,
        "assistant_carrier": assistant_carrier_protocol(carrier),
        "experiment_arm": carrier_experiment_arm(carrier),
        "prompt_hash": prompt_hash("hybrid", teacher=True, carrier=carrier),
        "provider_request_options": request_options,
        "provider_verification": {
            "authenticated": True,
            "model_available": True,
            "model": "deepseek-v4-flash",
        },
        "start": start,
        "requested_size": size,
    }


def _turn(
    carrier: str,
    *,
    action_count: int = 1,
    valid: bool = True,
    error_code: str | None = None,
) -> dict:
    result = (
        {"status": "success"}
        if error_code is None
        else {
            "status": "error",
            "error": {"type": "protocol_error", "code": error_code},
        }
    )
    base = {
        "provider_retry_events": [],
        "provider_attempt_events": [
            {
                "attempt_index": 1,
                "max_tokens": 4096,
                "finish_reason": "stop",
                "usage": {},
                "shape_category": "accepted_response",
                "response_envelope_sha256": None,
            }
        ],
        "result": result,
        "carrier_metrics": {
            "carrier": carrier,
            "provider_response_present": True,
            "authored_action_count": action_count,
            "exact_single_action": action_count == 1,
            "carrier_envelope_valid": valid,
            "carrier_error_code": error_code if not valid else None,
            "action_validation_error_code": error_code if valid else None,
            "provider_attempt_count": 1,
            "provider_elapsed_seconds": 0.25,
        },
    }
    if carrier == CARRIER_NATIVE_TOOL_CALLS:
        base["assistant_message"] = {"tool_calls": [{} for _ in range(action_count)]}
        if valid:
            base["action"] = {}
        else:
            base["native_rejection"] = {
                "code": error_code or "invalid_tool_call_count",
                "call_count": action_count,
            }
    else:
        if valid:
            base["action"] = {}
    return base


def _record(
    position: int,
    carrier: str,
    *,
    correct: bool,
    legal: bool,
    reject_then_recover: bool = False,
) -> dict:
    turns = []
    if reject_then_recover:
        turns.append(
            _turn(
                carrier,
                action_count=2,
                valid=False,
                error_code="invalid_tool_call_count",
            )
        )
    turns.append(_turn(carrier))
    return {
        "task_position": position,
        "example_id": f"task-{position}",
        "correct": correct,
        "legal": legal,
        "strict_artifact_accuracy": correct,
        "schema_match": legal,
        "turns": turns,
        "primitive_calls": len(turns),
        "restore_count": int(position == 14),
        "failure_type": None if legal else "wrong_answer",
        "provider_usage": {
            "prompt_tokens": 100 + position,
            "prompt_cache_hit_tokens": 20,
            "prompt_cache_miss_tokens": 80 + position,
            "completion_tokens": 10,
            "completion_tokens_details.reasoning_tokens": position
            + (2 if carrier == CARRIER_NATIVE_TOOL_CALLS else 0),
            "total_tokens": 110 + position,
        },
        "elapsed_seconds": float(position),
    }


def _write_batch(
    root: Path,
    name: str,
    carrier: str,
    records: list[dict],
) -> Path:
    directory = root / name
    directory.mkdir()
    manifest = _manifest(
        carrier,
        start=records[0]["task_position"],
        size=len(records),
    )
    records = [
        {
            **record,
            **{
                field: manifest.get(field)
                for field in analyzer_module.RECORD_MANIFEST_FIELDS
                if field in manifest
            },
        }
        for record in records
    ]
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "all.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    audit = {
        "passed": True,
        "issue_count": 0,
        "records": len(records),
        "fresh_replay": {
            "passed": True,
            "issue_count": 0,
            "records": len(records),
        },
        "manifest_binding": {
            "passed": True,
            "issue_count": 0,
            "records": len(records),
        },
        "artifact_sha256": {
            "manifest.json": _sha256(directory / "manifest.json"),
            "all.jsonl": _sha256(directory / "all.jsonl"),
        },
    }
    (directory / "checkpoint_relalg_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_audit_hashes(directory: Path) -> None:
    audit_path = directory / "checkpoint_relalg_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["artifact_sha256"] = {
        "manifest.json": _sha256(directory / "manifest.json"),
        "all.jsonl": _sha256(directory / "all.jsonl"),
    }
    audit_path.write_text(json.dumps(audit))


def _matched_dirs(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = CARRIER_TEXT_JSON
    native = CARRIER_NATIVE_TOOL_CALLS
    outcomes_a = [(True, True), (True, True), (False, False), (False, False)]
    outcomes_b = [(True, True), (False, False), (True, True), (False, False)]
    a_records = [
        _record(position, text, correct=correct, legal=legal)
        for position, (correct, legal) in zip(range(12, 16), outcomes_a)
    ]
    b_records = [
        _record(
            position,
            native,
            correct=correct,
            legal=legal,
            reject_then_recover=position == 12,
        )
        for position, (correct, legal) in zip(range(12, 16), outcomes_b)
    ]
    a_dirs = [
        _write_batch(tmp_path, "a_12_13", text, a_records[:2]),
        _write_batch(tmp_path, "a_14_15", text, a_records[2:]),
    ]
    b_dirs = [
        _write_batch(tmp_path, "b_12_13", native, b_records[:2]),
        _write_batch(tmp_path, "b_14_15", native, b_records[2:]),
    ]
    return a_dirs, b_dirs


def test_matched_multi_batch_analysis_is_paired_and_content_blind(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    report = analyze_result_dirs(a_dirs, b_dirs)
    assert report["cohort"] == {
        "size": 4,
        "task_positions": [12, 13, 14, 15],
        "task_ids": ["task-12", "task-13", "task-14", "task-15"],
    }
    assert report["paired"]["correct"] == {
        "both": 1,
        "a_only": 1,
        "b_only": 1,
        "neither": 1,
        "discordant": 2,
        "net_b_minus_a": 0,
        "exact_two_sided_p": 1.0,
    }
    assert report["arms"]["B"]["multi_action_turns"] == 1
    assert report["arms"]["B"]["carrier_rejections"] == 1
    assert report["arms"]["B"]["recovered_after_carrier_rejection"] == 1
    assert report["arms"]["A"]["tokens"]["total_tokens"]["available"] == 4
    assert report["paired"]["b_minus_a"]["elapsed_seconds"]["available"] == 4
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in ("question", "gold_sql", "reasoning_content", "answer_rows", "db_value"):
        assert forbidden not in encoded
    markdown = render_markdown(report)
    assert "| correct | 2 | 2 | 0 / 1 |" in markdown
    assert "prompt_cache_hit_tokens" in markdown


@pytest.mark.parametrize("native_identity", [None, "native-response-envelope-v1"])
def test_analyzer_requires_current_native_envelope_but_accepts_legacy_text_a(
    tmp_path: Path,
    native_identity: str | None,
):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    for directory in a_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        assert "provider_response_envelope_version" not in manifest
        for line in (directory / "all.jsonl").read_text().splitlines():
            assert "provider_response_envelope_version" not in json.loads(line)
    assert analyze_result_dirs(a_dirs, b_dirs)["cohort"]["size"] == 4

    for directory in b_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        records_path = directory / "all.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        if native_identity is None:
            manifest.pop("provider_response_envelope_version")
            for record in records:
                record.pop("provider_response_envelope_version")
        else:
            manifest["provider_response_envelope_version"] = native_identity
            for record in records:
                record["provider_response_envelope_version"] = native_identity
        manifest_path.write_text(json.dumps(manifest))
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        _refresh_audit_hashes(directory)

    with pytest.raises(
        CarrierAnalysisError,
        match="provider_response_envelope_version",
    ):
        analyze_result_dirs(a_dirs, b_dirs)

    assert (
        _manifest(CARRIER_NATIVE_TOOL_CALLS, start=12, size=2)[
            "provider_response_envelope_version"
        ]
        == NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
    )


def test_analyzer_rejects_synchronized_text_envelope_identity_forgery(
    tmp_path: Path,
):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    for directory in a_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provider_response_envelope_version"] = "forged-native-envelope"
        manifest_path.write_text(json.dumps(manifest))
        records_path = directory / "all.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        for record in records:
            record["provider_response_envelope_version"] = "forged-native-envelope"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        _refresh_audit_hashes(directory)

    with pytest.raises(
        CarrierAnalysisError,
        match="provider_response_envelope_version",
    ):
        analyze_result_dirs(a_dirs, b_dirs)


def test_analysis_refuses_failed_audit(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    audit_path = b_dirs[0] / "checkpoint_relalg_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["passed"] = False
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(CarrierAnalysisError, match="audit/fresh replay"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_analysis_refuses_incomplete_pair_and_budget_mismatch(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    with pytest.raises(CarrierAnalysisError, match="incomplete or task order"):
        analyze_result_dirs(a_dirs, b_dirs[:-1])

    a_dirs, b_dirs = _matched_dirs(tmp_path / "second")
    manifest_path = b_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_config"]["max_primitive_calls"] = 29
    manifest_path.write_text(json.dumps(manifest))
    records_path = b_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    for record in records:
        record["runtime_config"] = manifest["runtime_config"]
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(b_dirs[0])
    with pytest.raises(CarrierAnalysisError, match="batch manifests|common manifests"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_text_carrier_requires_content_free_metrics(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = a_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["turns"][0].pop("carrier_metrics")
    records[0]["turns"][0]["assistant_message"] = {
        "content": "opaque-and-forbidden-to-the-analyzer"
    }
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(a_dirs[0])
    # Keep the audit structurally passing to prove the content-blind guard is
    # independently enforced by the analyzer.
    with pytest.raises(CarrierAnalysisError, match="content-free carrier_metrics"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_analysis_enforces_frozen_arm_mapping_and_current_reaudit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    manifest_path = a_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["carrier"] = CARRIER_NATIVE_TOOL_CALLS
    manifest["assistant_carrier"] = assistant_carrier_protocol(CARRIER_NATIVE_TOOL_CALLS)
    manifest["experiment_arm"] = "B"
    manifest_path.write_text(json.dumps(manifest))
    _refresh_audit_hashes(a_dirs[0])
    with pytest.raises(CarrierAnalysisError, match="arm A must declare"):
        analyze_result_dirs(a_dirs, b_dirs)

    a_dirs, b_dirs = _matched_dirs(tmp_path / "reaudit")

    def fail_current_audit(directory, records, manifest):
        raise CarrierAnalysisError("current structural audit/fresh replay failed")

    monkeypatch.setattr(analyzer_module, "_run_current_audit", fail_current_audit)
    with pytest.raises(CarrierAnalysisError, match="current structural audit"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_capability_semantics_must_match_after_carrier_normalization(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    for directory in b_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["capability_manifest"]["tools"].append("different_tool")
        manifest_path.write_text(json.dumps(manifest))
        records_path = directory / "all.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        for record in records:
            record["capability_manifest"] = manifest["capability_manifest"]
        records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        _refresh_audit_hashes(directory)
    with pytest.raises(CarrierAnalysisError, match="current protocol|common manifests"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_semantic_protocol_error_is_not_a_carrier_rejection(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = b_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["turns"] = [
        _turn(
            CARRIER_NATIVE_TOOL_CALLS,
            action_count=1,
            valid=True,
            error_code="unknown_tool",
        )
    ]
    records[0]["primitive_calls"] = 1
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(b_dirs[0])
    report = analyze_result_dirs(a_dirs, b_dirs)
    assert report["arms"]["B"]["carrier_rejections"] == 0
    assert report["arms"]["B"]["semantic_protocol_errors"] == 1
    assert report["arms"]["B"]["protocol_errors"] == 1


def test_analysis_binds_hashes_records_and_provider_attempt_history(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = b_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["runtime_config"]["max_primitive_calls"] = 999
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(b_dirs[0])
    with pytest.raises(CarrierAnalysisError, match="runtime_config differs from manifest"):
        analyze_result_dirs(a_dirs, b_dirs)

    a_dirs, b_dirs = _matched_dirs(tmp_path / "attempt-count")
    records_path = b_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["turns"][0]["carrier_metrics"]["provider_attempt_count"] = 2
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(b_dirs[0])
    with pytest.raises(CarrierAnalysisError, match="differs from retry history"):
        analyze_result_dirs(a_dirs, b_dirs)

    a_dirs, b_dirs = _matched_dirs(tmp_path / "artifact-hash")
    manifest_path = b_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unbound_note"] = "tampered after audit"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CarrierAnalysisError, match="audit/fresh replay"):
        analyze_result_dirs(a_dirs, b_dirs)

    a_dirs, b_dirs = _matched_dirs(tmp_path / "synchronized-identity-forgery")
    for directory in a_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["protocol_hash"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest))
        records_path = directory / "all.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        for record in records:
            record["protocol_hash"] = manifest["protocol_hash"]
        records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        _refresh_audit_hashes(directory)
    with pytest.raises(CarrierAnalysisError, match="current tool scheme"):
        analyze_result_dirs(a_dirs, b_dirs)


def test_provider_attempt_categories_are_reported_without_payload_access(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = a_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    turn = records[0]["turns"][0]
    turn["provider_retry_events"] = [{"attempt": 1, "type": "empty_text"}]
    turn["provider_attempt_events"] = [
        {"shape_category": "empty_text"},
        {"shape_category": "accepted_response"},
    ]
    turn["carrier_metrics"]["provider_attempt_count"] = 2
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(a_dirs[0])
    report = analyze_result_dirs(a_dirs, b_dirs)
    assert report["arms"]["A"]["provider_attempts"] == 5
    assert report["arms"]["A"]["provider_attempt_categories"]["empty_text"] == 1
    assert report["arms"]["A"]["provider_attempt_categories"]["accepted_response"] == 4
    assert report["execution_order_evidence"]["schedule_verified"] is False
    assert report["execution_order_evidence"]["counterbalanced"] is None


def test_terminal_provider_categories_and_nested_reasoning_tokens(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = a_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]

    resource_turn = records[0]["turns"][0]
    accepted_event = resource_turn["provider_attempt_events"][0]
    accepted_event["raw_assistant_message"] = {
        "content": "opaque-provider-payload-never-read-by-analyzer",
        "reasoning_content": "opaque-reasoning-never-read-by-analyzer",
    }
    resource_turn["provider_retry_events"] = [
        {"attempt": 1, "type": "insufficient_system_resource"}
    ]
    resource_turn["provider_attempt_events"] = [
        {
            **accepted_event,
            "attempt_index": 1,
            "shape_category": "insufficient_system_resource",
        },
        {
            **accepted_event,
            "attempt_index": 2,
            "shape_category": "accepted_response",
        },
    ]
    resource_turn["carrier_metrics"]["provider_attempt_count"] = 2

    filtered_turn = records[1]["turns"][0]
    filtered_turn["provider_attempt_events"][0]["shape_category"] = "content_filter"
    filtered_turn["carrier_metrics"].update(
        {
            "provider_response_present": False,
            "authored_action_count": 0,
            "exact_single_action": False,
            "carrier_envelope_valid": False,
        }
    )
    filtered_turn.pop("action", None)
    filtered_turn.pop("result", None)
    filtered_turn["provider_error"] = {"type": "ProviderContentFiltered"}

    # Real runner output is flat and wins if both representations exist; the
    # nested shape remains a compatibility fallback.
    records[0]["provider_usage"]["completion_tokens_details"] = {
        "reasoning_tokens": 999
    }
    nested_reasoning = records[1]["provider_usage"].pop(
        "completion_tokens_details.reasoning_tokens"
    )
    records[1]["provider_usage"]["completion_tokens_details"] = {
        "reasoning_tokens": nested_reasoning
    }

    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(a_dirs[0])
    report = analyze_result_dirs(a_dirs, b_dirs)

    assert report["arms"]["A"]["provider_attempts"] == 5
    categories = report["arms"]["A"]["provider_attempt_categories"]
    assert categories["insufficient_system_resource"] == 1
    assert categories["content_filter"] == 1
    assert categories["accepted_response"] == 3
    reasoning = "completion_tokens_details.reasoning_tokens"
    assert report["arms"]["A"]["tokens"][reasoning]["total"] == 54
    assert report["arms"]["B"]["tokens"][reasoning]["total"] == 62
    assert report["paired"]["b_minus_a"]["tokens"][reasoning]["mean"] == 2.0
    assert report["pairs"][0]["tokens"][reasoning] == {"A": 12, "B": 14}
    encoded = json.dumps(report, sort_keys=True)
    assert "raw_assistant_message" not in encoded
    assert "opaque-provider-payload-never-read-by-analyzer" not in encoded
    assert "opaque-reasoning-never-read-by-analyzer" not in encoded


def test_missing_reasoning_token_stays_null_and_is_excluded_from_pairs(tmp_path: Path):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    records_path = a_dirs[0] / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["provider_usage"].pop("completion_tokens_details.reasoning_tokens")
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(a_dirs[0])
    report = analyze_result_dirs(a_dirs, b_dirs)
    reasoning = "completion_tokens_details.reasoning_tokens"
    assert report["pairs"][0]["tokens"][reasoning]["A"] is None
    assert report["arms"]["A"]["tokens"][reasoning]["available"] == 3
    assert report["paired"]["b_minus_a"]["tokens"][reasoning]["available"] == 3


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("protocol_hash", "current tool scheme"),
        ("tool_schema_hash", "current protocol"),
        ("prompt_hash", "prompt hash"),
    ],
)
def test_synchronized_manifest_record_identity_hash_forgery_is_rejected(
    tmp_path: Path,
    field: str,
    expected_error: str,
):
    a_dirs, b_dirs = _matched_dirs(tmp_path)
    directory = a_dirs[0]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = "e" * 64
    manifest_path.write_text(json.dumps(manifest))
    records_path = directory / "all.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    for record in records:
        record[field] = manifest[field]
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    _refresh_audit_hashes(directory)
    with pytest.raises(CarrierAnalysisError, match=expected_error):
        analyze_result_dirs(a_dirs, b_dirs)
