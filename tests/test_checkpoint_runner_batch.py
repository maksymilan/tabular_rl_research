from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_modules.checkpoint_relalg import runner


def _write_certified_dataset(tmp_path: Path, *, count: int = 6):
    tasks = [
        {
            "example_index": index,
            "example_id": f"public-{index}",
            "db_id": "db",
            "question": f"public question {index}",
            "external_knowledge": None,
        }
        for index in range(count)
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    source_manifest = {
        "schema_version": "teacher-dataset-v2",
        "status": "frozen",
        "outputs": {
            "harness_tasks": {
                "path": str(tasks_path.resolve()),
                "sha256": runner._file_sha256(tasks_path),
                "records": count,
            }
        },
        "selection": {
            "algorithm": "preserve certified order",
            "preserved_cohort": {"task_ids_and_order_preserved": True},
        },
    }
    manifest_path = tmp_path / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    return tasks, tasks_path, manifest_path


def _batch_record(
    *,
    correct: bool,
    failure_type: str | None,
    attempts: int = 1,
    tokens: int = 10,
):
    return {
        "correct": correct,
        "failure_type": failure_type,
        "elapsed_seconds": 0.25,
        "provider_usage": {"total_tokens": tokens},
        "turns": [
            {"provider_attempt_events": [{} for _ in range(attempts)]}
        ],
    }


def test_certified_dataset_and_selection_identity_are_hash_bound_without_text_leak(
    tmp_path: Path,
):
    tasks, tasks_path, manifest_path = _write_certified_dataset(tmp_path)
    identity = runner._build_dataset_identity(
        tasks_path,
        manifest_path,
        tasks,
        start=1,
        requested_size=3,
    )
    encoded = json.dumps(identity, sort_keys=True)
    assert "public question" not in encoded
    assert identity["selection_identity"]["selected_count"] == 3
    assert len(identity["selection_identity"]["selection_identity_sha256"]) == 64

    reordered = list(tasks)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    assert runner._selection_identity(
        reordered, start=1, requested_size=3
    )["selection_identity_sha256"] != identity["selection_identity"][
        "selection_identity_sha256"
    ]

    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    source["outputs"]["harness_tasks"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="task hash"):
        runner._build_dataset_identity(
            tasks_path,
            manifest_path,
            tasks,
            start=1,
            requested_size=3,
        )


def test_batch_counters_distinguish_semantic_and_infrastructure_failures():
    records = [
        _batch_record(correct=False, failure_type="wrong_answer"),
        _batch_record(correct=False, failure_type="max_model_turns"),
        _batch_record(correct=False, failure_type="provider_error"),
        _batch_record(correct=False, failure_type="hidden_verifier_error"),
        _batch_record(correct=False, failure_type="wrong_answer"),
        _batch_record(correct=True, failure_type=None),
    ]
    counters = runner._batch_counters_from_records(records)
    assert counters["semantic_failures"] == 3
    assert counters["consecutive_semantic_failures"] == 0
    assert counters["total_provider_failures"] == 1
    assert counters["consecutive_provider_failures"] == 0
    assert counters["provider_attempts"] == 6
    assert counters["provider_tokens"] == 60

    five_semantic = [
        _batch_record(correct=False, failure_type="wrong_answer") for _ in range(5)
    ]
    counters = runner._batch_counters_from_records(five_semantic)
    limits = {
        "max_provider_attempts": 100,
        "max_provider_tokens": 1000,
        "max_wall_seconds": 100.0,
        "max_consecutive_provider_failures": 2,
        "max_total_provider_failures": 3,
        "max_consecutive_semantic_failures": 5,
    }
    assert runner._batch_stop_decision(five_semantic, counters, limits) == (
        "max_consecutive_semantic_failures",
        {"counter": 5, "limit": 5},
    )

    billing_failure = _batch_record(
        correct=False,
        failure_type="provider_error",
        tokens=0,
    )
    billing_failure["turns"][0]["provider_error"] = {
        "http_status": 402,
        "retryable": False,
    }
    counters = runner._batch_counters_from_records([billing_failure])
    assert runner._batch_stop_decision(
        [billing_failure], counters, limits
    ) == (
        "non_retryable_provider_http",
        {"http_status": 402, "retryable": False},
    )


def test_batch_guard_caps_retries_and_accounts_provider_response():
    limits = {
        "max_provider_attempts": 3,
        "max_provider_tokens": 100,
        "max_wall_seconds": 100.0,
        "max_consecutive_provider_failures": 2,
        "max_total_provider_failures": 3,
        "max_consecutive_semantic_failures": 5,
    }
    counters = runner._batch_counters_from_records(
        [_batch_record(correct=True, failure_type=None, attempts=2, tokens=20)]
    )
    guard = runner.BatchRequestGuard(counters=counters, limits=limits)
    assert guard.allowed_retries(4) == 1

    class Response:
        provider_attempt_count = 1
        provider_attempt_events = [{}]
        usage = {"total_tokens": 30}

    guard.observe_response(Response())
    assert guard.stop_decision() == (
        "max_provider_attempts",
        {"counter": 3, "limit": 3},
    )


def test_main_stops_after_five_semantic_failures_and_resume_sends_no_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, tasks_path, manifest_path = _write_certified_dataset(tmp_path, count=6)
    result_dir = tmp_path / "result"
    episode_calls: list[int] = []

    class FakeClient:
        request_audit_options = {}

        def __init__(self, **kwargs):
            self.carrier = kwargs["carrier"]

        def verify_model(self):
            return {"model": runner.DEFAULT_MODEL, "verified": True}

    def fake_episode(task, *, task_position, artifact_identity_fields, batch_limits, **kwargs):
        episode_calls.append(task_position)
        return {
            **artifact_identity_fields,
            "batch_limits": dict(batch_limits),
            "task_position": task_position,
            "example_index": task["example_index"],
            "example_id": task["example_id"],
            "correct": False,
            "failure_type": "wrong_answer",
            "elapsed_seconds": 0.1,
            "provider_usage": {"total_tokens": 10},
            "turns": [{"provider_attempt_events": [{}]}],
            "steps": 1,
            "errors": 0,
        }

    monkeypatch.setattr(runner, "load_api_config", lambda path: ("ignored", "https://api.deepseek.com"))
    monkeypatch.setattr(runner, "DeepSeekNativeClient", FakeClient)
    monkeypatch.setattr(runner, "run_episode", fake_episode)
    monkeypatch.setattr(runner, "audit_result_dir", lambda path: {"passed": False})
    argv = [
        "--mode",
        "hybrid",
        "--carrier",
        "text-json",
        "--experiment-arm",
        "A",
        "--tasks-json",
        str(tasks_path),
        "--dataset-manifest",
        str(manifest_path),
        "--result-dir",
        str(result_dir),
        "--n",
        "6",
    ]
    assert runner.main(argv) == 2
    assert episode_calls == [0, 1, 2, 3, 4]
    status = json.loads((result_dir / "batch_status.json").read_text())
    assert status["state"] == "stopped"
    assert status["stop_code"] == "max_consecutive_semantic_failures"
    assert status["completed_records"] == 5
    assert status["counters"]["consecutive_semantic_failures"] == 5

    assert runner.main([*argv, "--resume"]) == 2
    assert episode_calls == [0, 1, 2, 3, 4]
