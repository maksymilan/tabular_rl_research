from __future__ import annotations

import sqlite3
import hashlib
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tool_modules.checkpoint_relalg.provider import (
    NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION,
    NativeAssistantResponse,
    _normalize_assistant_message,
)
from tool_modules.checkpoint_relalg.audit import audit_record, fresh_replay_record
from tool_modules.checkpoint_relalg.checkpoint_store import (
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
)
from tool_modules.checkpoint_relalg.protocol import capability_manifest, prompt_hash
from tool_modules.checkpoint_relalg.runner import (
    OPERATIONAL_RESUME_POLICY_VERSION,
    _open_artifact_writer,
    _phase_execution_styles,
    build_manifest,
    run_episode,
)
from tool_modules.checkpoint_relalg.runtime import RuntimeConfig
from tool_modules.registry import build_checkpoint_relalg_tool_scheme


class FakeClient:
    carrier = "native-tool-calls"
    request_audit_options = {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "fake",
        "carrier": "native-tool-calls",
        "assistant_carrier": "provider-native-single-tool-call-v1",
        "tool_choice": "auto",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }

    def __init__(self, actions):
        self.actions = list(actions)
        self.requests = []

    def request_turn_with_retries(self, *, messages, tools, **kwargs):
        self.requests.append(messages)
        action = self.actions.pop(0)
        raw_calls = [
            {
                "id": f"call_{len(self.requests)}_{index}",
                "index": index,
                "type": "function",
                "function": {
                    "name": item[0],
                    "arguments": __import__("json").dumps(item[1]),
                },
            }
            for index, item in enumerate(action)
        ]
        message, calls = _normalize_assistant_message(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "causal test",
                "tool_calls": raw_calls,
            }
        )
        usage = {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }
        envelope_hash = hashlib.sha256(json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")).hexdigest()
        return NativeAssistantResponse(
            message=message,
            calls=calls,
            finish_reason="tool_calls",
            usage=usage,
            response_metadata={"model": "fake"},
            raw_assistant_message=message,
            retry_events=[],
            provider_attempt_count=1,
            provider_elapsed_seconds=0.01,
            provider_attempt_events=[{
                "attempt_index": 1,
                "max_tokens": kwargs["max_tokens"],
                "finish_reason": "tool_calls",
                "usage": usage,
                "shape_category": "accepted_response",
                "response_envelope_sha256": envelope_hash,
                "response_model": "fake",
                "raw_assistant_message": message,
                "elapsed_seconds_from_request_start": 0.01,
            }],
        )


def _task(tmp_path: Path):
    db_path = tmp_path / "task.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        "CREATE TABLE items(id INTEGER PRIMARY KEY, category TEXT);"
        "INSERT INTO items VALUES (1, 'x'), (2, 'y');"
    )
    connection.close()
    return {
        "example_index": 7,
        "example_id": "test_7",
        "db_id": "test",
        "db_path": str(db_path),
        "question": "How many items are there?",
        "external_knowledge": None,
        "gold_sql": "SELECT COUNT(*) FROM items",
    }


def test_semantic_atomic_runner_and_fresh_replay_are_profile_bound(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [
                (
                    "group_aggregate",
                    {
                        "table": "items",
                        "group_by": [],
                        "metrics": [{"op": "count", "column": "*", "as": "n"}],
                    },
                )
            ],
            [("answer", {"table": "group_aggregate_001"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        client=client,
        runtime_config=RuntimeConfig(),
        max_model_turns=4,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["correct"] and record["legal"]
    assert record["atomic_operator_profile"] == "semantic-v2"
    assert "commit_checkpoint" in record["top_level_tools"]
    assert "restore_checkpoint" in record["top_level_tools"]
    assert "sort" not in record["top_level_tools"]
    assert record["atomic_calls"] == 1
    assert record["phase_execution_styles"] == {"phase_000": "atomic_only"}
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]


def test_semantic_milestone_v5_policy_is_bound_in_manifest_record_and_replay(
    tmp_path,
):
    task = _task(tmp_path)
    policy = CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
    client = FakeClient(
        [
            [
                (
                    "filter_rows",
                    {
                        "table": "items",
                        "conditions": {
                            "op": "=",
                            "left": {"column": "category"},
                            "right": {"value": "x"},
                        },
                    },
                )
            ],
            [
                (
                    "filter_rows",
                    {
                        "table": "items",
                        "conditions": {
                            "op": "=",
                            "left": {"column": "category"},
                            "right": {"value": "y"},
                        },
                    },
                )
            ],
            [
                (
                    "commit_checkpoint",
                    {
                        "progress_summary": [
                            "Two grounded category populations are available."
                        ],
                        "remaining_uncertainties": [
                            "The exact total relation remains to be constructed."
                        ],
                        "next_targets": ["Aggregate the complete item population."],
                    },
                )
            ],
            [
                (
                    "group_aggregate",
                    {
                        "table": "items",
                        "group_by": [],
                        "metrics": [{"op": "count", "column": "*", "as": "n"}],
                    },
                )
            ],
            [("answer", {"table": "group_aggregate_003"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        checkpoint_guidance_profile="semantic-milestone-v5",
        client=client,
        runtime_config=RuntimeConfig(
            checkpoint_commit_eligibility_policy=policy,
        ),
        max_model_turns=8,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )

    eligibility = record["capability_manifest"]["checkpoint_commit_eligibility"]
    assert record["correct"] and record["legal"]
    assert record["checkpoint_guidance_profile"] == "semantic-milestone-v5"
    assert record["checkpoint_commit_eligibility_policy"] == policy
    assert record["runtime_config"]["checkpoint_commit_eligibility_policy"] == policy
    assert record["final_runtime"]["checkpoint_commit_eligibility_policy"] == policy
    assert eligibility == {
        "policy": policy,
        "first_commit_min_milestone_producers": 2,
        "later_commit_min_milestone_producers": 3,
        "milestone_producer_tools": [
            "filter_rows",
            "group_aggregate",
            "join",
            "set_operation",
        ],
        "requires_new_active_artifacts_at_least_quota": True,
    }
    assert record["checkpoint_count"] == 1
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("{}\n", encoding="utf-8")
    manifest = build_manifest(
        SimpleNamespace(
            mode="atomic",
            carrier="native-tool-calls",
            atomic_operator_profile="semantic-v2",
            checkpoint_guidance_profile="semantic-milestone-v5",
            experiment_arm=None,
            within_batch_order=None,
            tasks_json=tasks_path,
            start=0,
            n=1,
            model="fake",
            max_primitive_calls=30,
            max_checkpoints=8,
            max_restores=3,
            sql_timeout_seconds=20.0,
            max_artifact_rows=100_000,
            max_artifact_bytes=64 * 1024 * 1024,
            max_cell_bytes=4 * 1024 * 1024,
            max_model_turns=8,
            max_tokens=128,
            max_completion_tokens=256,
            api_retries=2,
            api_timeout_seconds=300,
        ),
        provider_verification={"model": "fake"},
    )
    assert manifest["checkpoint_commit_eligibility_policy"] == policy
    assert manifest["runtime_config"]["checkpoint_commit_eligibility_policy"] == policy
    assert manifest["capability_manifest"]["checkpoint_commit_eligibility"] == eligibility
    assert manifest["protocol_hash"] == record["protocol_hash"]

    tampered_values = []

    top_level_tamper = deepcopy(record)
    top_level_tamper["checkpoint_commit_eligibility_policy"] = (
        CHECKPOINT_COMMIT_ELIGIBILITY_NONE
    )
    tampered_values.append(top_level_tamper)

    runtime_tamper = deepcopy(record)
    runtime_tamper["runtime_config"]["checkpoint_commit_eligibility_policy"] = (
        CHECKPOINT_COMMIT_ELIGIBILITY_NONE
    )
    tampered_values.append(runtime_tamper)

    capability_tamper = deepcopy(record)
    capability_tamper["capability_manifest"]["checkpoint_commit_eligibility"][
        "later_commit_min_milestone_producers"
    ] = 2
    tampered_values.append(capability_tamper)

    protocol_tamper = deepcopy(record)
    protocol_tamper["protocol_hash"] = "forged-protocol-hash"
    tampered_values.append(protocol_tamper)

    for tampered in tampered_values:
        assert not audit_record(tampered)["passed"]
        assert not fresh_replay_record(tampered, task)["passed"]

    # Even synchronized identity-field relabeling cannot turn the actual v5 provider
    # prompt/history into a v4 episode.
    synchronized = deepcopy(record)
    synchronized["checkpoint_guidance_profile"] = "semantic-milestone-v4"
    synchronized.pop("checkpoint_commit_eligibility_policy")
    synchronized["runtime_config"].pop("checkpoint_commit_eligibility_policy")
    synchronized["capability_manifest"] = capability_manifest(
        "atomic",
        "native-tool-calls",
        "semantic-v2",
        CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    )
    synchronized.update(
        build_checkpoint_relalg_tool_scheme(
            mode="atomic",
            carrier="native-tool-calls",
            atomic_operator_profile="semantic-v2",
            checkpoint_commit_eligibility_policy=(
                CHECKPOINT_COMMIT_ELIGIBILITY_NONE
            ),
        ).manifest_fields()
    )
    synchronized["prompt_hash"] = prompt_hash(
        "atomic",
        teacher=True,
        carrier="native-tool-calls",
        checkpoint_guidance_profile="semantic-milestone-v4",
        atomic_operator_profile="semantic-v2",
    )
    synchronized["teacher_prompt_sha256"] = synchronized["prompt_hash"]
    structural = audit_record(synchronized)
    replay = fresh_replay_record(synchronized, task)
    assert not structural["passed"]
    assert not replay["passed"]
    assert any("system prompt differs" in issue for issue in structural["issues"])
    assert any("system differs under fresh replay" in issue for issue in replay["issues"])


def test_operational_resume_preserves_original_start_and_rejects_identity_drift(
    tmp_path: Path,
):
    result_dir = tmp_path / "resume"
    first_manifest = {
        "run_started_at_utc": "2026-08-09T00:00:00+00:00",
        "protocol_hash": "new-native-protocol",
        "provider_response_envelope_version": (
            NATIVE_PROVIDER_RESPONSE_ENVELOPE_VERSION
        ),
    }
    _open_artifact_writer(result_dir, first_manifest, resume=False)

    resumed_manifest = {
        **first_manifest,
        "run_started_at_utc": "2026-08-09T00:05:00+00:00",
    }
    _open_artifact_writer(result_dir, resumed_manifest, resume=True)
    stored_manifest = json.loads((result_dir / "manifest.json").read_text())
    assert stored_manifest["run_started_at_utc"] == first_manifest["run_started_at_utc"]
    events = [
        json.loads(line)
        for line in (result_dir / "operational_resume_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(events) == 1
    assert set(events[0]["differences"]) == {"run_started_at_utc"}
    assert events[0]["metadata"]["policy_version"] == OPERATIONAL_RESUME_POLICY_VERSION

    old_native_identity = {
        **resumed_manifest,
        "protocol_hash": "old-native-protocol",
        "provider_response_envelope_version": "native-response-envelope-v1",
    }
    with pytest.raises(ValueError, match="existing manifest differs"):
        _open_artifact_writer(result_dir, old_native_identity, resume=True)


def test_causal_runner_hides_reference_and_resets_history_after_checkpoint(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [("describe_table", {"tables": ["items"]})],
            [
                (
                    "commit_checkpoint",
                    {
                        "progress_summary": ["Confirmed the item schema."],
                        "remaining_uncertainties": ["The count is not constructed."],
                        "next_targets": ["Construct the exact count relation."],
                    },
                )
            ],
            [("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM items"})],
            [("answer", {"table": "sql_001"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        runtime_config=RuntimeConfig(),
        max_model_turns=8,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["correct"] and record["legal"]
    assert "gold_sql" not in record
    assert "index" not in record["turns"][0]["action"]
    assert record["turns"][0]["assistant_message"]["tool_calls"][0]["index"] == 0
    assert [message["role"] for message in client.requests[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert record["turns"][1]["provider_phase_history_reset"]
    # New phase request contains no assistant/tool transcript from the old phase.
    assert [message["role"] for message in client.requests[2]] == ["system", "user"]
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]

    forged = deepcopy(record)
    final_runtime = forged["final_runtime"]
    forged["turns"].append({
        "model_turn_index": len(forged["turns"]) + 1,
        "carrier": "native-tool-calls",
        "carrier_metrics": {
            "carrier": "native-tool-calls",
            "provider_response_present": False,
            "authored_action_count": 0,
            "exact_single_action": False,
            "carrier_envelope_valid": False,
            "carrier_error_code": None,
            "action_validation_error_code": None,
            "provider_attempt_count": 1,
            "provider_elapsed_seconds": 0.01,
        },
        "model_input": deepcopy(forged["turns"][-1]["model_input"]),
        "phase_id_before": final_runtime["phase_id"],
        "checkpoint_id_before": final_runtime["checkpoint_id"],
        "environment_state_hash_before": final_runtime["environment_state_hash"],
        "provider_error": {
            "type": "ProviderError",
            "message": "forged post-terminal provider failure",
            "finish_reason": None,
            "response_envelope_sha256": None,
            "response_model": None,
        },
        "provider_retry_events": [],
        "provider_attempt_events": [{
            "attempt_index": 1,
            "max_tokens": 128,
            "finish_reason": None,
            "usage": {},
            "shape_category": "transport",
            "response_envelope_sha256": None,
            "response_model": None,
            "raw_assistant_message": None,
            "elapsed_seconds_from_request_start": 0.0,
        }],
        "provider_failed_usage": {},
        "provider_usage": {},
    })
    forged["failure_type"] = "provider_error"
    structural = audit_record(forged)
    replay = fresh_replay_record(forged, task)
    assert structural["passed"] is False
    assert any("after terminal" in issue for issue in structural["issues"])
    assert replay["passed"] is False
    assert any("after replay runtime" in issue for issue in replay["issues"])

    missing_reasoning = deepcopy(record)
    missing_reasoning["turns"][0]["assistant_message"]["reasoning_content"] = None
    missing_reasoning["turns"][1]["model_input"][2]["reasoning_content"] = None
    assert not audit_record(missing_reasoning)["passed"]
    assert not fresh_replay_record(missing_reasoning, task)["passed"]

    tampered = deepcopy(record)
    tampered["turns"][-1]["model_input"][-1]["content"] = (
        "FUTURE GOLD SQL: SELECT COUNT(*) FROM items; answer=2"
    )
    replay = fresh_replay_record(tampered, task)
    assert not replay["passed"]
    assert any("current context differs" in issue for issue in replay["issues"])

    checkpoint_tampered = deepcopy(record)
    checkpoint_tampered["final_runtime"]["active_checkpoint_path"] = ["root", "bogus"]
    checkpoint_tampered["final_runtime"]["checkpoint_count"] = 999
    checkpoint_tampered["turns"][1]["phase_id_after"] = "evil_phase"
    checkpoint_tampered["turns"][1]["checkpoint_id_after"] = "evil_checkpoint"
    replay = fresh_replay_record(checkpoint_tampered, task)
    assert not replay["passed"]
    assert any(
        "phase_id_after differs" in issue or "final_runtime differs" in issue
        for issue in replay["issues"]
    )


def test_checkpoint_stress_guidance_is_identity_bound_and_replayable(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [("describe_table", {"tables": ["items"]})],
            [
                (
                    "commit_checkpoint",
                    {
                        "progress_summary": ["Confirmed the item population."],
                        "remaining_uncertainties": ["The final count remains."],
                        "next_targets": ["Construct the exact count relation."],
                    },
                )
            ],
            [("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM items"})],
            [("answer", {"table": "sql_001"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        checkpoint_guidance_profile="checkpoint-stress-v1",
        runtime_config=RuntimeConfig(),
        max_model_turns=8,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["correct"] and record["legal"]
    assert record["checkpoint_guidance_profile"] == "checkpoint-stress-v1"
    assert "TEACHER CHECKPOINT STRESS GUIDANCE" in client.requests[0][0]["content"]
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]

    forged = deepcopy(record)
    forged["checkpoint_guidance_profile"] = "adaptive-v1"
    assert not audit_record(forged)["passed"]


@pytest.mark.parametrize(
    ("profile", "heading"),
    [
        ("restore-trigger-v1", "TEACHER RESTORE-TRIGGER DIAGNOSTIC GUIDANCE"),
        ("restore-target-v2", "TEACHER RESTORE-TARGET V2 DIAGNOSTIC GUIDANCE"),
        ("restore-probe-v3", "TEACHER RESTORE-PROBE V3 DIAGNOSTIC GUIDANCE"),
    ],
)
def test_restore_guidance_executes_and_replays_a_recovery_branch(
    tmp_path, profile, heading
):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [("describe_table", {"tables": ["items"]})],
            [("execute_sql", {"sql": "SELECT id FROM items ORDER BY id"})],
            [
                (
                    "commit_checkpoint",
                    {
                        "progress_summary": ["Confirmed the item population."],
                        "remaining_uncertainties": ["The final count remains."],
                        "next_targets": ["Construct the exact count relation."],
                    },
                )
            ],
            [("execute_sql", {"sql": "SELECT id FROM items WHERE id = 1"})],
            [
                (
                    "restore_checkpoint",
                    {
                        "checkpoint_id": "checkpoint_001",
                        "reason": "The branch used the wrong output grain.",
                        "next_targets": ["Build one grounded count relation."],
                    },
                )
            ],
            [("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM items"})],
            [("answer", {"table": "sql_003"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        checkpoint_guidance_profile=profile,
        runtime_config=RuntimeConfig(),
        max_model_turns=10,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["correct"] and record["legal"]
    assert record["checkpoint_count"] == 2
    assert record["restore_count"] == 1
    assert record["checkpoint_guidance_profile"] == profile
    assert heading in client.requests[0][0]["content"]
    assert record["turns"][4]["result"]["phase_transition"] == "restore"
    assert record["turns"][4]["provider_phase_history_reset"] is True
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]

    forged = deepcopy(record)
    forged["checkpoint_guidance_profile"] = "adaptive-v1"
    assert not audit_record(forged)["passed"]


def test_atomic_restore_probe_discards_the_disposable_branch(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [("describe_table", {"tables": ["items"]})],
            [
                (
                    "filter_rows",
                    {
                        "table": "items",
                        "conditions": {
                            "op": ">=",
                            "left": {"column": "id"},
                            "right": {"value": 1},
                        },
                    },
                )
            ],
            [
                (
                    "commit_checkpoint",
                    {
                        "progress_summary": ["Confirmed the item population."],
                        "remaining_uncertainties": ["The final count remains."],
                        "next_targets": ["Run the restore probe."],
                    },
                )
            ],
            [
                (
                    "project",
                    {
                        "table": "filter_001",
                        "outputs": [{"expression": {"column": "id"}}],
                    },
                )
            ],
            [
                (
                    "project",
                    {
                        "table": "project_002",
                        "outputs": [
                            {
                                "expression": {
                                    "column": "__restore_probe_missing_column__"
                                }
                            }
                        ],
                    },
                )
            ],
            [
                (
                    "restore_checkpoint",
                    {
                        "checkpoint_id": "checkpoint_001",
                        "reason": "The deliberate probe invalidated the disposable branch.",
                        "next_targets": ["Build the grounded count relation."],
                    },
                )
            ],
            [
                (
                    "aggregate",
                    {
                        "table": "filter_001",
                        "group_by": [],
                        "metrics": [{"op": "count", "column": "*", "as": "n"}],
                    },
                )
            ],
            [("answer", {"table": "aggregate_004"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="atomic",
        client=client,
        checkpoint_guidance_profile="restore-probe-v3",
        runtime_config=RuntimeConfig(),
        max_model_turns=10,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["correct"] and record["legal"]
    assert record["restore_count"] == 1
    assert record["turns"][4]["result"]["error"]["code"] == "unknown_column"
    assert record["turns"][5]["result"]["phase_transition"] == "restore"
    recovery = record["final_runtime"]["checkpoint_history"][-1]
    assert recovery["created_by"] == "restore_checkpoint"
    assert recovery["snapshot"]["active_artifact_ids"] == ["filter_001"]
    assert "project_002" not in record["turns"][6]["model_input"][-1]["content"]
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]


def test_multiple_native_calls_are_one_state_preserving_semantic_error(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [
                ("describe_table", {"tables": ["items"]}),
                ("read_rows", {"table": "items"}),
            ],
            [("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM items"})],
            [("answer", {"table": "sql_001"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        runtime_config=RuntimeConfig(),
        max_model_turns=6,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    first = record["turns"][0]
    assert first["result"]["status"] == "error"
    assert first["result"]["error"]["code"] == "invalid_tool_call_count"
    assert first["environment_state_hash_before"] == first["environment_state_hash_after"]
    assert len(first["tool_result_messages"]) == 2
    assert record["correct"]

    tampered = deepcopy(record)
    tampered["turns"][0]["result"]["attempted_action"] = {
        "tool": "answer",
        "arguments": {"table": "future"},
    }
    for message in tampered["turns"][0]["tool_result_messages"]:
        message["content"] = __import__("json").dumps(
            tampered["turns"][0]["result"], separators=(",", ":")
        )
    assert not audit_record(tampered)["passed"]
    assert not fresh_replay_record(tampered, task)["passed"]


def test_native_rejection_and_primitive_budget_are_audited_separately(tmp_path):
    task = _task(tmp_path)
    client = FakeClient(
        [
            [("describe_table", {"tables": ["items"]})],
            [
                ("describe_table", {"tables": ["items"]}),
                ("read_rows", {"table": "items"}),
            ],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        runtime_config=RuntimeConfig(max_primitive_calls=1),
        max_model_turns=4,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    last = record["turns"][-1]
    assert last["native_rejection"]["code"] == "invalid_tool_call_count"
    assert last["result"]["error"]["code"] == "primitive_call_limit_reached"
    assert audit_record(record)["passed"]
    assert fresh_replay_record(record, task)["passed"]


def test_hidden_verifier_failure_preserves_the_completed_causal_trajectory(tmp_path):
    task = _task(tmp_path)
    task["gold_sql"] = "SELECT missing_column FROM items"
    client = FakeClient(
        [
            [("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM items"})],
            [("answer", {"table": "sql_001"})],
        ]
    )
    record = run_episode(
        task,
        task_position=0,
        mode="direct",
        client=client,
        runtime_config=RuntimeConfig(),
        max_model_turns=4,
        max_tokens=128,
        max_completion_tokens=256,
        api_retries=2,
    )
    assert record["legal"]
    assert not record["correct"]
    assert record["failure_type"] == "hidden_verifier_error"
    assert record["scorer_error_type"] == "OperationalError"
    assert "gold_sql" not in record
    assert record["turns"][-1]["result"]["status"] == "success"


def test_phase_style_classification():
    assert _phase_execution_styles(
        [
            {"phase_id_before": "p", "action": {"tool": "execute_sql"}},
            {"phase_id_before": "p", "action": {"tool": "project"}},
        ]
    ) == {"p": "sql_to_atomic"}
