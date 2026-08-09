from __future__ import annotations

from copy import deepcopy
import json
import sqlite3

from tool_modules.checkpoint_relalg.audit import (
    audit_record,
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
                "tool_result_messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": json.dumps(result),
                    }
                ],
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
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": json.dumps(result),
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_2",
                        "content": json.dumps(result),
                    },
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
