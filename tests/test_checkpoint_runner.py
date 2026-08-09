from __future__ import annotations

import sqlite3
from pathlib import Path
from copy import deepcopy

from tool_modules.checkpoint_relalg.provider import NativeAssistantResponse, _normalize_assistant_message
from tool_modules.checkpoint_relalg.audit import audit_record, fresh_replay_record
from tool_modules.checkpoint_relalg.runner import _phase_execution_styles, run_episode
from tool_modules.checkpoint_relalg.runtime import RuntimeConfig


class FakeClient:
    request_audit_options = {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "fake",
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
        return NativeAssistantResponse(
            message=message,
            calls=calls,
            finish_reason="tool_calls",
            usage={"total_tokens": 10},
            response_metadata={"model": "fake"},
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
