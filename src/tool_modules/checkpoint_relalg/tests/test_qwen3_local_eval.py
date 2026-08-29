from __future__ import annotations

import sqlite3
from pathlib import Path

from tool_modules.checkpoint_relalg.qwen3_local_eval import (
    LocalCompletion,
    build_qwen3_model_input,
    fresh_replay_record,
    run_episode,
)


class FakeClient:
    model = "checkpoint-250"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, max_tokens: int) -> LocalCompletion:
        self.calls.append([dict(item) for item in messages])
        text = self.outputs.pop(0)
        return LocalCompletion(
            text=text,
            finish_reason="stop",
            response_model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            attempts=1,
            elapsed_seconds=0.01,
        )


def _task(tmp_path: Path) -> dict:
    path = tmp_path / "toy.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE items(x INTEGER); INSERT INTO items VALUES (1),(2);")
    connection.close()
    return {
        "example_id": "toy_001",
        "example_index": 7,
        "db_id": "toy",
        "db_path": str(path),
        "question": "Return every x value.",
        "external_knowledge": None,
        "gold_sql": "SELECT x FROM items",
    }


def _action(reasoning: str, tool: str, arguments: str) -> str:
    return f'<think>{reasoning}</think>{{"tool":"{tool}","arguments":{arguments}}}'


def test_build_qwen3_model_input_merges_feedback_and_current_context() -> None:
    history = [
        {"role": "user", "content": "context one"},
        {"role": "assistant", "content": '{"tool":"x","arguments":{}}'},
        {"role": "user", "content": "feedback one"},
    ]
    messages = build_qwen3_model_input(history, "context two")
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "feedback one\n\ncontext two"
    assert "<think>" not in messages[2]["content"]


def test_local_episode_uses_qwen_history_and_passes_fresh_replay(tmp_path: Path) -> None:
    task = _task(tmp_path)
    client = FakeClient(
        [
            _action("Inspect the source schema.", "describe_table", '{"tables":["items"]}'),
            _action("The source relation is the exact answer.", "answer", '{"table":"items"}'),
        ]
    )
    record = run_episode(task, task_position=0, db_root=None, client=client)
    assert record["correct"] is True
    assert record["legal"] is True
    assert record["source_atomic_operator_profile"] == "atomic-v24-frozen-v1"
    assert record["checkpoint_count"] == record["restore_count"] == 0
    assert record["audit"]["fresh_replay_passed"] is True
    assert fresh_replay_record(record, task, db_root=None) == []
    assert [item["role"] for item in client.calls[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert client.calls[1][2]["content"].startswith('{"tool":"describe_table"')
    assert "<think>" not in client.calls[1][2]["content"]
    assert "checkpoint_relalg_tool_result" in client.calls[1][3]["content"]
    assert "\n\nQUESTION" in client.calls[1][3]["content"]


def test_runtime_argument_error_remains_causal_and_recoverable(tmp_path: Path) -> None:
    task = _task(tmp_path)
    client = FakeClient(
        [
            _action("Try a table name.", "describe_table", '{"tables":["missing"]}'),
            _action("Use the catalog name.", "describe_table", '{"tables":["items"]}'),
            _action("Return the source.", "answer", '{"table":"items"}'),
        ]
    )
    record = run_episode(task, task_position=0, db_root=None, client=client)
    assert record["correct"] is True
    assert record["errors"] == 1
    assert record["turns"][0]["result"]["status"] == "error"
    assert record["audit"]["fresh_replay_passed"] is True


def test_invalid_inline_carrier_terminates_without_fabricated_action(tmp_path: Path) -> None:
    task = _task(tmp_path)
    client = FakeClient(['{"tool":"answer","arguments":{"table":"items"}}'])
    record = run_episode(task, task_position=0, db_root=None, client=client)
    assert record["failure_type"] == "local_carrier_error"
    assert record["legal"] is False
    assert "action" not in record["turns"][0]
    assert record["audit"]["fresh_replay_passed"] is True
