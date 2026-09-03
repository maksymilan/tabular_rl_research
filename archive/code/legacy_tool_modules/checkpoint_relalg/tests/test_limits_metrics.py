from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from src.tool_modules.checkpoint_relalg.errors import CheckpointRelalgError
from src.tool_modules.checkpoint_relalg.executors import ResultSizeTracker
from src.tool_modules.checkpoint_relalg.executors.sqlite_compiler import (
    SQLiteRelationalExecutor,
)
from src.tool_modules.checkpoint_relalg.runner import (
    _hidden_reference_rows,
    _process_metrics,
    build_manifest,
    parse_args,
)
from src.tool_modules.checkpoint_relalg.runtime import (
    CheckpointRelalgRuntime,
    RuntimeConfig,
)


def _runtime(
    mode: str,
    *,
    rows: list[tuple[object, ...]],
    config: RuntimeConfig,
) -> CheckpointRelalgRuntime:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE source_rows(value TEXT)")
    connection.executemany("INSERT INTO source_rows VALUES (?)", rows)
    return CheckpointRelalgRuntime(connection, mode=mode, config=config)


@pytest.mark.parametrize(
    ("mode", "tool", "arguments"),
    [
        (
            "atomic",
            "project",
            {"table": "source_rows", "outputs": [{"expression": {"column": "value"}}]},
        ),
        ("direct", "execute_sql", {"sql": "SELECT value FROM source_rows"}),
        ("hybrid", "execute_sql", {"sql": "SELECT value FROM source_rows"}),
    ],
)
def test_artifact_byte_limit_rejects_before_publish_in_every_mode(
    mode: str,
    tool: str,
    arguments: dict[str, object],
) -> None:
    runtime = _runtime(
        mode,
        rows=[("abcd",), ("abcd",), ("abcd",)],
        config=RuntimeConfig(max_artifact_bytes=14, max_cell_bytes=10),
    )
    before = runtime.state.logical_hash()
    result = runtime.apply(tool, arguments)
    assert result["status"] == "error"
    assert result["error"]["type"] == "resource_limit_error"
    assert result["error"]["code"] == "result_too_large"
    assert runtime.state.logical_hash() == before
    assert not runtime.state.active_artifact_ids


@pytest.mark.parametrize("mode", ["direct", "atomic", "hybrid"])
def test_perception_and_answer_cell_row_limits_preserve_state(mode: str) -> None:
    runtime = _runtime(
        mode,
        rows=[("abcdefgh",), ("small",)],
        config=RuntimeConfig(
            max_artifact_rows=1,
            max_artifact_bytes=100,
            max_cell_bytes=4,
        ),
    )
    before = runtime.state.logical_hash()
    perceived = runtime.apply(
        "read_rows",
        {"table": "source_rows", "columns": ["value"], "limit": 1},
    )
    assert perceived["error"]["code"] == "result_too_large"
    assert perceived["error"]["details"]["limit_kind"] == "cell_bytes"
    assert not runtime.state.active_observation_ids
    assert runtime.state.logical_hash() == before

    answered = runtime.apply("answer", {"table": "source_rows"})
    assert answered["error"]["code"] == "result_too_large"
    assert runtime.state.logical_hash() == before
    assert not runtime.done


def test_canonical_byte_accounting_is_utf8_and_streaming() -> None:
    tracker = ResultSizeTracker(
        max_rows=2,
        max_artifact_bytes=100,
        max_cell_bytes=3,
        operation="test",
    )
    with pytest.raises(CheckpointRelalgError) as captured:
        tracker.add_row(("éé",))
    assert captured.value.code == "result_too_large"
    assert captured.value.details["observed_cell_bytes"] == 4


def test_direct_deadline_covers_post_fetch_compiler_work(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(
        "direct",
        rows=[("ok",)],
        config=RuntimeConfig(sql_timeout_seconds=0.02),
    )
    original = runtime.executor._infer_direct_columns

    def slow_inference(*args, **kwargs):
        time.sleep(0.04)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.executor, "_infer_direct_columns", slow_inference)
    before = runtime.state.logical_hash()
    result = runtime.apply("execute_sql", {"sql": "SELECT value FROM source_rows"})
    assert result["error"]["code"] == "sql_timeout"
    assert runtime.state.logical_hash() == before
    assert not runtime.state.active_artifact_ids


def test_hidden_reference_is_streamed_through_same_resource_limits(tmp_path: Path) -> None:
    path = tmp_path / "hidden.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t(x TEXT)")
    connection.executemany("INSERT INTO t VALUES (?)", [("a",), ("b",)])
    connection.commit()
    connection.close()
    task = {"db_path": str(path), "gold_sql": "SELECT x FROM t"}
    with pytest.raises(Exception) as captured:
        _hidden_reference_rows(
            task,
            runtime_config=RuntimeConfig(
                max_artifact_rows=1,
                max_artifact_bytes=100,
                max_cell_bytes=10,
            ),
        )
    assert getattr(captured.value, "code") == "result_too_large"
    assert getattr(captured.value, "details")["operation"] == "hidden_reference"


def test_manifest_and_process_metrics_are_explicit(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--mode",
            "hybrid",
            "--result-dir",
            str(tmp_path / "result"),
            "--max-artifact-bytes",
            "2048",
            "--max-cell-bytes",
            "512",
        ]
    )
    manifest = build_manifest(args, provider_verification={"verified": True})
    for field in (
        "capability_manifest",
        "backend",
        "dialect",
        "environment_renderer_version",
        "checkpoint_policy_version",
        "executor_version",
    ):
        assert manifest[field]
    assert manifest["runtime_config"]["max_artifact_bytes"] == 2048
    assert manifest["runtime_config"]["max_cell_bytes"] == 512

    turns = [
        {
            "phase_id_before": "phase_000",
            "checkpoint_id_before": "root",
            "action": {"tool": "execute_sql"},
            "result": {"status": "success"},
        },
        {
            "phase_id_before": "phase_001",
            "checkpoint_id_before": "checkpoint_001",
            "action": {"tool": "project"},
            "result": {"status": "error", "error": {"code": "unknown_column"}},
        },
        {
            "phase_id_before": "phase_002",
            "checkpoint_id_before": "checkpoint_002",
            "result": {"status": "error", "error": {"code": "invalid_tool_call_count"}},
        },
    ]
    metrics = _process_metrics(
        turns,
        final_runtime={
            "active_checkpoint_path": ["root", "checkpoint_002"],
            "checkpoint_count": 2,
            "restore_count": 1,
        },
        mode="hybrid",
        failure_type="context_length_exceeded",
        provider_usage={"prompt_tokens": 123},
    )
    assert metrics["phase_lengths"] == {
        "phase_000": 1,
        "phase_001": 1,
        "phase_002": 1,
    }
    assert metrics["active_path_length"] == 2
    assert metrics["abandoned_path_cost"] == 1
    assert metrics["history_tokens"] is None
    assert metrics["total_prompt_tokens"] == 123
    assert metrics["context_overflow_rate"] == 1.0
