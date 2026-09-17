#!/usr/bin/env python3
"""Run the frozen version26 pass@1 evaluator with a bounded SQLite tool call.

The historical version26 runner has no tool-execution timeout.  Formal matched
evaluation requires the same ten-second boundary on both arms, but the frozen
``eval/sft/harness`` tree must remain byte-for-byte unchanged.  This wrapper
therefore imports that exact runner and replaces only its local ``execute_tool``
reference with a SQLite progress-handler boundary.  A timeout rolls back the
Harness and causal context snapshots before the frozen runner records the error.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


EXPECTED_TIMEOUT_SECONDS = 10.0
RUNTIME_ENV = "TABLE_AGENT_PROTOCOL_RUNTIME_ROOT"
TIMEOUT_ENV = "FORMAL_TOOL_EXECUTION_TIMEOUT_SECONDS"
FEEDBACK_ENV = "ERROR_FEEDBACK_VERSION"
_MISSING = object()


class ToolExecutionTimeoutError(RuntimeError):
    """A SQLite-backed tool exceeded the formal matched-evaluation limit."""


def bounded_execute_tool(
    original: Callable[..., Any],
    harness: Any,
    tool: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
    step_id: str,
    table_output_rows: int = 0,
    *,
    timeout_seconds: float = EXPECTED_TIMEOUT_SECONDS,
) -> Any:
    """Execute one frozen tool call and restore causal state after interruption."""

    if timeout_seconds <= 0:
        raise ValueError("tool execution timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    views_before = dict(harness.views)
    sequence_before = harness._n
    context_before = {
        key: deepcopy(context[key]) if key in context else _MISSING
        for key in ("history", "handle_to_step", "environment")
    }
    interrupted: sqlite3.OperationalError | None = None
    harness.conn.set_progress_handler(
        lambda: 1 if time.monotonic() >= deadline else 0,
        1_000,
    )
    try:
        return original(
            harness,
            tool,
            arguments,
            context,
            step_id,
            table_output_rows,
        )
    except sqlite3.OperationalError as exc:
        if "interrupted" not in str(exc).casefold():
            raise
        interrupted = exc
    finally:
        harness.conn.set_progress_handler(None, 0)

    try:
        harness.conn.rollback()
    finally:
        harness.views = views_before
        harness._n = sequence_before
        for key, value in context_before.items():
            if value is _MISSING:
                context.pop(key, None)
            else:
                context[key] = value
    raise ToolExecutionTimeoutError(
        f"SQLite tool execution exceeded {timeout_seconds:g}s"
    ) from interrupted


def import_frozen_runner():
    """Import the pinned runner without adding execution or feedback policies."""
    runtime_value = os.environ.get(RUNTIME_ENV)
    if not runtime_value:
        raise RuntimeError(f"{RUNTIME_ENV} is required")
    runtime = Path(runtime_value).resolve()
    runner_path = runtime / "src/eval/rollout_passk.py"
    if not runner_path.is_file():
        raise RuntimeError(f"frozen rollout runner is missing: {runner_path}")
    sys.path[:0] = [
        str(runtime / "src/eval"),
        str(runtime / "src/harness"),
        str(runtime / "src/sft"),
    ]
    import rollout_passk as frozen_runner  # noqa: PLC0415

    imported = Path(frozen_runner.__file__).resolve()
    if imported != runner_path.resolve():
        raise RuntimeError(
            f"rollout_passk escaped frozen runtime: {imported} != {runner_path.resolve()}"
        )
    return frozen_runner


def _load_frozen_runner():
    timeout_value = float(os.environ.get(TIMEOUT_ENV, str(EXPECTED_TIMEOUT_SECONDS)))
    if timeout_value != EXPECTED_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"formal timeout is fixed at {EXPECTED_TIMEOUT_SECONDS:g}s, got {timeout_value:g}s"
        )
    frozen_runner = import_frozen_runner()
    original = frozen_runner.execute_tool

    def execute_with_timeout(
        harness,
        tool,
        arguments,
        context,
        step_id,
        table_output_rows=0,
    ):
        return bounded_execute_tool(
            original,
            harness,
            tool,
            arguments,
            context,
            step_id,
            table_output_rows,
            timeout_seconds=timeout_value,
        )

    frozen_runner.execute_tool = execute_with_timeout
    feedback_version = os.environ.get(FEEDBACK_ENV, "legacy")
    if feedback_version != "legacy":
        # Load only the explicit feedback adapter from the controller tree; the
        # parser, executor, prompt and evaluator remain from the frozen runtime.
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from rl.runtime.error_feedback import FEEDBACK_VERSION
        from rl.evaluation.runners.feedback_overlay import install_feedback_overlay
        if feedback_version != FEEDBACK_VERSION:
            raise RuntimeError(f"unknown feedback variant: {feedback_version}")
        install_feedback_overlay(frozen_runner, sys.modules["protocol"])
    return frozen_runner


def main() -> int:
    return int(_load_frozen_runner().main())


if __name__ == "__main__":
    raise SystemExit(main())
