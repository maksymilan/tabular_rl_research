#!/usr/bin/env python3
"""Isolated atomic version26 environment for the frozen Qwen3 RL control.

This module intentionally targets the frozen version26 ``eval/sft/harness``
runtime.  It must not grow compatibility branches for later atomic protocols.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import protocol as v26_protocol
import rollout as v26_rollout
from executor import Harness
from tool_modules.registry import ATOMIC_TOOL_SCHEME, build_atomic_tool_scheme
from rl.runtime.error_feedback import (
    FEEDBACK_VERSION, LEGACY_FEEDBACK_VERSION, error_feedback_payload, merge_error_feedback,
    attach_strict_rejected_action,
)


ENVIRONMENT_IMPLEMENTATION = "atomic-v26-isolated-v1"
TOOL_EXECUTION_TIMEOUT_SECONDS = 10.0
# Runtime-only transport repair for carrier-shaped slips (unclosed <think>, markdown fence,
# extra prose around the action object).  OFF by default so every historical arm and every
# matched evaluation keeps the strict v26 carrier semantics; an arm opts in through the
# trainer's `--carrier-repair` flag (recorded in the run manifest) or CARRIER_REPAIR=1.  The
# repair never changes tool/argument semantics (repaired text is re-parsed strictly) and is
# recorded per turn as `carrier_repair`.
#
# The repair lives in the protocol module, so it exists only when the active version26
# runtime carries `parse_assistant_strict_with_repair`.  The pinned frozen runtime
# (commit 4cd47c95...) does not, therefore the default path here stays on the plain
# `parse_assistant_strict` call; requesting repair against a runtime without the API
# fails closed instead of silently degrading into per-step execution errors.
_CARRIER_REPAIR_ENABLED = os.environ.get("CARRIER_REPAIR", "0").strip() == "1"


def carrier_repair_supported() -> bool:
    """Whether the active protocol runtime implements the repair-capable parser."""

    return hasattr(v26_protocol, "parse_assistant_strict_with_repair")


def require_carrier_repair_support() -> None:
    if not carrier_repair_supported():
        raise RuntimeError(
            "carrier repair was requested but the active version26 protocol runtime does "
            f"not implement parse_assistant_strict_with_repair: {v26_protocol.__file__}"
        )


def carrier_repair_enabled() -> bool:
    return _CARRIER_REPAIR_ENABLED


def set_carrier_repair_enabled(value: bool) -> None:
    global _CARRIER_REPAIR_ENABLED
    if value:
        # Validate before mutating so a refused opt-in leaves the flag off.
        require_carrier_repair_support()
    _CARRIER_REPAIR_ENABLED = bool(value)

if v26_protocol.PROTOCOL_VERSION != "version26":
    raise ImportError(
        "tool_environment_v26 requires the frozen version26 protocol, got "
        f"{v26_protocol.PROTOCOL_VERSION!r}"
    )
_PROTOCOL_RUNTIME_ROOT = Path(v26_protocol.__file__).resolve().parents[2]
_EVAL_RUNTIME_ROOT = Path(v26_rollout.__file__).resolve().parents[2]
if _PROTOCOL_RUNTIME_ROOT != _EVAL_RUNTIME_ROOT:
    raise ImportError(
        "version26 protocol and evaluator came from different runtime roots: "
        f"{_PROTOCOL_RUNTIME_ROOT} != {_EVAL_RUNTIME_ROOT}"
    )


class ToolExecutionTimeoutError(RuntimeError):
    """A SQLite operation exceeded the local RL execution deadline."""

    code = "tool_execution_timeout"
    failure_type = "timeout_error"

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"SQLite tool execution exceeded {timeout_seconds:g}s")
        self.details = {
            "execution_engine": "sqlite",
            "timeout_seconds": float(timeout_seconds),
            "state_preserved": True,
        }


def _temp_table_names(harness: Harness) -> set[str]:
    return {
        str(row[0])
        for row in harness.conn.execute(
            "SELECT name FROM temp.sqlite_master WHERE type='table'"
        )
    }


@contextmanager
def bounded_harness_execution(
    harness: Harness,
    timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
):
    """Bound SQLite locally without modifying the frozen evaluator runtime."""

    if timeout_seconds <= 0:
        raise ValueError("tool execution timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    views_before = dict(harness.views)
    sequence_before = harness._n
    row_counts_before = dict(getattr(harness, "_row_counts", {}))
    materialized_before = set(getattr(harness, "_materialized_handles", set()))
    temp_tables_before = _temp_table_names(harness)
    timeout_cause: sqlite3.OperationalError | None = None
    harness.conn.set_progress_handler(
        lambda: 1 if time.monotonic() >= deadline else 0,
        1_000,
    )
    try:
        yield
    except sqlite3.OperationalError as exc:
        if "interrupted" not in str(exc).lower():
            raise
        timeout_cause = exc
    finally:
        harness.conn.set_progress_handler(None, 0)

    if timeout_cause is None:
        return
    harness.conn.rollback()
    harness.views = views_before
    harness._n = sequence_before
    if hasattr(harness, "_row_counts"):
        harness._row_counts = row_counts_before
    if hasattr(harness, "_materialized_handles"):
        harness._materialized_handles = materialized_before
    for table_name in _temp_table_names(harness) - temp_tables_before:
        quoted = table_name.replace('"', '""')
        harness.conn.execute(f'DROP TABLE IF EXISTS temp."{quoted}"')
    raise ToolExecutionTimeoutError(timeout_seconds) from timeout_cause


@dataclass
class EnvStep:
    done: bool
    observation: str | None
    turn: dict[str, Any]
    correct: bool = False
    legal: bool = False
    failure_type: str | None = None


class ToolUseEnv:
    """One exact version26 atomic episode over the frozen harness surface."""

    def __init__(
        self,
        example: dict,
        *,
        example_index: int | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        max_errors_per_type: int = v26_rollout.MAX_ERRORS_PER_TYPE,
        context_mode: str = "rolling-legal-history",
        history_turns: int = 4,
        compact_observations: bool = True,
        denotation_comparison: str = "bird-set",
        tool_execution_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
        error_feedback_version: str = LEGACY_FEEDBACK_VERSION,
    ):
        if error_feedback_version not in (LEGACY_FEEDBACK_VERSION, FEEDBACK_VERSION):
            raise ValueError(f"unknown error feedback version: {error_feedback_version}")
        self.error_feedback_version = error_feedback_version
        if carrier_repair_enabled():
            # Fail before the episode starts rather than scoring every turn as an error.
            require_carrier_repair_support()
        if denotation_comparison != "bird-set":
            raise ValueError("version26 RL requires denotation_comparison='bird-set'")
        if tool_execution_timeout_seconds <= 0:
            raise ValueError("tool execution timeout must be positive")
        self.example = example
        self.example_index = example_index
        self.system_prompt = system_prompt or v26_protocol.student_runtime_system_prompt(
            context_mode=context_mode,
            compact=False,
        )
        self.scheme = build_atomic_tool_scheme(system_prompt=self.system_prompt)
        self.max_steps = max_steps
        self.max_errors_per_type = max_errors_per_type
        self.context_mode = context_mode
        self.history_turns = history_turns
        self.compact_observations = compact_observations
        self.denotation_comparison = denotation_comparison
        self.tool_execution_timeout_seconds = tool_execution_timeout_seconds
        self.reset()

    def reset(self) -> list[dict]:
        self.close()
        self.harness = Harness(v26_rollout.task_db_path(self.example))
        self.overview = v26_rollout.overview(self.harness)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": v26_protocol.first_user_message(
                    self.overview,
                    self.example["question"],
                    self.example.get("external_knowledge"),
                ),
            },
        ]
        self.initial_messages = deepcopy(self.messages)
        self.created: set[str] = set()
        self.ctx = v26_rollout.new_ctx(self.overview)
        self.last_error: dict | None = None
        self.steps = 0
        self.errors = 0
        self.error_counts: dict[str, int] = {}
        self.error_events: list[dict[str, Any]] = []
        self.legal_history: list[dict[str, str]] = []
        self.turns: list[dict[str, Any]] = []
        self.done = False
        self.correct = False
        self.legal = False
        self.failure_type: str | None = None
        self.started = time.time()
        return deepcopy(self.messages)

    def model_messages(self) -> list[dict]:
        if self.context_mode == "rolling-legal-history":
            return v26_protocol.rolling_legal_history_messages(
                self.system_prompt,
                self.overview,
                self.example["question"],
                self.ctx["environment"].snapshot(),
                self.last_error,
                self.example.get("external_knowledge"),
                self.legal_history,
                self.history_turns,
                compact_observations=self.compact_observations,
            )
        if self.context_mode != "state-only":
            raise ValueError(f"unsupported context_mode: {self.context_mode}")
        return v26_protocol.model_context_messages(
            self.system_prompt,
            self.overview,
            self.example["question"],
            self.ctx["environment"].snapshot(),
            self.last_error,
            self.example.get("external_knowledge"),
        )

    def record(self) -> dict:
        return {
            **self.scheme.manifest_fields(),
            "environment_implementation": ENVIRONMENT_IMPLEMENTATION,
            "error_feedback_version": self.error_feedback_version,
            "example_index": self.example_index,
            "db_id": self.example["db_id"],
            "question": self.example["question"],
            "gold_sql": v26_rollout.task_gold_sql(self.example),
            "initial_model_input": deepcopy(self.initial_messages),
            "turns": deepcopy(self.turns),
            "correct": self.correct,
            "legal": self.legal,
            "steps": self.steps,
            "errors": self.errors,
            "failure_type": self.failure_type,
            "denotation_comparison": self.denotation_comparison,
            "tool_execution_timeout_seconds": self.tool_execution_timeout_seconds,
            "context_mode": self.context_mode,
            "history_turns": self.history_turns,
            "rolling_observation_style": (
                "resident" if self.compact_observations else "full"
            ),
            "error_events": deepcopy(self.error_events),
            "final_messages": deepcopy(self.messages),
            "elapsed_seconds": round(time.time() - self.started, 3),
        }

    def close(self) -> None:
        connection = getattr(getattr(self, "harness", None), "conn", None)
        if connection is not None:
            connection.close()

    def apply_model_output(self, text: str) -> EnvStep:
        if self.done:
            raise RuntimeError("episode is already done; call reset() before stepping again")
        self.steps += 1
        state_before = self.ctx["environment"].snapshot()
        step_id = f"step_{self.steps}"
        turn = {
            "turn_index": len(self.turns),
            "model_input": self.model_messages(),
            "model_output": text,
            "feedback_recovery": bool(self.last_error),
            "recovered_from_error_type": (
                (self.last_error or {}).get("error", {}).get("type")
            ),
        }
        self.messages.append({"role": "assistant", "content": text})

        try:
            if carrier_repair_enabled():
                think, tool, arguments, carrier_repair = (
                    v26_protocol.parse_assistant_strict_with_repair(
                        text,
                        allow_repair=True,
                    )
                )
            else:
                # Strict v26 carrier, byte-identical to the pinned runtime behaviour.
                think, tool, arguments = v26_protocol.parse_assistant_strict(text)
                carrier_repair = None
            turn["parsed"] = {
                "think": think,
                "tool": tool,
                "arguments": arguments,
            }
            if carrier_repair:
                # Transport-level repair (unclosed think / code fence / extra prose).  Kept
                # hidden from the model on purpose; the flag exists so audits can count how
                # much of the corpus was repaired instead of being scored as protocol errors.
                turn["carrier_repair"] = carrier_repair
            with bounded_harness_execution(
                self.harness,
                self.tool_execution_timeout_seconds,
            ):
                if tool == "answer_from_context":
                    self.correct, turn["pred_sample"], turn["gold_sample"] = (
                        v26_rollout.score(
                            self.harness,
                            v26_rollout.task_gold_sql(self.example),
                            arguments,
                            self.created,
                            denotation_comparison=self.denotation_comparison,
                        )
                    )
                else:
                    output, table_name = v26_rollout.execute_tool(
                        self.harness,
                        tool,
                        arguments,
                        self.ctx,
                        step_id,
                    )
            if tool == "answer_from_context":
                self.legal = True
                self.failure_type = None if self.correct else "wrong_answer"
                self.done = True
                self.turns.append(turn)
                return EnvStep(
                    done=True,
                    observation=None,
                    turn=turn,
                    correct=self.correct,
                    legal=True,
                    failure_type=self.failure_type,
                )

            turn["tool_output"] = output
            self.turns.append(turn)
            self.last_error = None
            if table_name:
                self.created.add(table_name)
            observation = v26_protocol.tool_output_message(step_id, output)
            self.legal_history.append({"assistant": text, "observation": observation})
            self.messages.append({"role": "user", "content": observation})
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
            return EnvStep(done=self.done, observation=observation, turn=turn)
        except Exception as exc:  # noqa: BLE001 - model failures are episode data
            self.errors += 1
            parsed = turn.get("parsed") or {}
            if isinstance(exc, ToolExecutionTimeoutError):
                error_type = exc.failure_type
            elif isinstance(exc, v26_protocol.ProtocolError):
                error_type = v26_rollout.protocol_failure_type(exc)
            else:
                error_type = "execution_error"
            error = f"{type(exc).__name__}: {exc}"
            state_after = self.ctx["environment"].snapshot()
            if (
                error_type == "execution_error"
                and v26_rollout.state_digest(state_after)
                != v26_rollout.state_digest(state_before)
            ):
                error_type = "nonrecoverable_execution_error"
            turn["execution_error"] = error
            turn["execution_error_type"] = error_type
            event = {
                "action_index": self.steps,
                "step_id": step_id,
                "error_type": error_type,
                "message": error,
                "error_code": getattr(exc, "code", type(exc).__name__),
                "state_before_hash": v26_rollout.state_digest(state_before),
                "state_after_hash": v26_rollout.state_digest(state_after),
            }
            details = getattr(exc, "details", None)
            if details:
                event["details"] = deepcopy(details)
            if parsed.get("tool"):
                event["attempted_tool"] = parsed["tool"]
                event["attempted_arguments"] = parsed.get("arguments") or {}
            turn["error_event"] = event
            self.error_events.append(event)
            self.turns.append(turn)
            observation = v26_protocol.tool_error_message(step_id, error_type, error)
            self.last_error = json.loads(observation)
            if self.error_feedback_version == FEEDBACK_VERSION:
                attach_strict_rejected_action(exc, text, v26_protocol)
                self.last_error = merge_error_feedback(self.last_error, error_feedback_payload(
                    exc, parsed=parsed, state=state_before, history=self.ctx.get("history", {}),
                ))
                observation = json.dumps(self.last_error, ensure_ascii=False, separators=(",", ":"))
                event["model_visible_feedback"] = deepcopy(self.last_error)
            if error_type == "nonrecoverable_execution_error":
                self.done = True
                self.failure_type = error_type
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            if self.error_counts[error_type] >= self.max_errors_per_type:
                self.done = True
                self.failure_type = error_type
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = self.failure_type or "max_steps"
            if not self.done:
                self.messages.append({"role": "user", "content": observation})
            return EnvStep(
                done=self.done,
                observation=observation,
                turn=turn,
                failure_type=self.failure_type or error_type,
            )


def create_tool_use_env(
    example: dict,
    *,
    tool_scheme: str = ATOMIC_TOOL_SCHEME,
    max_batch_calls: int = 5,
    **kwargs,
) -> ToolUseEnv:
    """Construct only the frozen atomic v26 environment."""

    del max_batch_calls
    if tool_scheme != ATOMIC_TOOL_SCHEME:
        raise ValueError("tool_environment_v26 supports only the atomic tool scheme")
    return ToolUseEnv(example, **kwargs)
