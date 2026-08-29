#!/usr/bin/env python3
"""Reusable closed-loop table-tool environment for RL experiments.

The trainer owns model generation. This module owns everything after a model turn:
parsing, tool execution, observation construction, scoring, and episode bookkeeping.
It intentionally reuses the exact eval protocol/harness functions so SFT, eval, and RL
do not drift.
"""
from __future__ import annotations

import json
import os
import sys
import time
import collections
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    AdjacentActionGuard,
    ProtocolError,
    first_user_message,
    model_context_messages,
    parse_assistant_strict,
    rolling_legal_history_messages,
    student_runtime_system_prompt,
    tool_error_message,
    tool_output_message,
)
from rollout import (  # noqa: E402
    MAX_ERRORS_PER_TYPE,
    TOOL_EXECUTION_TIMEOUT_SECONDS,
    bounded_harness_execution,
    execute_tool,
    new_ctx,
    overview,
    protocol_failure_type,
    score,
    state_digest,
    task_db_path,
    task_gold_sql,
    validate_tool_arguments_against_state,
)
from tool_modules.action_block.protocol import (  # noqa: E402
    BATCH_PLAN_TOOL,
    TERMINAL_TOOL,
    build_sequential_messages,
    lower_sequential_atomic_call,
    parse_batch_plan_assistant,
    prepare_simple_scalar_cell_arguments,
    publicize_simple_scalar_error,
    render_sequential_observation,
    validate_sequential_atomic_call,
)
from tool_modules.action_block.evaluator import (  # noqa: E402
    _error_event as batch_error_event,
    _error_type as batch_error_type,
    _execute_action_block,
    _top_level_error_message,
)
from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
    build_action_block_tool_scheme,
    build_atomic_tool_scheme,
    require_tool_scheme,
)


@dataclass
class EnvStep:
    """Result of applying one model output to the harness."""

    done: bool
    observation: str | None
    turn: dict[str, Any]
    correct: bool = False
    legal: bool = False
    failure_type: str | None = None


class ToolUseEnv:
    """One dataset-adapter task episode with rolling model context and harness state."""

    def __init__(
        self,
        example: dict,
        *,
        example_index: int | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        max_errors_per_type: int = MAX_ERRORS_PER_TYPE,
        context_mode: str = "rolling-legal-history",
        history_turns: int = 4,
        compact_observations: bool = True,
        denotation_comparison: str = "bird-set",
        tool_execution_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
    ):
        self.example = example
        self.example_index = example_index
        runtime_prompt = system_prompt or student_runtime_system_prompt(
            context_mode=context_mode,
            compact=False,
        )
        self.scheme = build_atomic_tool_scheme(system_prompt=runtime_prompt)
        self.system_prompt = self.scheme.system_prompt
        self.max_steps = max_steps
        self.max_errors_per_type = max_errors_per_type
        self.context_mode = context_mode
        self.history_turns = history_turns
        self.compact_observations = compact_observations
        if denotation_comparison != "bird-set":
            raise ValueError("active RL environments require denotation_comparison='bird-set'")
        self.denotation_comparison = denotation_comparison
        if tool_execution_timeout_seconds <= 0:
            raise ValueError("tool_execution_timeout_seconds must be positive")
        self.tool_execution_timeout_seconds = tool_execution_timeout_seconds
        self.reset()

    def reset(self) -> list[dict]:
        previous = getattr(self, "harness", None)
        previous_connection = getattr(previous, "conn", None)
        if previous_connection is not None:
            previous_connection.close()
        self.harness = Harness(task_db_path(self.example))
        self.overview = overview(self.harness)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": first_user_message(
                    self.overview,
                    self.example["question"],
                    self.example.get("external_knowledge"),
                ),
            },
        ]
        self.initial_messages = deepcopy(self.messages)
        self.created: set[str] = set()
        self.ctx = new_ctx(self.overview)
        self.last_error: dict | None = None
        self.steps = 0
        self.errors = 0
        self.error_counts: dict[str, int] = {}
        self.error_events: list[dict[str, Any]] = []
        self.adjacent_action_guard = AdjacentActionGuard()
        self.legal_history: list[dict[str, str]] = []
        self.turns: list[dict] = []
        self.done = False
        self.correct = False
        self.legal = False
        self.failure_type: str | None = None
        self.started = time.time()
        return deepcopy(self.messages)

    def model_messages(self) -> list[dict]:
        if self.context_mode == "rolling-legal-history":
            return rolling_legal_history_messages(
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
        return model_context_messages(
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
            "example_index": self.example_index,
            "db_id": self.example["db_id"],
            "question": self.example["question"],
            "gold_sql": task_gold_sql(self.example),
            "initial_model_input": self.initial_messages,
            "turns": self.turns,
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
        harness = getattr(self, "harness", None)
        connection = getattr(harness, "conn", None)
        if connection is not None:
            connection.close()

    def apply_model_output(self, text: str) -> EnvStep:
        """Apply one assistant message, append feedback, and return the environment transition."""
        if self.done:
            raise RuntimeError("episode is already done; call reset() before stepping again")

        self.steps += 1
        state_before = self.ctx["environment"].snapshot()
        step_id = f"step_{self.steps}"
        turn = {
            "turn_index": len(self.turns),
            "model_input": self.model_messages(),
            "feedback_recovery": bool(self.last_error),
            "recovered_from_error_type": (self.last_error or {}).get("error", {}).get("type"),
        }
        turn["model_output"] = text
        self.messages.append({"role": "assistant", "content": text})

        try:
            think, tool, args = parse_assistant_strict(
                text,
                adjacent_guard=self.adjacent_action_guard,
                step_id=step_id,
            )
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            if tool == "answer_from_context":
                validate_tool_arguments_against_state(self.harness, tool, args)
                with bounded_harness_execution(
                    self.harness,
                    self.tool_execution_timeout_seconds,
                ):
                    self.correct, turn["pred_sample"], turn["gold_sample"] = score(
                        self.harness,
                        task_gold_sql(self.example),
                        args,
                        self.created,
                        denotation_comparison=self.denotation_comparison,
                    )
                self.legal = True
                if not self.correct:
                    self.failure_type = "wrong_answer"
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

            output, table_name = execute_tool(
                self.harness,
                tool,
                args,
                self.ctx,
                step_id,
                tool_execution_timeout_seconds=self.tool_execution_timeout_seconds,
            )
            turn["tool_output"] = output
            self.adjacent_action_guard.mark_last("success")
            self.turns.append(turn)
            self.last_error = None
            if table_name:
                self.created.add(table_name)
            observation = tool_output_message(
                step_id,
                output,
            )
            self.legal_history.append({"assistant": text, "observation": observation})
            self.messages.append({"role": "user", "content": observation})
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
            return EnvStep(done=self.done, observation=observation, turn=turn)

        except Exception as exc:  # noqa: BLE001 - every model action becomes an audited transition
            self.errors += 1
            parsed = turn.get("parsed") or {}
            attempted_tool = parsed.get("tool") or getattr(exc, "attempted_tool", None)
            attempted_arguments = (
                parsed.get("arguments")
                if parsed.get("tool")
                else getattr(exc, "attempted_arguments", None)
            )
            error_type = protocol_failure_type(exc) if isinstance(exc, ProtocolError) else "execution_error"
            error = f"{type(exc).__name__}: {exc}"
            state_after = self.ctx["environment"].snapshot()
            self.adjacent_action_guard.mark_last("rejected")
            if error_type == "execution_error" and state_digest(state_after) != state_digest(state_before):
                error_type = "nonrecoverable_execution_error"
            turn["execution_error"] = error
            turn["execution_error_type"] = error_type
            event = {
                "action_index": self.steps,
                "step_id": step_id,
                "error_type": error_type,
                "message": error,
                "error_code": getattr(exc, "code", type(exc).__name__),
                "state_before_hash": state_digest(state_before),
                "state_after_hash": state_digest(state_after),
            }
            details = getattr(exc, "details", None)
            if details:
                event["details"] = deepcopy(details)
            if attempted_tool:
                event["attempted_tool"] = attempted_tool
                event["attempted_arguments"] = attempted_arguments or {}
            turn["error_event"] = event
            self.error_events.append(event)
            self.turns.append(turn)
            observation = tool_error_message(
                step_id,
                error_type,
                error,
                error_code=getattr(exc, "code", type(exc).__name__),
                details=details,
                attempted_tool=attempted_tool,
                attempted_arguments=attempted_arguments,
            )
            self.last_error = json.loads(observation)
            self.adjacent_action_guard.mark_last(
                "rejected",
                self.last_error["error"],
            )
            if error_type == "nonrecoverable_execution_error":
                self.done = True
                self.failure_type = error_type
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=error_type,
                )
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            if self.error_counts[error_type] >= self.max_errors_per_type:
                self.done = True
                self.failure_type = error_type
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=error_type,
                )
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=self.failure_type,
                )
            self.messages.append({"role": "user", "content": observation})
            return EnvStep(done=False, observation=observation, turn=turn, failure_type=error_type)


class ActionBlockToolUseEnv:
    """Result-only RL environment for the independently selectable action-block scheme."""

    def __init__(
        self,
        example: dict,
        *,
        example_index: int | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        max_batch_calls: int = 5,
        max_errors_per_type: int = MAX_ERRORS_PER_TYPE,
        context_mode: str = "rolling-legal-history",
        history_turns: int = 4,
        compact_observations: bool = True,
        denotation_comparison: str = "bird-set",
        tool_execution_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
    ):
        if context_mode != "rolling-legal-history":
            raise ValueError(
                "action-block RL currently requires rolling-legal-history"
            )
        if history_turns != 4:
            raise ValueError(
                "active action-block context retains exactly four action blocks"
            )
        if denotation_comparison != "bird-set":
            raise ValueError(
                "active RL environments require denotation_comparison='bird-set'"
            )
        self.scheme = build_action_block_tool_scheme(
            max_batch_calls=max_batch_calls,
        )
        self.example = example
        self.example_index = example_index
        self.system_prompt = system_prompt or self.scheme.system_prompt
        self.max_action_blocks = max_steps
        self.max_batch_calls = max_batch_calls
        self.max_errors_per_type = max_errors_per_type
        self.context_mode = context_mode
        self.history_turns = history_turns
        self.compact_observations = compact_observations
        self.denotation_comparison = denotation_comparison
        if tool_execution_timeout_seconds <= 0:
            raise ValueError("tool_execution_timeout_seconds must be positive")
        self.tool_execution_timeout_seconds = tool_execution_timeout_seconds
        self.reset()

    def reset(self) -> list[dict]:
        previous = getattr(self, "harness", None)
        previous_connection = getattr(previous, "conn", None)
        if previous_connection is not None:
            previous_connection.close()
        self.harness = Harness(task_db_path(self.example))
        self.overview = overview(self.harness)
        self.created: set[str] = set()
        self.ctx = new_ctx(self.overview)
        self.last_error: dict | None = None
        self.legal_history: list[dict[str, str]] = []
        self.turns: list[dict] = []
        self.atomic_events: list[dict] = []
        self.error_events: list[dict] = []
        self.interface_resolution_events: list[dict] = []
        self.error_counts: collections.Counter = collections.Counter()
        self.pending_feedback_recovery = False
        self.model_turns = 0
        self.atomic_actions = 0
        self.action_blocks = 0
        self.submitted_calls = 0
        self.blocked_nodes = 0
        self.done = False
        self.correct = False
        self.legal = False
        self.failure_type: str | None = None
        self.started = time.time()
        self.initial_messages = self.model_messages()
        return deepcopy(self.initial_messages)

    def model_messages(self) -> list[dict]:
        return build_sequential_messages(
            system_prompt=self.system_prompt,
            overview=self.overview,
            question=self.example["question"],
            external_knowledge=self.example.get("external_knowledge"),
            state=self.ctx["environment"].snapshot(),
            last_error=self.last_error,
            legal_history=self.legal_history,
            history_turns=self.history_turns,
        )

    @property
    def steps(self) -> int:
        """Compatibility alias: one RL optimization turn is one model action block."""
        return self.model_turns

    @steps.setter
    def steps(self, value: int) -> None:
        self.model_turns = value

    @property
    def errors(self) -> int:
        return len(self.error_events)

    def record(self) -> dict:
        return {
            **self.scheme.manifest_fields(),
            "example_index": self.example_index,
            "db_id": self.example["db_id"],
            "question": self.example["question"],
            "gold_sql": task_gold_sql(self.example),
            "initial_model_input": deepcopy(self.initial_messages),
            "turns": deepcopy(self.turns),
            "atomic_events": deepcopy(self.atomic_events),
            "error_events": deepcopy(self.error_events),
            "interface_resolution_events": deepcopy(
                self.interface_resolution_events
            ),
            "correct": self.correct,
            "legal": self.legal,
            "steps": self.model_turns,
            "model_turns": self.model_turns,
            "atomic_actions": self.atomic_actions,
            "action_blocks": self.model_turns,
            "executed_action_blocks": self.action_blocks,
            "submitted_calls": self.submitted_calls,
            "blocked_nodes": self.blocked_nodes,
            "errors": self.errors,
            "failure_type": self.failure_type,
            "denotation_comparison": self.denotation_comparison,
            "tool_execution_timeout_seconds": self.tool_execution_timeout_seconds,
            "context_mode": self.context_mode,
            "history_turns": self.history_turns,
            "rolling_observation_style": "full-atomic-results-plus-resident-state",
            "final_environment_state": self.ctx["environment"].snapshot(),
            "elapsed_seconds": round(time.time() - self.started, 3),
        }

    def close(self) -> None:
        connection = getattr(getattr(self, "harness", None), "conn", None)
        if connection is not None:
            connection.close()

    def _finish_if_budget_exhausted(self) -> None:
        if self.done:
            return
        if self.model_turns >= self.max_action_blocks:
            self.done = True
            self.failure_type = "max_action_blocks"

    def apply_model_output(self, text: str) -> EnvStep:
        if self.done:
            raise RuntimeError("episode is already done; call reset() before stepping again")
        self.model_turns += 1
        state_before = self.ctx["environment"].snapshot()
        turn = {
            "turn_index": len(self.turns),
            "model_input": self.model_messages(),
            "model_output": text,
            "feedback_recovery": self.pending_feedback_recovery,
            "recovered_from_error_type": (
                (self.last_error or {}).get("error") or {}
            ).get("type"),
        }
        try:
            reasoning, tool, arguments = parse_batch_plan_assistant(
                text,
                max_batch_calls=self.max_batch_calls,
            )
            turn["parsed"] = {
                "think": reasoning,
                "tool": tool,
                "arguments": deepcopy(arguments),
            }
            if tool == TERMINAL_TOOL:
                self.atomic_actions += 1
                self.submitted_calls += 1
                step_id = f"step_{self.atomic_actions}"
                with bounded_harness_execution(
                    self.harness,
                    self.tool_execution_timeout_seconds,
                ):
                    self.correct, turn["pred_sample"], turn["gold_sample"] = score(
                        self.harness,
                        task_gold_sql(self.example),
                        arguments,
                        self.created,
                        denotation_comparison=self.denotation_comparison,
                    )
                self.legal = True
                self.done = True
                self.failure_type = None if self.correct else "wrong_answer"
                scored_output = {
                    "correct": self.correct,
                    "pred_sample": turn["pred_sample"],
                    "gold_sample": turn["gold_sample"],
                }
                terminal_event = {
                    "call_id": "__answer__",
                    "step_id": step_id,
                    "tool": TERMINAL_TOOL,
                    "status": "success",
                    "arguments": deepcopy(arguments),
                    "resolved_arguments": deepcopy(arguments),
                    "resolved_evidence_arguments": deepcopy(arguments),
                    "output": deepcopy(scored_output),
                    "environment_state_before": state_before,
                    "environment_state": self.ctx["environment"].snapshot(),
                }
                self.atomic_events.append(terminal_event)
                turn["terminal_result"] = deepcopy(scored_output)
                self.turns.append(turn)
                return EnvStep(
                    done=True,
                    observation=None,
                    turn=turn,
                    correct=self.correct,
                    legal=True,
                    failure_type=self.failure_type,
                )
            if tool != BATCH_PLAN_TOOL:
                raise RuntimeError(f"unexpected action-block tool: {tool}")
            proposed = len(arguments["calls"])
            self.action_blocks += 1
            self.submitted_calls += proposed
            (
                self.atomic_actions,
                results,
                events,
                nonrecoverable,
            ) = _execute_action_block(
                h=self.harness,
                ctx=self.ctx,
                arguments=arguments,
                created=self.created,
                atomic_count=self.atomic_actions,
                model_turn=self.model_turns,
                batch_index=self.action_blocks,
                table_output_rows=0,
                error_counts=self.error_counts,
                error_events=self.error_events,
                prior_bindings=None,
                structured_error_feedback=False,
                low_friction_interface=False,
                safe_low_friction_interface=False,
                interface_resolution_events=self.interface_resolution_events,
                validate_call=validate_sequential_atomic_call,
                prepare_call_arguments=prepare_simple_scalar_cell_arguments,
                lower_call=lower_sequential_atomic_call,
                publicize_error=publicize_simple_scalar_error,
            )
            terminal_result = next(
                (
                    result
                    for result in results
                    if result.get("tool") == TERMINAL_TOOL
                ),
                None,
            )
            if (
                terminal_result is not None
                and terminal_result.get("status") == "success"
            ):
                score_arguments = terminal_result["terminal_score_arguments"]
                with bounded_harness_execution(
                    self.harness,
                    self.tool_execution_timeout_seconds,
                ):
                    self.correct, turn["pred_sample"], turn["gold_sample"] = score(
                        self.harness,
                        task_gold_sql(self.example),
                        score_arguments,
                        self.created,
                        denotation_comparison=self.denotation_comparison,
                    )
                self.legal = True
                self.done = True
                self.failure_type = None if self.correct else "wrong_answer"
                scored_output = {
                    "correct": self.correct,
                    "pred_sample": turn["pred_sample"],
                    "gold_sample": turn["gold_sample"],
                }
                terminal_result["output"] = deepcopy(scored_output)
                for event in events:
                    if event.get("call_id") == terminal_result.get("call_id"):
                        event["output"] = deepcopy(scored_output)
                        event["resolved_evidence_arguments"] = deepcopy(
                            terminal_result["resolved_evidence_arguments"]
                        )
                        break
                self.atomic_events.extend(events)
                turn["batch_index"] = self.action_blocks
                turn["batch_results"] = deepcopy(results)
                self.turns.append(turn)
                return EnvStep(
                    done=True,
                    observation=None,
                    turn=turn,
                    correct=self.correct,
                    legal=True,
                    failure_type=self.failure_type,
                )

            self.atomic_events.extend(events)
            blocked = sum(result.get("status") == "blocked" for result in results)
            root_errors = sum(
                result.get("status") == "error" for result in results
            )
            self.blocked_nodes += blocked
            observation = render_sequential_observation(
                self.action_blocks,
                results,
            )
            turn["batch_index"] = self.action_blocks
            turn["batch_results"] = deepcopy(results)
            turn["root_error_count"] = root_errors
            turn["blocked_count"] = blocked
            turn["observation"] = observation
            self.turns.append(turn)
            self.legal_history.append({
                "assistant": text.strip(),
                "observation": observation,
            })
            self.last_error = None
            self.pending_feedback_recovery = bool(root_errors or blocked)
            if nonrecoverable:
                self.done = True
                self.failure_type = "nonrecoverable_execution_error"
            else:
                exhausted = [
                    name
                    for name, count in self.error_counts.items()
                    if count >= self.max_errors_per_type
                ]
                if exhausted:
                    self.done = True
                    self.failure_type = sorted(exhausted)[0]
            self._finish_if_budget_exhausted()
            return EnvStep(
                done=self.done,
                observation=observation,
                turn=turn,
                failure_type=self.failure_type,
            )
        except Exception as exc:  # noqa: BLE001
            state_after = self.ctx["environment"].snapshot()
            error_type = batch_error_type(exc)
            if (
                error_type == "execution_error"
                and state_digest(state_after) != state_digest(state_before)
            ):
                error_type = "nonrecoverable_execution_error"
            message = str(exc)
            self.error_counts[error_type] += 1
            parsed = turn.get("parsed") or {}
            event = batch_error_event(
                atomic_index=None,
                model_turn=self.model_turns,
                batch_index=None,
                call_id=None,
                tool=parsed.get("tool"),
                arguments=parsed.get("arguments"),
                error_type=error_type,
                message=message,
                state_before=state_before,
                state_after=state_after,
            )
            self.error_events.append(event)
            turn["execution_error_type"] = error_type
            turn["execution_error"] = message
            turn["error_event"] = event
            self.turns.append(turn)
            self.last_error = _top_level_error_message(
                block_action_index=self.model_turns,
                error_type=error_type,
                message=message,
            )
            self.pending_feedback_recovery = True
            if (
                error_type == "nonrecoverable_execution_error"
                or self.error_counts[error_type] >= self.max_errors_per_type
            ):
                self.done = True
                self.failure_type = error_type
            self._finish_if_budget_exhausted()
            observation = json.dumps(
                self.last_error,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return EnvStep(
                done=self.done,
                observation=observation,
                turn=turn,
                failure_type=error_type,
            )


def create_tool_use_env(
    example: dict,
    *,
    tool_scheme: str = ATOMIC_TOOL_SCHEME,
    max_batch_calls: int = 5,
    **kwargs,
) -> ToolUseEnv | ActionBlockToolUseEnv:
    """Construct one environment without exposing both action spaces to the model."""
    require_tool_scheme(tool_scheme)
    if tool_scheme == ATOMIC_TOOL_SCHEME:
        return ToolUseEnv(example, **kwargs)
    if tool_scheme == ACTION_BLOCK_TOOL_SCHEME:
        return ActionBlockToolUseEnv(
            example,
            max_batch_calls=max_batch_calls,
            **kwargs,
        )
    raise ValueError(
        f"RL environment does not implement tool scheme {tool_scheme!r}; "
        "ongoing RL remains on the frozen atomic/action-block environments, while "
        "checkpoint-relalg-v1 is diagnostic-only until its admission gates pass"
    )
